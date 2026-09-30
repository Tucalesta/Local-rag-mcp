import os
import time
import logging
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
from mcp.server import MCPServer
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import requests

import rag_common
from search_core import (
    PhotoSearcher,
    build_answer_prompt,
    format_overview,
    format_passages,
    format_photo_results,
    search_documents,
)

# Logger dedicato per capire dove va il tempo durante una query (embedding
# vs retrieval vs generazione LLM). Scrive su console (stderr), visibile nei
# log del processo MCP.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [RAG] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local-rag")

# Timeout (in secondi) applicati sia all'embedding che alla generazione LLM.
# Alzati rispetto al default perche' su CPU il caricamento del modello in RAM
# puo' richiedere parecchi secondi prima ancora di iniziare a generare.
OLLAMA_TIMEOUT = 300.0
OLLAMA_URL = "http://localhost:11434"

# Inizializziamo il server con la classe standard v2
mcp = MCPServer("Local RAG Server")

# Directory in cui si trova questo script: usata come base per risolvere
# i path relativi definiti in config.json, cosi' il server funziona
# indipendentemente dalla working directory da cui viene lanciato.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stesso config.json usato da ingest.py e query.py, cosi' db_path, modello di
# embedding, collezione e modelli restano coerenti tra indicizzazione e ricerca.
_config = rag_common.load_config(BASE_DIR)
db_path = rag_common.resolve_path(_config, ("db_path",), "./chroma_db", BASE_DIR)
llm_model = _config.get("llm_model", "gemma3")
embed_model = rag_common.embed_model_name(_config)
COLLECTION = rag_common.collection_name(_config)
TOP_K_DOCS = rag_common.top_k_docs(_config)
TOP_K_PHOTOS = rag_common.top_k_photos(_config)

# Configurazione di Ollama e LlamaIndex. Il modello di embedding DEVE essere lo
# stesso usato da ingest.py per costruire la collezione.
Settings.embed_model = rag_common.make_embed_model(_config, OLLAMA_TIMEOUT)
Settings.llm = Ollama(model=llm_model, request_timeout=OLLAMA_TIMEOUT)

_state = {"collection": None, "photos": None}


def _collection():
    if _state["collection"] is None:
        if not os.path.exists(db_path):
            raise RuntimeError(f"Database vettoriale non trovato in {db_path}. Esegui prima ingest.py!")
        _state["collection"] = rag_common.open_collection(db_path, COLLECTION)
    return _state["collection"]


def _embed_query(text):
    return Settings.embed_model.get_query_embedding(text)


def _photos():
    if _state["photos"] is None:
        res_dir = rag_common.resolve_path(
            _config, ("res_path", "res_dir", "gpx_dir"), rag_common.GPX_DIR_NAME, BASE_DIR)
        gpx_points = rag_common.load_gpx_database(res_dir, logger)
        _state["photos"] = PhotoSearcher(_collection(), _embed_query, gpx_points)
    return _state["photos"]


@mcp.tool()
def health_check() -> str:
    """
    Verifica rapidamente lo stato del server RAG: presenza del database
    vettoriale, numero di documenti e foto indicizzati e raggiungibilita' di
    Ollama. NON genera alcuna risposta con l'LLM: e' pensato per un controllo
    veloce (pochi secondi) senza dover caricare/scaldare i modelli.
    """
    lines = []

    # 1. Database vettoriale
    if not os.path.exists(db_path):
        lines.append(f"[X] Database non trovato in: {db_path}")
        return "\n".join(lines)
    lines.append(f"[OK] Database trovato in: {db_path}")

    try:
        collection = _collection()
        total = collection.count()
        photos = len(collection.get(where={"type": "image"}, include=[])["ids"])
        texts = total - photos
        lines.append(f"[OK] Collezione '{COLLECTION}' accessibile: {total} chunk "
                     f"({photos} foto, {texts} passaggi di documenti)")
        if total == 0:
            lines.append("[X] La collezione e' vuota: esegui ingest.py")
    except Exception as e:
        lines.append(f"[X] Errore aprendo la collezione ChromaDB: {e}")
        return "\n".join(lines)

    # 2. Ollama raggiungibile (senza generare nulla, solo elenco modelli)
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        model_names = [m["name"] for m in resp.json().get("models", [])]
        lines.append(f"[OK] Ollama raggiungibile, {len(model_names)} modelli disponibili")

        for needed in (embed_model, llm_model):
            match = [m for m in model_names if m == needed or m.split(":")[0] == needed.split(":")[0]]
            if match:
                lines.append(f"[OK] Modello '{needed}' presente ({match[0]})")
            else:
                lines.append(f"[X] Modello '{needed}' NON trovato tra quelli scaricati")
    except Exception as e:
        lines.append(f"[X] Ollama non raggiungibile su {OLLAMA_URL}: {e}")

    return "\n".join(lines)


@mcp.tool()
def search_local_docs(question: str, top_k: int = 0, answer: bool = False) -> str:
    """
    Cerca nei DOCUMENTI locali indicizzati (PDF, Word, Excel, TXT, MD) i passaggi
    piu' pertinenti a una domanda e li restituisce con le fonti. Non cerca tra le
    foto: per quelle usa find_photos.

    Per impostazione predefinita restituisce solo i passaggi, senza far scrivere
    una risposta al modello locale (piu' veloce): la risposta la compone chi
    chiama, leggendo i passaggi.

    Args:
        question: La domanda o l'argomento da cercare nei documenti.
        top_k: Quanti passaggi restituire (0 = valore di config.json, "top_k_docs").
        answer: Se True, il modello locale scrive anche una risposta basata sui passaggi (piu' lento).
    """
    t0 = time.time()
    logger.info(f"Domanda ricevuta: {question!r}")
    try:
        k = int(top_k) if top_k and top_k > 0 else TOP_K_DOCS
        passages = search_documents(_collection(), _embed_query, question, k)
        logger.info(f"Passaggi trovati in {time.time() - t0:.1f}s")
        if not passages:
            return "Nessun passaggio trovato tra i documenti indicizzati."
        if not answer:
            return "PASSAGGI TROVATI:\n" + format_passages(passages)

        response = Settings.llm.complete(build_answer_prompt(question, passages))
        logger.info(f"Risposta generata in {time.time() - t0:.1f}s")
        sources = []
        for p in passages:
            entry = f"- {p['file_name']} ({p['path']})"
            if entry not in sources:
                sources.append(entry)
        return f"RISPOSTA:\n{response.text}\n\nFONTI UTILIZZATE:\n" + "\n".join(sources)
    except Exception as e:
        logger.error(f"Fallita dopo {time.time() - t0:.1f}s: {e}")
        return f"Errore durante l'interrogazione del RAG: {str(e)}"


@mcp.tool()
def find_photos(query: str = "", location: str = "", region: str = "", year: int = 0,
                year_from: int = 0, year_to: int = 0, camera: str = "", near: str = "",
                radius_km: float = 20.0, top_k: int = 0) -> str:
    """
    Cerca FOTO nell'archivio personale, combinando la ricerca per significato
    sulla descrizione della foto con filtri esatti su luogo, anno e fotocamera.
    Usa i filtri ogni volta che la richiesta li contiene: "foto del 2014 a
    Cortina con la neve" = query="neve", location="Cortina", year=2014.
    Restituisce, per ogni foto, percorso del file, data, luogo, coordinate,
    fotocamera e descrizione (nessun modello locale genera testo: piu' veloce).

    Args:
        query: Cosa si vede nella foto, in italiano ("cascata di ghiaccio", "castello nella nebbia"). Puo' essere vuoto se si usano solo filtri.
        location: Luogo in cui e' stata scattata, anche parziale (es. "Cortina").
        region: Regione (es. "Veneto").
        year: Anno preciso. Alternativa: year_from / year_to per un intervallo.
        year_from: Primo anno dell'intervallo (0 = nessun limite).
        year_to: Ultimo anno dell'intervallo (0 = nessun limite).
        camera: Fotocamera o telefono (es. "nikon", "pixel").
        near: Luogo di riferimento per "foto vicino a ...". Per omonimi scrivere "Nome, Regione".
        radius_km: Raggio in km per near (default 20).
        top_k: Quante foto restituire (0 = valore di config.json, "top_k_photos"; massimo 50).

    Se un filtro non corrisponde a nulla, la risposta elenca i valori presenti.
    Per sapere quali luoghi, anni e fotocamere esistono usa photo_archive_overview.
    """
    t0 = time.time()
    logger.info(f"Ricerca foto: query={query!r} location={location!r} region={region!r} year={year} "
                f"da={year_from} a={year_to} camera={camera!r} near={near!r} raggio={radius_km}")
    try:
        k = int(top_k) if top_k and top_k > 0 else TOP_K_PHOTOS
        res = _photos().search(query=query, location=location, region=region, year=year,
                               year_from=year_from, year_to=year_to, camera=camera,
                               near=near, radius_km=radius_km, top_k=k)
        logger.info(f"Ricerca foto completata in {time.time() - t0:.1f}s")
        return format_photo_results(res)
    except Exception as e:
        logger.error(f"Ricerca foto fallita dopo {time.time() - t0:.1f}s: {e}")
        return f"Errore durante la ricerca delle foto: {str(e)}"


@mcp.tool()
def photo_archive_overview() -> str:
    """
    Panoramica dell'archivio fotografico: quante foto, con quante coordinate GPS,
    e quali anni, localita', regioni e fotocamere sono presenti (con il numero di
    foto per ciascuno). Utile per scoprire i valori validi da usare come filtri
    in find_photos.
    """
    try:
        return format_overview(_photos().overview())
    except Exception as e:
        return f"Errore leggendo la panoramica delle foto: {str(e)}"

if __name__ == "__main__":
    mcp.run()
