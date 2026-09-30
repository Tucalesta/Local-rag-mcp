"""
Funzioni condivise da ingest.py, mcp_server.py, query.py e search_core.py:
lettura di config.json, risoluzione dei percorsi, modello di embedding,
database delle localita' (GPX) e confronto di nomi (accenti, maiuscole,
piccole differenze). llama_index e chromadb vengono importati solo dentro le
funzioni che li usano, cosi' questo modulo si carica in un attimo.
"""

import difflib
import json
import logging
import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_COLLECTION = "local_docs"
DEFAULT_TOP_K_DOCS = 3
DEFAULT_TOP_K_PHOTOS = 10
DEFAULT_OLLAMA_TIMEOUT = 300.0
GPX_DIR_NAME = "res"


# ---------------------------------------------------------------- configurazione

def load_config(base_dir):
    """Legge config.json dalla cartella dello script (stesso file per ingest,
    server MCP e CLI, cosi' i parametri restano coerenti)."""
    path = os.path.join(base_dir, "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def resolve_path(config, keys, default, base_dir):
    """Primo valore presente tra 'keys' (o 'default'); se relativo e' risolto
    rispetto a base_dir e non alla cartella da cui parte il processo."""
    raw = default
    for key in keys:
        if key in config:
            raw = config[key]
            break
    return raw if os.path.isabs(raw) else os.path.normpath(os.path.join(base_dir, raw))


def embed_model_name(config):
    return config.get("embed_model", DEFAULT_EMBED_MODEL)


def collection_name(config):
    return config.get("collection_name", DEFAULT_COLLECTION)


def top_k_docs(config):
    return int(config.get("top_k_docs", DEFAULT_TOP_K_DOCS))


def top_k_photos(config):
    return int(config.get("top_k_photos", DEFAULT_TOP_K_PHOTOS))


def make_embed_model(config, timeout=DEFAULT_OLLAMA_TIMEOUT):
    """Modello di embedding di LlamaIndex per Ollama, scelto da config.json.
    embed_query_instruction / embed_text_instruction servono solo ai modelli
    che li richiedono (es. qwen3-embedding); con bge-m3 non servono."""
    from llama_index.embeddings.ollama import OllamaEmbedding

    kwargs = {"model_name": embed_model_name(config), "client_kwargs": {"timeout": timeout}}
    if config.get("embed_query_instruction"):
        kwargs["query_instruction"] = config["embed_query_instruction"]
    if config.get("embed_text_instruction"):
        kwargs["text_instruction"] = config["embed_text_instruction"]
    return OllamaEmbedding(**kwargs)


def open_collection(db_path, name, create=False):
    """Apre la collezione ChromaDB. In lettura (create=False) una collezione
    mancante e' un errore esplicito, non una collezione vuota creata di nascosto."""
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    if create:
        # distanza coseno: la piu' adatta agli embedding di testo
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(name)
    except Exception as e:
        raise RuntimeError(
            f"Collezione '{name}' non trovata in {db_path} ({e}). "
            "Esegui prima ingest.py con la stessa configurazione (embed_model / collection_name)."
        )


# ---------------------------------------------------------------- geografia

def haversine_km(lat1, lon1, lat2, lon2):
    """Distanza in km tra due coordinate (formula di Haversine, libreria
    standard, nessuna dipendenza esterna)."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_gpx_database(res_dir, logger=None):
    """
    Carica tutti i GPX in res_dir come un unico elenco di localita'
    (lat, lon, nome, regione). Il nome del file GPX (senza estensione)
    diventa la regione - cosi' come organizzati dall'utente. GPX con tracce
    invece che waypoint, o non parsabili, vengono segnalati e saltati senza
    bloccare l'indicizzazione.
    """
    logger = logger or logging.getLogger("rag_common")
    points = []
    if not os.path.isdir(res_dir):
        return points

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    for filename in os.listdir(res_dir):
        if not filename.lower().endswith(".gpx"):
            continue
        region = os.path.splitext(filename)[0]
        file_path = os.path.join(res_dir, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for wpt in root.findall("gpx:wpt", ns):
                lat = wpt.get("lat")
                lon = wpt.get("lon")
                name_el = wpt.find("gpx:name", ns)
                if lat is None or lon is None or name_el is None or not name_el.text:
                    continue
                points.append((float(lat), float(lon), name_el.text.strip(), region))
        except Exception as e:
            logger.warning(f"Impossibile leggere il GPX '{filename}': {e}")

    logger.info(f"  -> Database localita' (GPX): {len(points)} punti caricati da {res_dir}")
    return points


def find_nearest_location(lat, lon, gpx_points, max_distance_km=15):
    """
    Trova la localita' piu' vicina nel database GPX personale. Restituisce
    SEMPRE il punto piu' vicino trovato insieme alla sua distanza (anche se
    supera max_distance_km), piu' un flag "within_threshold": questo separa
    la logica di matching da quella di decidere se usarlo o segnalarlo come
    possibile buco del database - il chiamante decide cosa farne.
    Confronto lineare su tutti i punti: con poche migliaia di localita' e'
    comunque questione di millisecondi, non serve un indice spaziale.
    """
    best = None
    best_dist = None
    for plat, plon, name, region in gpx_points:
        dist = haversine_km(lat, lon, plat, plon)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (name, region, dist)
    if best is None:
        return None
    name, region, dist = best
    return {"name": name, "region": region, "distance_km": dist, "within_threshold": dist <= max_distance_km}


# ---------------------------------------------------------------- confronto di nomi

def normalize_text(value):
    """minuscolo, senza accenti ne' punteggiatura: "Cortina d'Ampezzo" -> "cortina d ampezzo"."""
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def resolve_names(wanted, candidates, cutoff=0.8):
    """
    Trova tra 'candidates' i nomi che corrispondono a 'wanted', in quest'ordine:
    1) uguali (senza accenti/maiuscole); 2) tutte le parole cercate compaiono nel
    candidato o viceversa ("Cortina" -> "Cortina d'Ampezzo"); 3) molto simili
    (piccoli errori di battitura). Restituisce i candidati originali.
    """
    w = normalize_text(wanted)
    if not w:
        return []
    norm = {c: normalize_text(c) for c in candidates}
    exact = sorted(c for c, n in norm.items() if n == w)
    if exact:
        return exact
    wanted_tokens = set(w.split())
    by_tokens = sorted(
        c for c, n in norm.items()
        if n and (wanted_tokens <= set(n.split()) or set(n.split()) <= wanted_tokens)
    )
    if by_tokens:
        return by_tokens
    close = set(difflib.get_close_matches(w, list(set(norm.values())), n=5, cutoff=cutoff))
    return sorted(c for c, n in norm.items() if n in close)
