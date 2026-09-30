import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

import rag_common
from search_core import (
    PhotoSearcher,
    build_answer_prompt,
    format_overview,
    format_passages,
    format_photo_results,
    search_documents,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parametri della ricerca foto: chiave scritta dall'utente -> (argomento di PhotoSearcher.search, tipo)
PHOTO_KEYS = {
    "luogo": ("location", str),
    "regione": ("region", str),
    "anno": ("year", int),
    "da": ("year_from", int),
    "a": ("year_to", int),
    "fotocamera": ("camera", str),
    "vicino": ("near", str),
    "raggio": ("radius_km", float),
    "n": ("top_k", int),
}

HELP = """Comandi:
  <domanda>                      cerca nei documenti e risponde con il modello locale
  foto: <cosa si vede> | luogo=Cortina | anno=2014 | vicino=San Leo | raggio=30
                                 cerca foto; i filtri (dopo '|') sono facoltativi
                                 chiavi: luogo, regione, anno, da, a, fotocamera, vicino, raggio, n
  foto: | luogo=Cortina          solo filtri, senza testo da cercare
  panoramica                     anni, luoghi e fotocamere presenti nell'archivio foto
  aiuto                          questo elenco
  exit                           esce"""


def parse_photo_command(text):
    """'castello | luogo=San Leo | anno=2014' -> argomenti per PhotoSearcher.search()."""
    parts = [p.strip() for p in text.split("|")]
    kwargs = {"query": parts[0]}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Filtro non valido: '{part}' (usa chiave=valore, es. anno=2014)")
        key, value = (x.strip() for x in part.split("=", 1))
        if key.lower() not in PHOTO_KEYS:
            raise ValueError(f"Chiave sconosciuta: '{key}'. Valide: {', '.join(PHOTO_KEYS)}")
        arg, typ = PHOTO_KEYS[key.lower()]
        try:
            kwargs[arg] = typ(value)
        except ValueError:
            raise ValueError(f"Valore non valido per '{key}': '{value}'")
    return kwargs


def main():
    config = rag_common.load_config(BASE_DIR)
    db_path = rag_common.resolve_path(config, ("db_path",), "./chroma_db", BASE_DIR)
    coll_name = rag_common.collection_name(config)

    if not os.path.exists(db_path):
        print(f"L'indice in '{db_path}' non esiste. Esegui prima 'ingest.py'!")
        return

    # Stessi modelli e stessa collezione di ingest.py e del server MCP (config.json)
    Settings.embed_model = rag_common.make_embed_model(config)
    Settings.llm = Ollama(model=config.get("llm_model", "gemma3"), request_timeout=300.0)

    print("1. Connessione al database vettoriale locale (ChromaDB)...")
    try:
        collection = rag_common.open_collection(db_path, coll_name)
    except RuntimeError as e:
        print(e)
        return

    def embed_query(text):
        return Settings.embed_model.get_query_embedding(text)

    res_dir = rag_common.resolve_path(config, ("res_path", "res_dir", "gpx_dir"), rag_common.GPX_DIR_NAME, BASE_DIR)
    photos = PhotoSearcher(collection, embed_query, rag_common.load_gpx_database(res_dir))
    top_docs = rag_common.top_k_docs(config)
    top_photos = rag_common.top_k_photos(config)

    print("\n" + "=" * 50)
    print(f" Sistema RAG Locale pronto! Collezione '{coll_name}' ({collection.count()} chunk)")
    print(" Scrivi 'aiuto' per i comandi, 'exit' per uscire")
    print("=" * 50 + "\n")

    while True:
        question = input("Fai una domanda ai tuoi documenti o cerca foto: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "esci"]:
            print("Arrivederci!")
            break
        if question.lower() in ["aiuto", "help", "?"]:
            print(HELP + "\n")
            continue
        if question.lower() == "panoramica":
            try:
                print("\n" + format_overview(photos.overview()) + "\n")
            except Exception as e:
                print(f"Errore: {e}\n")
            continue

        if question.lower().startswith("foto:"):
            try:
                kwargs = parse_photo_command(question[5:])
                kwargs.setdefault("top_k", top_photos)
                print("\nSto cercando tra le foto...\n")
                print(format_photo_results(photos.search(**kwargs)) + "\n")
            except ValueError as e:
                print(f"{e}\n")
            except Exception as e:
                print(f"Errore durante la ricerca delle foto: {e}\n")
            continue

        print("\nSto cercando nei documenti e generando la risposta...")
        try:
            passages = search_documents(collection, embed_query, question, top_docs)
            if not passages:
                print("\nNessun passaggio trovato tra i documenti indicizzati.\n")
                continue
            response = Settings.llm.complete(build_answer_prompt(question, passages))
            print(f"\n--- RISPOSTA ---\n{response.text}\n")

            print("--- FONTI UTILIZZATE ---")
            sources_seen = set()
            for p in passages:
                key = (p["file_name"], p["path"])
                if key not in sources_seen:
                    sources_seen.add(key)
                    print(f"• {p['file_name']} ({p['path']})")
            print("-" * 30 + "\n")

        except Exception as e:
            print(f"Errore durante l'interrogazione: {e}\n")

if __name__ == "__main__":
    main()
