"""
Confronto tra modelli di embedding per la ricerca delle foto.

Usa le didascalie GIA' generate da un banco di prova (results.json prodotto da
bench_captions.py): non richiama nessun modello vision. Tu scrivi in un file di
testo alcune ricerche in italiano, come le faresti davvero, indicando per
ciascuna quali foto ti aspetti (il numero #N che vedi accanto alla foto nel
report.html). Lo script embedda didascalie e ricerche con ogni modello e dice
quanto spesso le foto giuste finiscono in cima.

File delle ricerche (una per riga; le righe che iniziano con # sono commenti):
    cascata di ghiaccio o stalattiti | 2
    castello su una collina con la nebbia | 3
    foto con piu' persone in montagna | 5,9,14

Esempi:
  python bench_embeddings.py --results bench_out/2026-09-20_17-07-58/results.json --queries queries_test.txt
  python bench_embeddings.py --results ... --queries ... --run "qwen3-vl" --models nomic-embed-text bge-m3 qwen3-embedding:0.6b
  python bench_embeddings.py --results ... --queries ... --extra-jsonl E:/local_rag_db/captions.jsonl

Con solo 20 didascalie il test e' poco severo (quasi tutto finisce in cima):
--extra-jsonl aggiunge come "elementi di disturbo" le didascalie della cache di
ingest.py (escluse le foto del banco di prova), cosi' la ricerca somiglia a
quella reale. Piu' didascalie ci sono, piu' il confronto e' affidabile.

Modelli da scaricare (a quanto so, presenti su Ollama):
  ollama pull nomic-embed-text      (quello usato ora)
  ollama pull bge-m3                (multilingue)
  ollama pull qwen3-embedding:0.6b  (multilingue, leggero)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import requests

try:
    import numpy as np
except ImportError:
    sys.exit("Serve numpy (di solito gia' installato con llama-index): pip install numpy")

BASE_URL = "http://localhost:11434"

DEFAULT_MODELS = ["nomic-embed-text", "bge-m3", "qwen3-embedding:0.6b"]

QWEN3_TASK = "Given a search query for a personal photo archive, retrieve the photo descriptions that match the query"

# Per ogni modello: (etichetta, prefisso query, prefisso documento).
# nomic-embed-text e qwen3-embedding rendono meglio con prefissi/istruzioni,
# che LlamaIndex NON aggiunge da solo: si vede qui quanto pesano.
PROFILES = {
    "nomic-embed-text": [
        ("nomic senza prefissi (come ora)", "", ""),
        ("nomic con prefissi", "search_query: ", "search_document: "),
    ],
    "qwen3-embedding": [
        ("qwen3-emb senza istruzione", "", ""),
        ("qwen3-emb con istruzione", f"Instruct: {QWEN3_TASK}\nQuery: ", ""),
    ],
}


def run_label(run):
    label = f"{run['model']} | {run['prompt']}"
    if run.get("variant"):
        label += f" | {run['variant']}"
    return label


def load_docs(results_path, run_sel):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    if not runs:
        sys.exit("Nessuna esecuzione dentro results.json.")
    if run_sel:
        matches = [r for r in runs if run_sel.lower() in run_label(r).lower()]
        if len(matches) != 1:
            elenco = "\n  ".join(run_label(r) for r in runs)
            sys.exit(f"--run {run_sel!r} corrisponde a {len(matches)} esecuzioni. Disponibili:\n  {elenco}")
        run = matches[0]
    else:
        run = runs[0]
        if len(runs) > 1:
            print(f"Nota: results.json contiene {len(runs)} esecuzioni, uso la prima. Scegline una con --run.")
    docs = []
    for i, res in enumerate(run["results"]):
        if res.get("caption"):
            docs.append({"id": i + 1, "text": res["caption"], "photo": res["photo"]})
    return docs, run_label(run)


def load_queries(path, valid_ids):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                sys.exit(f"{path}, riga {n}: manca '|'. Formato: testo della ricerca | 3,7")
            text, ids = line.rsplit("|", 1)
            try:
                expected = {int(x) for x in re.split(r"[,\s]+", ids.strip()) if x}
            except ValueError:
                sys.exit(f"{path}, riga {n}: dopo '|' servono numeri di foto separati da virgola.")
            if not expected:
                sys.exit(f"{path}, riga {n}: nessun numero di foto indicato.")
            bad = expected - valid_ids
            if bad:
                sys.exit(f"{path}, riga {n}: foto {sorted(bad)} non valide (disponibili: {min(valid_ids)}-{max(valid_ids)}).")
            queries.append({"text": text.strip(), "expected": expected})
    if not queries:
        sys.exit(f"Nessuna ricerca trovata in {path}.")
    return queries


def load_extra(path, limit, exclude_paths):
    norm = lambda p: os.path.normcase(os.path.normpath(p))
    exclude = {norm(p) for p in exclude_paths}
    seen, out = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cap = (rec.get("caption") or "").strip()
            if not cap or cap in seen or norm(rec.get("path", "")) in exclude:
                continue
            seen.add(cap)
            out.append(cap)
            if len(out) >= limit:
                break
    return out


def embed(base_url, model, texts, batch=16, timeout=300):
    vectors = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{base_url}/api/embed", json={"model": model, "input": texts[i:i + batch]}, timeout=timeout)
        if not r.ok:
            try:
                detail = r.json().get("error", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"Ollama {r.status_code}: {detail}")
        vectors.extend(r.json()["embeddings"])
    v = np.array(vectors, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def evaluate(scores, queries, doc_ids):
    """scores: (n_query, n_doc). Per ogni ricerca: posizione della prima foto attesa."""
    per_query = []
    for qi, q in enumerate(queries):
        order = np.argsort(-scores[qi], kind="stable")
        ranked = [doc_ids[j] for j in order]
        positions = sorted(ranked.index(e) + 1 for e in q["expected"])
        per_query.append({
            "first_rank": positions[0],
            "top1": ranked[0],
            "recall5": sum(1 for p in positions if p <= 5) / len(positions),
        })
    n = len(per_query)
    return {
        "hit1": sum(1 for p in per_query if p["first_rank"] == 1) / n,
        "hit3": sum(1 for p in per_query if p["first_rank"] <= 3) / n,
        "mrr": sum(1 / p["first_rank"] for p in per_query) / n,
        "recall5": sum(p["recall5"] for p in per_query) / n,
        "per_query": per_query,
    }


def main():
    ap = argparse.ArgumentParser(description="Confronta modelli di embedding sulle didascalie del banco di prova.")
    ap.add_argument("--results", required=True, help="results.json prodotto da bench_captions.py")
    ap.add_argument("--queries", required=True, help="file con le ricerche: 'testo | numeri foto attese'")
    ap.add_argument("--run", help="parte dell'etichetta dell'esecuzione da usare (es. 'qwen3-vl')")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--extra-jsonl", help="captions.jsonl di ingest.py: didascalie aggiunte come elementi di disturbo")
    ap.add_argument("--extra-max", type=int, default=2000)
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()

    docs, label = load_docs(args.results, args.run)
    queries = load_queries(args.queries, {d["id"] for d in docs})
    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    extra = []
    if args.extra_jsonl:
        extra = load_extra(args.extra_jsonl, args.extra_max, [d["photo"] for d in docs])
        doc_texts += extra
        doc_ids += [None] * len(extra)

    print(f"Didascalie di: {label}")
    print(f"{len(docs)} foto del banco di prova + {len(extra)} elementi di disturbo; {len(queries)} ricerche.")
    if len(doc_texts) < 50:
        print("Attenzione: con cosi' poche didascalie il test e' poco severo. Usa --extra-jsonl appena hai una cache piu' grande.")

    configs = []  # {"label", "metrics"}
    doc_cache, query_cache = {}, {}
    for model in args.models:
        profiles = PROFILES.get(model.split(":")[0], [(model, "", "")])
        print(f"\n### {model}")
        for plabel, qpre, dpre in profiles:
            try:
                if (model, dpre) not in doc_cache:
                    doc_cache[(model, dpre)] = embed(args.base_url, model, [dpre + t for t in doc_texts])
                if (model, qpre) not in query_cache:
                    query_cache[(model, qpre)] = embed(args.base_url, model, [qpre + q["text"] for q in queries])
            except Exception as e:
                msg = str(e)
                hint = f"  -> prova: ollama pull {model}" if "not found" in msg.lower() else ""
                print(f"  SALTATO: {msg}{hint}")
                break
            scores = query_cache[(model, qpre)] @ doc_cache[(model, dpre)].T
            m = evaluate(scores, queries, doc_ids)
            configs.append({"label": plabel, "model": model, "metrics": m})
            print(f"  {plabel}: prima foto giusta al 1o posto {m['hit1']:.0%}, nei primi 3 {m['hit3']:.0%}, MRR {m['mrr']:.2f}")

    if not configs:
        sys.exit("\nNessun modello ha prodotto risultati.")

    # tabella: righe = ricerche, colonne = configurazioni (posizione della prima foto giusta)
    print("\n" + "=" * 100)
    print("Posizione della prima foto giusta per ogni ricerca (1 = in cima). Tra parentesi la foto arrivata prima, se sbagliata.")
    letters = [chr(ord("A") + i) for i in range(len(configs))]
    for lt, c in zip(letters, configs):
        print(f"  {lt} = {c['label']}")
    print("-" * 100)
    print(f"{'ricerca':<44}" + "".join(f"{lt:<14}" for lt in letters))
    for qi, q in enumerate(queries):
        cells = []
        for c in configs:
            pq = c["metrics"]["per_query"][qi]
            cell = str(pq["first_rank"]) if pq["first_rank"] == 1 else f"{pq['first_rank']} (#{pq['top1'] if pq['top1'] else 'x'})"
            cells.append(f"{cell:<14}")
        text = q["text"] if len(q["text"]) <= 42 else q["text"][:41] + "…"
        print(f"{text:<44}" + "".join(cells))
    print("=" * 100)
    print(f"{'configurazione':<36}{'1o posto':<11}{'primi 3':<10}{'MRR':<7}{'richiamo@5'}")
    for c in sorted(configs, key=lambda c: (-c["metrics"]["mrr"], -c["metrics"]["hit1"])):
        m = c["metrics"]
        print(f"{c['label']:<36}{m['hit1']:<11.0%}{m['hit3']:<10.0%}{m['mrr']:<7.2f}{m['recall5']:.0%}")
    print("(#x = un elemento di disturbo, non una foto del banco di prova)")

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.results)),
                            f"embeddings_{datetime.now().strftime('%H-%M-%S')}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"run": label, "n_docs": len(docs), "n_extra": len(extra),
                       "queries": [{"text": q["text"], "expected": sorted(q["expected"])} for q in queries],
                       "configs": configs}, f, ensure_ascii=False, indent=2)
        print(f"\nRisultati salvati in {out_path}")
    except OSError as e:
        print(f"\n(risultati non salvati su file: {e})")


if __name__ == "__main__":
    main()
