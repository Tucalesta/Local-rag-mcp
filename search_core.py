"""
Ricerca su foto e documenti, condivisa da mcp_server.py e query.py.

Non dipende da llama_index: riceve la collezione ChromaDB gia' aperta e una
funzione che trasforma una frase in vettore (embed_query). Cosi' si puo'
provare da sola, con embedding finti, senza Ollama.

  - PhotoSearcher: ricerca semantica sulle descrizioni delle foto combinata con
    filtri esatti sui metadati (localita', regione, anno, fotocamera, raggio in
    km da un luogo). I filtri agiscono PRIMA della ricerca per significato:
    "foto del 2014 a Cortina con la neve" = filtro (anno 2014, Cortina) +
    somiglianza con "neve".
  - search_documents: passaggi piu' pertinenti tra i documenti di testo.
"""

import math
import re
import time
from collections import Counter

from rag_common import DEFAULT_TOP_K_PHOTOS, haversine_km, resolve_names

# Chiavi che LlamaIndex aggiunge ai metadati di Chroma per uso interno.
INTERNAL_KEYS = {"_node_content", "_node_type", "document_id", "doc_id", "ref_doc_id"}
MAX_FILTER_ONLY = 5000     # foto lette al massimo quando non c'e' testo da cercare
MAX_TOP_K = 50
FACET_TTL_SECONDS = 300    # ogni quanto rileggere l'elenco di luoghi/anni/fotocamere


def clean_metadata(meta):
    return {k: v for k, v in (meta or {}).items() if k not in INTERNAL_KEYS}


def format_date(date_taken):
    """'2014:08:12 10:31:04' -> '2014-08-12'."""
    if not date_taken:
        return ""
    m = re.match(r"(\d{4})[:\-/](\d{2})[:\-/](\d{2})", str(date_taken))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else str(date_taken)


def _is_cosine(collection):
    try:
        return (collection.metadata or {}).get("hnsw:space") == "cosine"
    except Exception:
        return False


# ---------------------------------------------------------------- foto

class PhotoSearcher:
    def __init__(self, collection, embed_query, gpx_points=None):
        self.collection = collection
        self.embed_query = embed_query
        self.gpx_points = gpx_points or []
        self._facets = None
        self._facets_at = 0.0

    # -- elenco dei valori presenti (località, regioni, anni, fotocamere)
    def facets(self, force=False):
        if self._facets is None or force or time.time() - self._facets_at > FACET_TTL_SECONDS:
            data = self.collection.get(where={"type": "image"}, include=["metadatas"])
            f = {"location_name": Counter(), "region": Counter(), "camera": Counter(),
                 "year": Counter(), "n": 0, "n_gps": 0, "coords": {}}
            for meta in data.get("metadatas") or []:
                meta = meta or {}
                f["n"] += 1
                for key in ("location_name", "region", "camera"):
                    if meta.get(key):
                        f[key][meta[key]] += 1
                if meta.get("year") is not None:
                    f["year"][int(meta["year"])] += 1
                if meta.get("gps_lat") is not None and meta.get("gps_lon") is not None:
                    f["n_gps"] += 1
                    if meta.get("location_name"):
                        f["coords"].setdefault(meta["location_name"], []).append((meta["gps_lat"], meta["gps_lon"]))
            self._facets = f
            self._facets_at = time.time()
        return self._facets

    def overview(self, max_items=30):
        f = self.facets(force=True)
        return {
            "photos": f["n"],
            "with_gps": f["n_gps"],
            "years": sorted(f["year"].items()),
            "locations": f["location_name"].most_common(max_items),
            "n_locations": len(f["location_name"]),
            "regions": f["region"].most_common(max_items),
            "cameras": f["camera"].most_common(10),
        }

    def resolve_field(self, field, wanted):
        return resolve_names(wanted, list(self.facets()[field]))

    # -- "vicino a": nome del luogo -> coordinate
    def resolve_near(self, near):
        """Restituisce (lat, lon, etichetta). 'Nome' oppure 'Nome, Regione' per
        distinguere omonimi. Cerca prima nel database GPX, poi tra le foto."""
        name, _, region_hint = near.partition(",")
        name, region_hint = name.strip(), region_hint.strip()

        if self.gpx_points:
            names = resolve_names(name, sorted({p[2] for p in self.gpx_points}))
            cands = [p for p in self.gpx_points if p[2] in names]
            if region_hint and cands:
                regions = resolve_names(region_hint, sorted({p[3] for p in cands}))
                cands = [p for p in cands if p[3] in regions]
            distinct = sorted({(p[2], p[3]) for p in cands})
            if len(distinct) == 1:
                lat, lon, n, r = cands[0]
                return lat, lon, f"{n} ({r})"
            if len(distinct) > 1:
                elenco = ", ".join(f"{n} ({r})" for n, r in distinct[:10])
                raise ValueError(
                    f"'{near}' e' ambiguo: {elenco}. Indica la regione, per esempio 'Nome, Regione'."
                )

        coords = self.facets()["coords"]
        names = resolve_names(name, list(coords))
        points = [c for n in names for c in coords[n]]
        if points:
            lat = sum(p[0] for p in points) / len(points)
            lon = sum(p[1] for p in points) / len(points)
            return lat, lon, names[0]
        raise ValueError(f"Luogo '{near}' non trovato ne' nel database delle localita' ne' tra le foto.")

    # -- ricerca
    def search(self, query="", location="", region="", year=0, year_from=0, year_to=0,
               camera="", near="", radius_km=20.0, top_k=DEFAULT_TOP_K_PHOTOS):
        notes = []
        conds = [{"type": "image"}]

        for field, wanted, label in (("location_name", location, "località"),
                                     ("region", region, "regione"),
                                     ("camera", camera, "fotocamera")):
            if not wanted:
                continue
            matches = self.resolve_field(field, wanted)
            if not matches:
                sample = ", ".join(n for n, _ in self.facets()[field].most_common(15)) or "(nessuno)"
                return {"error": f"Nessuna foto con {label} simile a '{wanted}'. "
                                 f"Valori presenti (i più frequenti): {sample}"}
            conds.append({field: matches[0]} if len(matches) == 1 else {field: {"$in": matches}})
            notes.append(f"{label}: {', '.join(matches)}")

        if year:
            conds.append({"year": int(year)})
            notes.append(f"anno {int(year)}")
        else:
            if year_from:
                conds.append({"year": {"$gte": int(year_from)}})
            if year_to:
                conds.append({"year": {"$lte": int(year_to)}})
            if year_from or year_to:
                notes.append(f"anni {year_from or '…'}-{year_to or '…'}")

        center = None
        near_label = ""
        if near:
            try:
                lat, lon, near_label = self.resolve_near(near)
            except ValueError as e:
                return {"error": str(e)}
            center = (lat, lon)
            dlat = radius_km / 111.0
            dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
            conds += [{"gps_lat": {"$gte": lat - dlat}}, {"gps_lat": {"$lte": lat + dlat}},
                      {"gps_lon": {"$gte": lon - dlon}}, {"gps_lon": {"$lte": lon + dlon}}]
            notes.append(f"entro {radius_km:g} km da {near_label}")

        where = conds[0] if len(conds) == 1 else {"$and": conds}
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        if query.strip():
            n = min(top_k * 5, 250) if center else top_k
            res = self.collection.query(
                query_embeddings=[self.embed_query(query)], n_results=n, where=where,
                include=["documents", "metadatas", "distances"])
            rows = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
        else:
            res = self.collection.get(where=where, include=["documents", "metadatas"], limit=MAX_FILTER_ONLY)
            rows = [(d, m, None) for d, m in zip(res["documents"] or [], res["metadatas"] or [])]

        cosine = _is_cosine(self.collection)
        items = []
        for doc, meta, dist in rows:
            meta = clean_metadata(meta)
            km = None
            if center:
                if meta.get("gps_lat") is None or meta.get("gps_lon") is None:
                    continue
                km = haversine_km(center[0], center[1], meta["gps_lat"], meta["gps_lon"])
                if km > radius_km:
                    continue
            items.append({
                "path": meta.get("path", ""),
                "file_name": meta.get("file_name", ""),
                "folder": meta.get("folder", ""),
                "date": format_date(meta.get("date_taken")),
                "date_ts": meta.get("date_ts"),
                "year": meta.get("year"),
                "year_source": meta.get("year_source", ""),
                "location": meta.get("location_name", ""),
                "region": meta.get("region", ""),
                "lat": meta.get("gps_lat"),
                "lon": meta.get("gps_lon"),
                "camera": meta.get("camera", ""),
                "text": doc or "",
                "relevance": (round(1 - dist, 3) if (cosine and dist is not None) else None),
                "distance": (round(dist, 3) if dist is not None else None),
                "km": km,
            })

        if not query.strip():
            if center:
                items.sort(key=lambda i: i["km"])
            else:
                items.sort(key=lambda i: (i["date_ts"] is None, i["date_ts"] or 0, i["path"]))

        return {"results": items[:top_k], "total": len(items) if not query.strip() else None,
                "notes": notes, "query": query.strip(), "near_label": near_label, "error": None}


def format_photo_results(res):
    """Testo leggibile del risultato di PhotoSearcher.search()."""
    if res.get("error"):
        return res["error"]
    items = res["results"]
    filters = "; ".join(res["notes"]) if res["notes"] else "nessun filtro"
    if res["query"]:
        head = f"Ricerca: \"{res['query']}\" | Filtri: {filters}"
    else:
        head = f"Solo filtri: {filters} (ordinate per data)" if not res.get("near_label") else f"Solo filtri: {filters} (dalla più vicina)"
    if not items:
        return head + "\nNessuna foto trovata. Prova ad allargare i filtri (raggio, anni) o a togliere qualche condizione."

    total = res.get("total")
    counted = f"{len(items)} foto mostrate" + (f" su {total} corrispondenti" if total and total > len(items) else "")
    lines = [head, counted, ""]
    for i, it in enumerate(items, 1):
        where_bits = []
        if it["location"]:
            where_bits.append(f"{it['location']} ({it['region']})" if it["region"] else it["location"])
        if it["km"] is not None:
            where_bits.append(f"a {it['km']:.1f} km")
        date = it["date"] or (f"{it['year']} (dalla cartella)" if it["year"] and it["year_source"] == "cartella" else "")
        title = " · ".join(b for b in (date, " - ".join(where_bits)) if b) or "(senza data né luogo)"
        lines.append(f"{i}) {title}")
        lines.append(f"   File: {it['path']}")
        if it["folder"]:
            lines.append(f"   Cartella: {it['folder']}")
        if it["lat"] is not None and it["lon"] is not None:
            lines.append(f"   GPS: {it['lat']}, {it['lon']}")
        if it["camera"]:
            lines.append(f"   Fotocamera: {it['camera']}")
        description = [ln for ln in it["text"].splitlines() if ln and not ln.startswith(("Luogo:", "Cartella:"))]
        text = " | ".join(description)
        if len(text) > 400:
            text = text[:397] + "..."
        lines.append(f"   Descrizione: {text}")
        if it["relevance"] is not None:
            lines.append(f"   Pertinenza: {it['relevance']}")
        elif it["distance"] is not None:
            lines.append(f"   Distanza: {it['distance']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_overview(ov):
    lines = [f"{ov['photos']} foto indicizzate, {ov['with_gps']} con coordinate GPS."]
    if ov["years"]:
        lines.append("Anni: " + ", ".join(f"{y} ({n})" for y, n in ov["years"]))
    if ov["locations"]:
        extra = f" (mostrate le {len(ov['locations'])} più frequenti su {ov['n_locations']})" if ov["n_locations"] > len(ov["locations"]) else ""
        lines.append(f"Località{extra}: " + ", ".join(f"{n} ({c})" for n, c in ov["locations"]))
    if ov["regions"]:
        lines.append("Regioni: " + ", ".join(f"{n} ({c})" for n, c in ov["regions"]))
    if ov["cameras"]:
        lines.append("Fotocamere: " + ", ".join(f"{n} ({c})" for n, c in ov["cameras"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- documenti

def search_documents(collection, embed_query, question, top_k):
    """Passaggi di testo piu' pertinenti (escluse le descrizioni delle foto). Il
    filtro e' "tipo diverso da image" e non "tipo uguale a text" cosi' funziona
    anche con le collezioni create prima che i documenti avessero il campo type."""
    res = collection.query(
        query_embeddings=[embed_query(question)], n_results=max(1, int(top_k)),
        where={"type": {"$ne": "image"}}, include=["documents", "metadatas", "distances"])
    passages = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        meta = clean_metadata(meta)
        passages.append({"text": doc or "", "file_name": meta.get("file_name", "Sconosciuto"),
                         "path": meta.get("path", ""), "distance": round(dist, 3)})
    return passages


def format_passages(passages, max_chars=1200):
    if not passages:
        return "Nessun passaggio trovato tra i documenti indicizzati."
    lines = []
    for i, p in enumerate(passages, 1):
        text = p["text"].strip()
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        lines.append(f"[{i}] {p['file_name']} ({p['path']})\n{text}\n")
    return "\n".join(lines).rstrip()


def build_answer_prompt(question, passages):
    context = "\n\n".join(f"[{i}] ({p['file_name']})\n{p['text']}" for i, p in enumerate(passages, 1))
    return (
        "Rispondi alla domanda usando SOLO i passaggi qui sotto. Se non contengono la risposta, dillo. "
        "Cita i numeri dei passaggi tra parentesi quadre.\n\n"
        f"PASSAGGI:\n{context}\n\nDOMANDA: {question}\n\nRISPOSTA:"
    )
