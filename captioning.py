"""
Didascalie delle immagini via Ollama.

Modulo condiviso da ingest.py e bench_captions.py, cosi' il banco di prova
usa ESATTAMENTE lo stesso codice (prompt, preparazione immagine, chiamata
a Ollama) dell'indicizzazione vera.

Contiene:
  - PROMPTS: prompt versionati. Cambiare/aggiungere un prompt = nuova chiave,
    mai modificare una versione gia' usata (le didascalie in cache sono
    associate alla versione con cui sono state generate).
  - caption_image(): prepara l'immagine, chiama Ollama, ripulisce il testo.
  - CaptionCache: archivio su disco (JSONL, una riga per didascalia) che
    sopravvive a force_reindex e a un cambio di modello embedding: rifare
    l'embedding di una didascalia costa millisecondi, generarla ~10 secondi.
"""

import base64
import io
import json
import os
import re
import time

import requests

OLLAMA_BASE_URL = "http://localhost:11434"

# Versione di prompt usata se config.json non specifica "caption_prompt_version".
DEFAULT_PROMPT_VERSION = "v3"

# Parametri di generazione di default (sovrascrivibili da config.json con
# "caption_options"). Temperatura bassa = meno invenzioni; num_predict limita
# la lunghezza e impedisce a un modello "logorroico" di andare avanti a lungo.
DEFAULT_OPTIONS = {"temperature": 0.2, "num_predict": 400}

# Lato massimo (px) a cui riscalare la foto prima di inviarla. I modelli
# ridimensionano comunque internamente; alcuni (famiglia Qwen-VL) pero'
# trasformano una foto grande in migliaia di token visivi, con tempi molto
# piu' lunghi e nessun beneficio per una didascalia. 0/None = invia il file
# originale senza toccarlo.
DEFAULT_MAX_SIDE = 1024

PROMPTS = {
    # Prompt originale di ingest.py (tenuto solo per confronto/riproducibilita').
    "v1": (
        "Descrivi in modo dettagliato e oggettivo il contenuto di questa immagine: "
        "soggetti principali, ambientazione, colori, testo visibile, contesto. "
        "Scrivi in italiano, in un unico paragrafo di massimo 5-6 frasi."
    ),
    # Stile telegrafico a campi: niente giri di parole ("L'immagine mostra...")
    # e parole chiave finali, piu' adatte alla ricerca semantica.
    "v2": (
        "Sei un archivista fotografico. Descrivi questa foto per una ricerca testuale.\n"
        "Rispondi solo in italiano, in stile telegrafico, senza frasi introduttive: "
        "non iniziare con \"L'immagine mostra\" o simili.\n"
        "Formato:\n"
        "Soggetto: ... (se ci sono persone: quante e cosa fanno, senza identificarle)\n"
        "Ambiente: ... (paesaggio, edificio, interno/esterno)\n"
        "Attivita': ...\n"
        "Oggetti e testo leggibile: ...\n"
        "Stagione/meteo: ...\n"
        "Parole chiave: 8-10 parole singole\n"
        "Se un elemento non e' chiaramente visibile, omettilo. Non inventare."
    ),
    # Stessa struttura di v2, in inglese: serve a confrontare la resa dei
    # modelli nelle due lingue (embedding nomic = addestrato soprattutto su inglese).
    "v2_en": (
        "You are a photo archivist. Describe this photo for text search.\n"
        "Answer only in English, in telegraphic style, with no introductory phrases: "
        "do not start with \"The image shows\" or similar.\n"
        "Format:\n"
        "Subject: ... (if people are present: how many and what they are doing, do not identify them)\n"
        "Setting: ... (landscape, building, indoor/outdoor)\n"
        "Activity: ...\n"
        "Objects and legible text: ...\n"
        "Season/weather: ...\n"
        "Keywords: 8-10 single words\n"
        "If an element is not clearly visible, omit it. Do not invent details."
    ),
    # v2 corretto dopo il banco di prova: accenti scritti bene (qwen3-vl copiava
    # "Attivita':" dal prompt) e righe senza informazione da saltare del tutto
    # (i modelli scrivevano "Soggetto: nessuno", rumore per la ricerca).
    "v3": (
        "Sei un archivista fotografico. Descrivi questa foto per una ricerca testuale.\n"
        "Rispondi solo in italiano, in stile telegrafico, senza frasi introduttive: "
        "non iniziare con \"L'immagine mostra\" o simili.\n"
        "Scrivi solo le righe per cui hai un'informazione; salta del tutto quelle "
        "senza contenuto (niente \"nessuno\", \"non visibile\", \"n/d\").\n"
        "Formato:\n"
        "Soggetto: ... (se ci sono persone: quante e cosa fanno, senza identificarle)\n"
        "Ambiente: ... (paesaggio, edificio, interno/esterno)\n"
        "Attività: ...\n"
        "Oggetti e testo leggibile: ...\n"
        "Stagione/meteo: ...\n"
        "Parole chiave: 8-10 parole singole, senza ripetizioni\n"
        "Non inventare: se non sei sicuro di un dettaglio, omettilo."
    ),
}


def get_prompt(version):
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"Versione di prompt sconosciuta: {version!r}. Disponibili: {', '.join(PROMPTS)}"
        )


def prepare_image_b64(file_path, max_side=DEFAULT_MAX_SIDE):
    """Restituisce l'immagine in base64 (JPEG riscalato). Applica l'orientamento
    EXIF, altrimenti le foto scattate in verticale arrivano al modello ruotate e
    vengono descritte male. Se Pillow non riesce ad aprire il file, ripiega
    sul file originale."""
    if max_side:
        try:
            from PIL import Image, ImageOps

            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.thumbnail((max_side, max_side), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# Righe "Etichetta: valore" il cui valore non dice nulla ("Soggetto: nessuno",
# "Stagione/meteo: non visibile"): nell'indice sono solo rumore.
_LABEL_LINE_RE = re.compile(r"^([^:\n]{1,40}):\s*(.*)$")
_EMPTY_VALUE_RE = re.compile(
    r"^(?:"
    r"nessun[oa]?"
    r"(?:\s+(?:persona|persone|attività|attivita|oggetto|oggetti|testo|elemento|elementi"
    r"|dettaglio|dettagli|animale|animali)(?:\s+\w+){0,2})?"
    r"(?:\s+(?:visibil[ei]|presente|presenti|leggibil[ei]))?"
    r"|non\s+(?:è\s+)?(?:visibil[ei]|determinabil[ei]|applicabil[ei]|identificabil[ei]|presenti?)"
    r"|impossibile\s+da\s+(?:determinare|stabilire)"
    r"|n/?[ad]|inanimat[oa]|-+|—|none|not\s+(?:visible|applicable)|no\s+one"
    r")\.?$",
    re.IGNORECASE,
)
_KEYWORD_LABELS = {"parole chiave", "keywords"}


def _dedupe_keywords(value):
    seen, out = set(), []
    for kw in re.split(r"\s*,\s*", value.strip().rstrip(".")):
        kw = kw.strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            out.append(kw)
    return ", ".join(out)


def clean_caption(text):
    """Pulizia: blocchi <think>, grassetti markdown, righe vuote, righe
    "Etichetta: nessuno / non visibile" e parole chiave ripetute."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    text = text.replace("**", "")
    kept = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = _LABEL_LINE_RE.match(ln)
        if m:
            label, value = m.group(1).strip(), m.group(2).strip()
            if not value or _EMPTY_VALUE_RE.match(value):
                continue
            if label.lower() in _KEYWORD_LABELS:
                ln = f"{label}: {_dedupe_keywords(value)}"
        kept.append(ln)
    return "\n".join(kept).strip()


def _error_detail(response):
    try:
        return response.json().get("error", response.text)
    except Exception:
        return response.text


def caption_image(
    file_path,
    model,
    prompt_version=DEFAULT_PROMPT_VERSION,
    options=None,
    max_side=DEFAULT_MAX_SIDE,
    timeout=180,
    think=None,
    base_url=OLLAMA_BASE_URL,
    keep_alive="10m",
    return_meta=False,
):
    """
    Genera la didascalia di un'immagine con un modello vision di Ollama.

    Restituisce il testo (str); con return_meta=True restituisce (testo, meta)
    dove meta contiene tempi e numero di token riportati da Ollama
    (prompt_eval_count include i token visivi dell'immagine: utile per capire
    quanto pesa la risoluzione).

    think: None = non inviare il parametro; False = chiede a Ollama di
    disattivare il "ragionamento" nei modelli che lo supportano.
    """
    merged_options = dict(DEFAULT_OPTIONS)
    if options:
        merged_options.update(options)

    t0 = time.time()
    payload = {
        "model": model,
        "prompt": get_prompt(prompt_version),
        "images": [prepare_image_b64(file_path, max_side)],
        "stream": False,
        "options": merged_options,
        "keep_alive": keep_alive,
    }
    if think is not None:
        payload["think"] = think

    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    if not response.ok:
        # Il corpo della risposta di Ollama contiene di solito il vero motivo
        # dell'errore (modello non presente, out-of-memory GPU, ...).
        raise RuntimeError(f"Ollama {response.status_code}: {_error_detail(response)}")

    data = response.json()
    caption = clean_caption(data.get("response", ""))
    if not return_meta:
        return caption

    ns = 1e9
    meta = {
        "wall_s": round(time.time() - t0, 2),
        "total_s": round(data.get("total_duration", 0) / ns, 2),
        "load_s": round(data.get("load_duration", 0) / ns, 2),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "eval_s": round(data.get("eval_duration", 0) / ns, 2),
    }
    return caption, meta


class CaptionCache:
    """
    Cache persistente delle didascalie: file JSONL, una riga per didascalia,
    in sola aggiunta (append-only) e con flush ad ogni scrittura: se il
    processo viene interrotto si perde al massimo la riga in corso, che al
    caricamento successivo viene ignorata.

    Chiave = (percorso, modello, versione del prompt); una voce e' valida solo
    se data di modifica e dimensione del file coincidono con quelle salvate.
    Cosi' varianti diverse (altro modello, altro prompt) convivono senza
    sovrascriversi e un file modificato viene ridescritto.
    """

    def __init__(self, path):
        self.path = path
        self._data = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._data[(rec["path"], rec["model"], rec["prompt"])] = rec
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # riga troncata da un'interruzione o malformata

    def __len__(self):
        return len(self._data)

    def get(self, file_path, signature, model, prompt_version):
        rec = self._data.get((file_path, model, prompt_version))
        if not rec:
            return None
        if rec.get("mtime") != signature["mtime"] or rec.get("size") != signature["size"]:
            return None
        return rec.get("caption") or None

    def put(self, file_path, signature, model, prompt_version, caption):
        rec = {
            "path": file_path,
            "model": model,
            "prompt": prompt_version,
            "mtime": signature["mtime"],
            "size": signature["size"],
            "caption": caption,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._data[(file_path, model, prompt_version)] = rec
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def path_is_under(path, roots):
    """True se 'path' e' una delle cartelle 'roots' o sta sotto una di esse."""
    p = os.path.normcase(os.path.normpath(path))
    for root in roots:
        r = os.path.normcase(os.path.normpath(root)).rstrip("\\/")
        if p == r or p.startswith(r + os.sep):
            return True
    return False


def prune_cache_file(path, roots, only_model=None, only_prompt=None, dry_run=False, force=False,
                     max_orphan_fraction=0.5, in_use_seconds=120):
    """
    Compatta la cache delle didascalie (ingest.py --prune-cache):
      - tiene una sola riga per foto/modello/prompt (l'ultima);
      - toglie le righe illeggibili (per esempio troncate da un'interruzione);
      - toglie le didascalie di file che non esistono piu' SOLO se il file sta
        sotto una cartella radice attualmente raggiungibile: con un disco
        scollegato non viene cancellato niente;
      - con only_model/only_prompt tiene solo le didascalie di quel modello e
        prompt (le altre versioni vengono eliminate).
    Prima di riscrivere fa una copia "<file>.bak-<data>". Rifiuta di procedere
    (a meno di force=True) se il file e' stato modificato da meno di
    'in_use_seconds' (ingest.py potrebbe stare scrivendo) o se oltre
    'max_orphan_fraction' delle didascalie sembra riferirsi a file spariti.
    Restituisce un dizionario di statistiche; RuntimeError se non e' sicuro.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Cache non trovata: {path}")
    if not force and time.time() - os.path.getmtime(path) < in_use_seconds:
        raise RuntimeError(
            f"La cache e' stata modificata negli ultimi {in_use_seconds} secondi: ingest.py potrebbe "
            "essere in esecuzione. Fermalo e riprova (oppure usa --force)."
        )

    records = {}  # chiave -> record, in ordine di ultima comparsa
    total_lines = bad = valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            try:
                rec = json.loads(line)
                key = (rec["path"], rec["model"], rec["prompt"])
                rec["caption"]
            except (json.JSONDecodeError, KeyError, TypeError):
                bad += 1
                continue
            valid += 1
            records.pop(key, None)
            records[key] = rec

    reachable = [r for r in roots if os.path.isdir(r)]
    kept, orphans, other = [], [], []
    for rec in records.values():
        if only_model is not None and (rec["model"], rec["prompt"]) != (only_model, only_prompt):
            other.append(rec)
        elif path_is_under(rec["path"], reachable) and not os.path.exists(rec["path"]):
            orphans.append(rec)
        else:
            kept.append(rec)

    considered = len(records) - len(other)
    if not force and considered and len(orphans) > max_orphan_fraction * considered:
        raise RuntimeError(
            f"{len(orphans)} didascalie su {considered} sembrano riferirsi a file non piu' presenti "
            f"(oltre il {max_orphan_fraction:.0%}): probabile disco non raggiungibile o cartelle spostate. "
            "Controlla e, se e' voluto, riprova con --force."
        )

    stats = {"lines": total_lines, "unreadable": bad, "duplicates": valid - len(records),
             "orphans": len(orphans), "other_version": len(other), "kept": len(kept),
             "dry_run": dry_run, "backup": None}
    changed = bool(bad or stats["duplicates"] or orphans or other)
    if changed and not dry_run:
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(path, backup)
        os.replace(tmp, path)
        stats["backup"] = backup
    return stats
