"""
Banco di prova per le didascalie delle foto.

Confronta piu' modelli vision di Ollama (ed eventualmente piu' versioni di
prompt e varianti di risoluzione/contesto) sulle STESSE foto, misurando tempi
e uso della GPU e producendo un report HTML con le didascalie affiancate,
dove puoi dare un voto a ciascuna.

Usa lo stesso modulo (captioning.py) dell'indicizzazione vera: cio' che vedi
qui e' cio' che otterrai con ingest.py.

Esempi:
  python bench_captions.py --src "X:/02_Memoria_Personale/Foto e video per anno/2014"
  python bench_captions.py --models gemma3:4b qwen3-vl:8b-instruct --prompts v3
  python bench_captions.py --models qwen3-vl:8b-instruct --variants 1024 768:2048
  python bench_captions.py --photos "X:/foto/a.jpg" "X:/foto/b.jpg"

Varianti (--variants): ognuna e' "LATO_MAX" oppure "LATO_MAX:NUM_CTX", dove
LATO_MAX e' il lato lungo in px a cui riscalare la foto (0 = originale) e
NUM_CTX la finestra di contesto di Ollama. Un contesto piu' piccolo e una foto
piu' piccola occupano meno VRAM: utile se il modello non sta tutto in GPU.

Prima di lanciarlo: chiudi ComfyUI e ferma eventuali ingest.py in corso,
altrimenti la VRAM e' contesa e i tempi non sono confrontabili.
Risultati in: bench_out/<data_ora>/report.html  (+ results.json)
"""

import argparse
import base64
import html
import io
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

import requests

import captioning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = ["gemma3:4b", "qwen2.5vl:7b", "qwen3-vl:8b-instruct"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SKIP_DIR_NAMES = {"$recycle.bin", "system volume information", "__pycache__"}

# Didascalie che iniziano con giri di parole senza valore ("L'immagine mostra...")
FILLER_RE = re.compile(
    r"^\W*(l['’]\s*immagine|la\s+(foto|fotografia|immagine)|in\s+(questa|quest['’])\s*(immagine|foto)"
    r"|questa\s+(immagine|foto)|the\s+(image|photo|picture)|this\s+(image|photo|picture)"
    r"|in\s+this\s+(image|photo|picture))",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(r"(parole chiave|keywords)\s*:", re.IGNORECASE)


def short(path):
    """cartella/nome.jpg: nell'archivio per anno i nomi file si ripetono tra cartelle."""
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))


def parse_variants(specs, default_max_side, default_num_ctx):
    """'1024' -> {max_side:1024}; '768:2048' -> {max_side:768, num_ctx:2048}."""
    if not specs:
        specs = [f"{default_max_side}" + (f":{default_num_ctx}" if default_num_ctx else "")]
    variants = []
    for spec in specs:
        parts = spec.split(":")
        try:
            max_side = int(parts[0])
            num_ctx = int(parts[1]) if len(parts) > 1 and parts[1] else None
        except ValueError:
            sys.exit(f"Variante non valida: {spec!r} (formato: LATO_MAX oppure LATO_MAX:NUM_CTX)")
        label = "originale" if max_side == 0 else f"{max_side}px"
        if num_ctx:
            label += f", ctx {num_ctx}"
        variants.append({"max_side": max_side, "num_ctx": num_ctx, "label": label})
    return variants


# ---------------------------------------------------------------- selezione foto

def load_default_src():
    cfg_path = os.path.join(BASE_DIR, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["directories"][0]["path"]
    except Exception:
        return None


def collect_images(src, min_side=256):
    """Elenca le foto sotto src, scartando cartelle di sistema e immagini troppo
    piccole (icone/grafica)."""
    found = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIR_NAMES and not d.startswith("$")]
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, name))
    try:
        from PIL import Image

        kept = []
        for p in found:
            try:
                with Image.open(p) as im:
                    if min(im.size) >= min_side:
                        kept.append(p)
            except Exception:
                continue
        return kept
    except ImportError:
        return found


def pick_photos(paths, src, n, seed):
    """Campione riproducibile, distribuito a rotazione tra le sottocartelle di
    primo livello (nel tuo archivio: gli anni), cosi' il test copre epoche e
    fotocamere diverse invece di 20 foto della stessa gita."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for p in paths:
        rel = os.path.relpath(p, src)
        top = rel.split(os.sep)[0] if os.sep in rel else "."
        groups[top].append(p)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    chosen = []
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    return sorted(chosen)


# ---------------------------------------------------------------- Ollama helpers

def ollama_call(base_url, payload, timeout):
    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    if not r.ok:
        try:
            detail = r.json().get("error", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Ollama {r.status_code}: {detail}")
    return r.json()


def preload(base_url, model, timeout):
    """Carica il modello in memoria (senza generare) e restituisce i secondi impiegati."""
    t0 = time.time()
    ollama_call(base_url, {"model": model, "keep_alive": "10m"}, timeout)
    return round(time.time() - t0, 1)


def unload(base_url, model):
    try:
        requests.post(f"{base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def gpu_percent(base_url, model):
    """Percentuale del modello effettivamente in VRAM (100 = tutto su GPU;
    meno = parte spostata su CPU/RAM, con forte perdita di velocita')."""
    try:
        r = requests.get(f"{base_url}/api/ps", timeout=10)
        base = model.split(":")[0]
        for m in r.json().get("models", []):
            name = m.get("name", "") or m.get("model", "")
            if name == model or name.split(":")[0] == base:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                return round(100 * vram / size) if size else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- statistiche

def summarize_run(run):
    ok = [r for r in run["results"] if r.get("caption")]
    errs = [r for r in run["results"] if r.get("error")]
    walls = [r["meta"]["wall_s"] for r in ok if r.get("meta")]
    s = {
        "ok": len(ok),
        "errors": len(errs),
        "avg_s": round(sum(walls) / len(walls), 1) if walls else None,
        "median_s": round(statistics.median(walls), 1) if walls else None,
        "avg_chars": round(sum(len(r["caption"]) for r in ok) / len(ok)) if ok else None,
        "filler": sum(1 for r in ok if FILLER_RE.match(r["caption"])),
        "format_ok": sum(1 for r in ok if FORMAT_RE.search(r["caption"])),
        "prompt_tokens": round(sum(r["meta"]["prompt_tokens"] for r in ok) / len(ok)) if ok else None,
        "output_tokens": round(sum(r["meta"]["output_tokens"] for r in ok) / len(ok)) if ok else None,
    }
    return s


def print_summary(runs):
    print("\n" + "=" * 118)
    print(f"{'modello':<24}{'prompt':<8}{'variante':<18}{'ok/err':<8}{'s/foto':<8}{'mediana':<9}{'GPU%':<6}{'car.':<6}{'filler':<8}{'formato'}")
    print("-" * 118)
    for run in runs:
        s = run["summary"]
        print(
            f"{run['model']:<24}{run['prompt']:<8}{run.get('variant', ''):<18}"
            f"{str(s['ok']) + '/' + str(s['errors']):<8}{str(s['avg_s']):<8}{str(s['median_s']):<9}"
            f"{str(run.get('gpu_pct')):<6}{str(s['avg_chars']):<6}"
            f"{str(s['filler']) + '/' + str(s['ok']):<8}{str(s['format_ok']) + '/' + str(s['ok'])}"
        )
    print("=" * 118)
    print("filler = didascalie che iniziano con 'L'immagine mostra...' | formato = didascalie con 'Parole chiave' (prompt seguito)")


# ---------------------------------------------------------------- report HTML

def thumb_b64(path, size=360):
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


REPORT_CSS = """
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#6b6b6b;--card:#fff;--line:#dcdcdc;--accent:#2a6fdb;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#9a9a9a;--card:#202020;--line:#3a3a3a;--accent:#6ea8ff;--bad:#ff8a80}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--muted);margin-bottom:14px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card);margin-bottom:18px}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}
th{background:var(--bg);position:sticky;top:0;z-index:1}
td.photo{min-width:200px;max-width:240px} td.photo img{width:100%;border-radius:4px;display:block}
td.photo .name{font-size:11px;color:var(--muted);word-break:break-all;margin-top:4px}
td.cap{min-width:300px} .cap .txt{white-space:pre-wrap}
.meta{color:var(--muted);font-size:12px;margin-top:6px} .err{color:var(--bad)}
select{margin-top:6px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:2px 4px}
.num{text-align:right;font-variant-numeric:tabular-nums} .best{font-weight:600;color:var(--accent)}
"""

REPORT_JS = """
const RUNS = __RUNS__;
const PHOTOS = __PHOTOS__;
const KEY = "bench-__ID__";
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ratings = {}; }

function summarize() {
  const cells = document.querySelectorAll("[data-vote]");
  cells.forEach(el => { const v = ratings[el.dataset.vote]; el.value = (v === undefined) ? "" : String(v); });
  const box = document.getElementById("votes");
  let best = -1; const rows = [];
  RUNS.forEach((run, ri) => {
    let n = 0, sum = 0;
    PHOTOS.forEach((p, pi) => { const v = ratings[pi + "|" + ri]; if (v !== undefined) { n++; sum += Number(v); } });
    const avg = n ? sum / n : null;
    if (avg !== null && avg > best) best = avg;
    rows.push([run, n, avg]);
  });
  box.innerHTML = rows.map(([run, n, avg]) =>
    "<tr><td>" + run + "</td><td class='num'>" + n + "/" + PHOTOS.length + "</td><td class='num" +
    (avg !== null && avg === best ? " best" : "") + "'>" + (avg === null ? "-" : avg.toFixed(2)) + "</td></tr>").join("");
}
document.addEventListener("change", e => {
  const k = e.target && e.target.dataset && e.target.dataset.vote;
  if (!k) return;
  if (e.target.value === "") delete ratings[k]; else ratings[k] = Number(e.target.value);
  try { localStorage.setItem(KEY, JSON.stringify(ratings)); } catch (err) {}
  summarize();
});
summarize();
"""


def build_report(data, out_path):
    runs = data["runs"]
    photos = data["photos"]
    src = data["settings"].get("src") or ""

    def rel(p):
        try:
            return os.path.relpath(p, src) if src else p
        except ValueError:
            return p

    multi_variant = len({r.get("variant") for r in runs}) > 1
    run_labels = [
        f"{r['model']} | {r['prompt']}" + (f" | {r.get('variant', '')}" if multi_variant else "")
        for r in runs
    ]
    esc = html.escape

    # tabella riassuntiva
    sum_rows = []
    for r in runs:
        s = r["summary"]
        sum_rows.append(
            "<tr><td>{m}</td><td>{p}</td><td>{v}</td><td class='num'>{ok}/{er}</td><td class='num'>{avg}</td>"
            "<td class='num'>{med}</td><td class='num'>{gpu}</td><td class='num'>{load}</td>"
            "<td class='num'>{chars}</td><td class='num'>{ptok}</td><td class='num'>{filler}/{ok}</td>"
            "<td class='num'>{fmt}/{ok}</td></tr>".format(
                m=esc(r["model"]), p=esc(r["prompt"]), v=esc(r.get("variant", "")),
                ok=s["ok"], er=s["errors"], avg=s["avg_s"], med=s["median_s"],
                gpu="-" if r.get("gpu_pct") is None else f"{r['gpu_pct']}%",
                load=f"{r.get('load_s', '-')}s", chars=s["avg_chars"], ptok=s["prompt_tokens"],
                filler=s["filler"], fmt=s["format_ok"],
            )
        )
    summary_table = (
        "<div class='wrap'><table><thead><tr><th>Modello</th><th>Prompt</th><th>Variante</th><th>OK/err</th>"
        "<th>s/foto</th><th>mediana s</th><th>GPU</th><th>caricamento</th><th>caratteri</th>"
        "<th>token prompt (incl. immagine)</th><th>attacco 'L'immagine...'</th><th>formato rispettato</th></tr></thead><tbody>"
        + "".join(sum_rows) + "</tbody></table></div>"
    )

    # tabella foto x run
    head = "<tr><th>Foto</th>" + "".join(f"<th>{esc(l)}</th>" for l in run_labels) + "</tr>"
    body = []
    for pi, p in enumerate(photos):
        cells = [
            "<td class='photo'><img alt='' src='data:image/jpeg;base64,{b}'><div class='name'>#{i} &middot; {n}</div></td>".format(
                b=thumb_b64(p), i=pi + 1, n=esc(rel(p))
            )
        ]
        for ri, r in enumerate(runs):
            res = r["results"][pi] if pi < len(r["results"]) else None
            if res is None:
                cells.append("<td class='cap'><span class='meta'>non eseguita</span></td>")
                continue
            if res.get("error"):
                inner = f"<div class='err'>{esc(res['error'])}</div>"
            else:
                m = res.get("meta") or {}
                inner = (
                    f"<div class='txt'>{esc(res.get('caption', ''))}</div>"
                    f"<div class='meta'>{m.get('wall_s', '-')} s &middot; {m.get('output_tokens', '-')} token in uscita</div>"
                )
            vote = (
                f"<select data-vote='{pi}|{ri}'><option value=''>voto...</option>"
                "<option value='0'>0 - errata/inventata</option><option value='1'>1 - scarsa</option>"
                "<option value='2'>2 - buona</option><option value='3'>3 - ottima</option></select>"
            )
            cells.append(f"<td class='cap'>{inner}{vote}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    votes_table = (
        "<h2 style='font-size:15px;margin:0 0 6px'>Voti (media)</h2>"
        "<div class='wrap'><table><thead><tr><th>Modello | prompt</th><th>Foto votate</th><th>Voto medio</th></tr></thead>"
        "<tbody id='votes'></tbody></table></div>"
    )

    js = (
        REPORT_JS.replace("__RUNS__", json.dumps(run_labels, ensure_ascii=False))
        .replace("__PHOTOS__", json.dumps([os.path.basename(p) for p in photos], ensure_ascii=False))
        .replace("__ID__", data["id"])
    )

    st = data["settings"]
    variants_txt = ", ".join(v["label"] for v in st.get("variants", []))
    sub = (
        f"{len(photos)} foto da {esc(str(src))} &middot; seed {st['seed']} &middot; "
        f"varianti: {esc(variants_txt)} &middot; {esc(data['created'])}"
    )
    doc = (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Banco di prova didascalie</title><style>" + REPORT_CSS + "</style></head><body>"
        "<h1>Banco di prova didascalie</h1><div class='sub'>" + sub + "</div>"
        + summary_table + votes_table
        + "<div class='wrap'><table><thead>" + head + "</thead><tbody>" + "".join(body) + "</tbody></table></div>"
        "<script>" + js + "</script></body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)


def save_results(data, out_dir):
    for r in data["runs"]:
        r["summary"] = summarize_run(r)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    build_report(data, os.path.join(out_dir, "report.html"))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Confronta modelli vision di Ollama nella generazione di didascalie.")
    ap.add_argument("--src", help="cartella da cui pescare le foto (default: prima cartella di config.json)")
    ap.add_argument("--photos", nargs="+", help="elenco esplicito di foto (ignora --src e -n)")
    ap.add_argument("-n", "--num", type=int, default=20, help="numero di foto (default 20)")
    ap.add_argument("--seed", type=int, default=42, help="seme del campionamento (stesso seme = stesse foto)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=[captioning.DEFAULT_PROMPT_VERSION],
                    help=f"versioni di prompt da confrontare (disponibili: {', '.join(captioning.PROMPTS)})")
    ap.add_argument("--max-side", type=int, default=captioning.DEFAULT_MAX_SIDE,
                    help="lato massimo in px a cui riscalare le foto (0 = originale)")
    ap.add_argument("--num-ctx", type=int, default=None, help="finestra di contesto di Ollama (default: quella del modello)")
    ap.add_argument("--variants", nargs="+", metavar="LATO[:CTX]",
                    help="varianti da confrontare, es. 1024 768:2048 (sostituisce --max-side/--num-ctx)")
    ap.add_argument("--timeout", type=int, default=240, help="timeout per foto in secondi")
    ap.add_argument("--no-think", action="store_true", help="invia think=false (modelli con ragionamento)")
    ap.add_argument("--base-url", default=captioning.OLLAMA_BASE_URL)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "bench_out"))
    args = ap.parse_args()

    for p in args.prompts:
        captioning.get_prompt(p)  # errore subito se la versione non esiste
    variants = parse_variants(args.variants, args.max_side, args.num_ctx)

    src = args.src or load_default_src()
    if args.photos:
        photos = [os.path.abspath(p) for p in args.photos]
        src = os.path.commonpath([os.path.dirname(p) for p in photos]) if photos else src
    else:
        if not src or not os.path.isdir(src):
            sys.exit(f"Cartella sorgente non valida: {src!r}. Usa --src.")
        print(f"Scansione di {src} ...")
        candidates = collect_images(src)
        if not candidates:
            sys.exit("Nessuna foto trovata.")
        photos = pick_photos(candidates, src, args.num, args.seed)
        print(f"{len(candidates)} foto trovate, ne uso {len(photos)}.")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "photos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(photos) + "\n")

    data = {
        "id": stamp,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "settings": {"src": src, "seed": args.seed, "variants": variants,
                     "models": args.models, "prompts": args.prompts, "no_think": args.no_think},
        "photos": photos,
        "runs": [],
    }
    think = False if args.no_think else None

    try:
        for model in args.models:
            print(f"\n### {model}")
            try:
                load_s = preload(args.base_url, model, args.timeout)
            except Exception as e:
                msg = str(e)
                hint = f"  -> prova: ollama pull {model}" if "not found" in msg.lower() else ""
                print(f"  SALTATO: {msg}{hint}")
                continue
            print(f"  modello caricato in {load_s}s")

            for prompt in args.prompts:
                for var in variants:
                    opts = {"num_ctx": var["num_ctx"]} if var["num_ctx"] else None
                    tag = f"{model} | {prompt} | {var['label']}"

                    # Chiamata di riscaldamento (non registrata): la prima inferenza
                    # alloca contesto e cache e sarebbe sempre la piu' lenta; con un
                    # num_ctx diverso Ollama ricarica pure il modello.
                    try:
                        captioning.caption_image(photos[0], model, prompt, options=opts,
                                                 max_side=var["max_side"], timeout=args.timeout,
                                                 think=think, base_url=args.base_url)
                    except Exception as e:
                        print(f"  [{tag}] SALTATA (errore al warm-up): {e}")
                        continue
                    gpu = gpu_percent(args.base_url, model)
                    print(f"  [{tag}] GPU: {gpu}% del modello in VRAM")

                    run = {"model": model, "prompt": prompt, "variant": var["label"],
                           "max_side": var["max_side"], "num_ctx": var["num_ctx"],
                           "load_s": load_s, "gpu_pct": gpu, "results": []}
                    data["runs"].append(run)
                    for i, photo in enumerate(photos, 1):
                        try:
                            caption, meta = captioning.caption_image(
                                photo, model, prompt, options=opts, max_side=var["max_side"],
                                timeout=args.timeout, think=think, base_url=args.base_url,
                                return_meta=True)
                            if not caption:
                                run["results"].append({"photo": photo, "error": "didascalia vuota"})
                                print(f"  [{tag}] {i}/{len(photos)} VUOTA  {short(photo)}")
                            else:
                                run["results"].append({"photo": photo, "caption": caption, "meta": meta})
                                print(f"  [{tag}] {i}/{len(photos)} {meta['wall_s']:>5}s  {short(photo)}")
                        except Exception as e:
                            run["results"].append({"photo": photo, "error": str(e)})
                            print(f"  [{tag}] {i}/{len(photos)} ERRORE {short(photo)}: {e}")
                        save_results(data, out_dir)  # salvataggio incrementale: nulla va perso se interrompi
            unload(args.base_url, model)
    except KeyboardInterrupt:
        print("\nInterrotto: salvo quanto raccolto finora.")
    finally:
        if data["runs"]:
            save_results(data, out_dir)
            print_summary(data["runs"])
            print(f"\nReport: {os.path.join(out_dir, 'report.html')}")
        else:
            print("\nNessun modello ha prodotto risultati.")


if __name__ == "__main__":
    main()
