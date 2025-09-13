# LimpiezaGPU.py — GPU 4 GB: chunking por tokens + HINTS VTEX + salida <md>…</md> + anti-OOM
import os, re, time, traceback
from pathlib import Path
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# ========= Config =========
MODEL_ID = "jinaai/reader-lm-0.5b"     # luego puedes probar ReaderLM-v2 (ideal con 4-bit)
MAX_HTML_CHARS    = 14_000             # recorte por caracteres (agresivo para 4 GB)
MAX_TOKENS_PROMPT = 896                # tokens totales de entrada por chunk (prompt + HINTS + HTML)
MAX_NEW_TOKENS    = 128                # salida corta (reduce KV-cache)
FOCUS_WINDOW      = 2500               # ventana alrededor de matches útiles

PROMPT_TMPL = """Convierte el siguiente HTML en Markdown limpio para análisis de producto.

Instrucciones:
- Conserva únicamente información del producto (nombre, marca, cantidad/unidad, precio, descripción, referencia).
- Elimina menús, footers, anuncios, scripts y botones sociales.
- Usa Markdown con títulos (## para nombre de producto).
- Representa el precio en formato **Precio:** $XX.XX (si aparece).
- No inventes información.
- Si un dato no aparece en el HTML, escribe N/D.
- Devuelve la respuesta EXCLUSIVAMENTE entre <md> y </md>, sin repetir estas instrucciones ni mostrar el bloque HTML.

HINTS (si aparecen, respétalos):
{HINTS}

HTML:
{HTML}

<md>
"""

# ========= Utilidades comunes =========
def html_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ========= Pre-limpieza agresiva =========
CLEAN_PATTERNS = [
    (r"<!--.*?-->", ""),
    (r"<script\b[^>]*>.*?</script>", ""),
    (r"<style\b[^>]*>.*?</style>", ""),
    (r"<svg\b[^>]*>.*?</svg>", ""),
    (r"\s{2,}", " "),
]
_clean = [(re.compile(p, re.S | re.I), repl) for p, repl in CLEAN_PATTERNS]

def preclean_html(raw: str) -> str:
    for rx, repl in _clean:
        raw = rx.sub(repl, raw)
    for cls in ["breadcrumb","share","relatedProducts","carousel","swiper","drawer",
                "searchBar","shelf","header","footer"]:
        raw = re.sub(rf"<[^>]*class=['\"][^\"']*{cls}[^\"']*['\"][^>]*>.*?</[^>]+>",
                     "", raw, flags=re.S|re.I)
    return raw

def focus_html(raw: str) -> str:
    pats = [r'Precio', r'\$\s?\d', r'currencyInteger', r'currencyFraction',
            r'productName', r'productBrandName', r'brand', r'Referencia',
            r'product-identifier', r'productDescription']
    seen = []
    chunks = []
    for pat in pats:
        m = re.search(pat, raw, re.I)
        if m:
            i = max(0, m.start() - FOCUS_WINDOW); j = min(len(raw), m.end() + FOCUS_WINDOW)
            key = (i//1000, j//1000)
            if key not in seen:
                chunks.append(raw[i:j]); seen.append(key)
    s = "\n\n".join(chunks) if chunks else raw
    return s[:MAX_HTML_CHARS]

# ========= Heurísticos VTEX (pre-extracción) =========
def parse_fields(raw: str) -> Dict[str, str]:
    fields = {"nombre":"", "marca":"", "precio":"", "referencia":"", "cantidad":"", "unidad":"", "descripcion":""}

    # Marca
    m = re.search(r'class="[^"]*productBrandName[^"]*">([^<]+)', raw, re.I)
    if m: fields["marca"] = m.group(1).strip()

    # Nombre (prefiere <h1>)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.S|re.I)
    if m:
        fields["nombre"] = html_text(m.group(1))
    else:
        # Breadcrumb término final
        m = re.search(r'class="[^"]*breadcrumb[^"]*term[^"]*"[^>]*>.*?<span[^>]*>([^<]+)</span>', raw, re.S|re.I)
        if m: fields["nombre"] = m.group(1).strip()

    # Referencia
    m = re.search(r'product-identifier__value">([^<]+)', raw, re.I)
    if m: fields["referencia"] = m.group(1).strip()

    # Precio (entero + fracción)
    mi = re.search(r'currencyInteger[^<]*>(\d+)', raw, re.I)
    mf = re.search(r'currencyFraction[^<]*>(\d{2})', raw, re.I)
    if mi and mf:
        fields["precio"] = f"${mi.group(1)}.{mf.group(1)}"
    else:
        # Fallback $xx.xx
        m = re.search(r"\$\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))", raw)
        if m:
            amt = m.group(1).replace(".", "").replace(",", ".")
            fields["precio"] = f"${amt}"

    # Descripción
    m = re.search(r'class="[^"]*(?:productDescriptionText|tabDescriptionValue)[^"]*"[^>]*>(.*?)</', raw, re.S|re.I)
    if m:
        fields["descripcion"] = html_text(m.group(1))[:300]

    # Cantidad/Unidad desde nombre (p.ej. "900g", "1 kg", "500 ml")
    if fields["nombre"]:
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|g|l|ml|un|u|unidad(?:es)?|pack|pza(?:s)?)\b', fields["nombre"], re.I)
        if m:
            fields["cantidad"] = m.group(1).replace(",", ".")
            unit = m.group(2).lower()
            unit = {"unidad":"un","unidades":"un","u":"un","pza":"pza","pzas":"pza"}.get(unit, unit)
            fields["unidad"] = unit

    return fields

def build_hints(fields: Dict[str,str]) -> str:
    lines = []
    for k,label in [("nombre","Nombre"),("marca","Marca"),("precio","Precio"),
                    ("referencia","Referencia"),("cantidad","Cantidad"),("unidad","Unidad"),
                    ("descripcion","Descripcion")]:
        v = fields.get(k,"").strip()
        if v:
            if k == "descripcion" and len(v) > 160:
                v = v[:160] + "…"
            lines.append(f"- {label}: {v}")
    s = "\n".join(lines)
    # recorte de seguridad
    return s[:400]

# ========= Ajustes CUDA/atención =========
os.environ.setdefault("PYTORCH_SDP_ATTENTION", "0")              # usa 'eager'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

def _disable_sdp():
    try:
        from torch.nn.attention import sdpa_kernel
        sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            pass

# ========= Carga del modelo =========
def load_model(model_id=MODEL_ID):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    max_memory = {"cpu": "8GiB"}
    if use_cuda:
        max_memory[0] = "3.0GiB"  # deja >1 GiB libre al sistema de video

    _disable_sdp()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    model.config.use_cache = False  # reduce KV-cache

    print(f"[INFO] CUDA disponible: {use_cuda}")
    if use_cuda:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | dtype={dtype}")

    return tok, model

# ========= Tokens utils =========
def count_tokens(tok, s: str) -> int:
    return tok(s, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]

def chunk_html_by_tokens(tok, html: str, max_prompt_tokens: int, hints: str) -> List[str]:
    """Ajusta el tamaño del HTML por chunk considerando HINTS + prompt fijo."""
    prefix = PROMPT_TMPL.format(HTML="", HINTS=hints)
    prefix_toks = count_tokens(tok, prefix)
    max_html_tokens = max(96, max_prompt_tokens - prefix_toks - 16)  # margen
    ids = tok(html, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(ids), max_html_tokens):
        sub_ids = ids[i:i+max_html_tokens]
        chunk = tok.decode(sub_ids, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks or [""]

# ========= Post-proceso Markdown =========
def extract_md(txt: str) -> str:
    start = txt.find("<md>"); end = txt.rfind("</md>")
    if start != -1 and end != -1 and end > start:
        return txt[start+4:end].strip()
    if start == -1 and end != -1:
        return txt[:end].strip()
    cut = txt.rfind("Markdown:")
    if cut != -1:
        txt = txt[cut + len("Markdown:"):]
    # Filtra líneas de prompt/HTML
    drop_keys = [
        "Conserva únicamente información del producto",
        "Elimina menús", "Usa Markdown", "Representa el precio",
        "No inventes información", "Devuelve la respuesta",
        "HTML:", "<html", "</html", "<head", "<body", "class=", "vtex-", "<svg", "</svg"
    ]
    lines = []
    for ln in txt.splitlines():
        if any(k in ln for k in drop_keys):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def sanitize_md(md: str) -> str:
    md = re.sub(r"<[^>]+>", "", md)             # quita cualquier etiqueta residual
    md = "\n".join(ln for ln in md.splitlines() if "vtex-" not in ln)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

def ensure_min_fields(md: str, fields: Dict[str,str]) -> str:
    """Garantiza presencia de las 5 claves: Precio, Nombre, Cantidad, Unidad, Marca (y referencia opcional)."""
    def has(label): return re.search(rf"^\*\*{label}:\*\*", md, re.I|re.M) is not None
    out = md

    # Asegura título con nombre
    if not re.search(r"^##\s+", out, re.M):
        name = fields.get("nombre","") or "Producto"
        out = f"## {name}\n\n" + out

    # Marca
    if not has("Marca") and fields.get("marca"):
        out += f"\n**Marca:** {fields['marca']}"
    # Cantidad / Unidad
    if not has("Cantidad") or not has("Unidad"):
        if fields.get("cantidad") or fields.get("unidad"):
            qty = fields.get("cantidad","N/D")
            unit = fields.get("unidad","N/D")
            if not has("Cantidad"): out += f"\n**Cantidad:** {qty}"
            if not has("Unidad"):   out += f"\n**Unidad:** {unit}"
    # Precio
    if not has("Precio"):
        price = fields.get("precio","N/D")
        out += f"\n**Precio:** {price}"
    # Referencia (opcional pero útil)
    if "Referencia" not in out and fields.get("referencia"):
        out += f"\n**Referencia:** {fields['referencia']}"

    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

# ========= Stopping criteria para </md> =========
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: List[int]): self.stop_ids = stop_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        if len(seq) < len(self.stop_ids): return False
        return seq[-len(self.stop_ids):] == self.stop_ids

# ========= Generación segura (con retry) =========
def generate_md_from_chunk(tok, model, html_chunk: str, hints: str,
                           max_prompt_tokens: int, max_new_tokens: int) -> str:
    prompt = PROMPT_TMPL.format(HTML=html_chunk, HINTS=hints)

    stop_ids = tok("</md>", add_special_tokens=False)["input_ids"]
    stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

    def _run(mpt, mnt):
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=mpt)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        print(f"[INFO] tokens_entrada={inputs['input_ids'].shape[1]}, max_new_tokens={mnt}")
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=mnt,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopping,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
        md_only = sanitize_md(extract_md(txt))
        return md_only

    try:
        return _run(max_prompt_tokens, max_new_tokens)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print("[WARN] OOM en generación. Reintentando con límites más bajos…")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _run(max(384, max_prompt_tokens // 2), max(64, max_new_tokens // 2))

# ========= Pipeline de archivo =========
def html_to_md(tok, model, html: str, fields: Dict[str,str]) -> str:
    hints = build_hints(fields)
    chunks = chunk_html_by_tokens(tok, html, MAX_TOKENS_PROMPT, hints)
    outputs = []
    for idx, ch in enumerate(chunks, 1):
        print(f"[CHUNK] {idx}/{len(chunks)}")
        md_part = generate_md_from_chunk(tok, model, ch, hints, MAX_TOKENS_PROMPT, MAX_NEW_TOKENS)
        if md_part:
            outputs.append(md_part)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    md = "\n\n".join(outputs)
    md = ensure_min_fields(md, fields)
    return md

def process_file(tok, model, in_path: Path, out_dir: Path) -> Optional[Path]:
    try:
        raw = in_path.read_text(encoding="utf-8", errors="ignore")
        raw_focus = focus_html(raw[:MAX_HTML_CHARS])
        cleaned = preclean_html(raw_focus)
        fields = parse_fields(raw)   # usa el HTML original para extraer señales
        md = html_to_md(tok, model, cleaned, fields)
        out_path = out_dir / (in_path.stem + ".md")
        out_path.write_text(md, encoding="utf-8")
        return out_path
    except Exception:
        err = traceback.format_exc()
        (out_dir / (in_path.stem + ".error.txt")).write_text(err, encoding="utf-8")
        print(f"[ERROR] {in_path.name}:\n{err}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ========= Main =========
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="archivo .html o carpeta")
    ap.add_argument("-o","--out", default="tests_md", help="carpeta de salida")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # CUDA/atención
    os.environ.setdefault("PYTORCH_SDP_ATTENTION", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    # Carga
    tok, model = load_model(MODEL_ID)

    files = [in_path] if in_path.is_file() else sorted(in_path.rglob("*.html"))
    ok = fail = 0
    for f in files:
        t0 = time.time()
        print(f"[RUN] {f.name} …")
        p = process_file(tok, model, f, out_dir)
        dt = time.time() - t0
        if p: print(f"[OK]   {f.name} → {p.name}  ({dt:.1f}s)"); ok += 1
        else: print(f"[FAIL] {f.name}  ({dt:.1f}s)"); fail += 1

    print(f"\nHecho. Éxitos: {ok} | Fallos: {fail} | Salida: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
