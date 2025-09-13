# LimpiezaGPU_csv.py — versión para bodies_scraping.bodies.csv
# Usa columnas: text_raw (HTML) y _id (identificador)

import os, time, traceback
from pathlib import Path
from typing import Optional
import torch
import pandas as pd

from LimpiezaGPU import (
    load_model, focus_html, preclean_html, parse_fields, html_to_md
)

def process_row(tok, model, html: str, out_dir: Path, row_id: str) -> Optional[Path]:
    try:
        if not isinstance(html, str) or not html.strip():
            raise ValueError("text_raw vacío o no es string")
        # PASA HTML COMPLETO; focus_html ya recorta internamente
        raw_focus = focus_html(html)
        cleaned = preclean_html(raw_focus)
        # señales/heurísticas desde el HTML crudo
        fields = parse_fields(html)
        md = html_to_md(tok, model, cleaned, fields)
        out_path = out_dir / f"row_{row_id}.md"
        out_path.write_text(md, encoding="utf-8")
        return out_path
    except Exception:
        err = traceback.format_exc()
        (out_dir / f"row_{row_id}.error.txt").write_text(err, encoding="utf-8")
        print(f"[ERROR] row_{row_id}:\n{err}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ruta del CSV (p. ej. bodies_scraping.bodies.csv)")
    ap.add_argument("-o","--out", default="tests_md_csv", help="carpeta de salida")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Carga modelo/Tokenizer una sola vez
    tok, model = load_model()

    # Lee como string para no perder HTML
    df = pd.read_csv(in_path, dtype=str, on_bad_lines="skip")

    ok = fail = 0
    t0_all = time.time()
    for idx, row in df.iterrows():
        html = row.get("text_raw", "") or ""
        row_id = row.get("_id", str(idx)) or str(idx)

        t0 = time.time()
        print(f"[RUN] row {row_id} …")
        p = process_row(tok, model, html, out_dir, str(row_id))
        dt = time.time() - t0

        if p:
            print(f"[OK]   row {row_id} → {p.name}  ({dt:.1f}s)")
            ok += 1
        else:
            print(f"[FAIL] row {row_id}  ({dt:.1f}s)")
            fail += 1

    dt_all = time.time() - t0_all
    print(f"\nHecho. Éxitos: {ok} | Fallos: {fail} | Total: {ok+fail} | "
          f"Salida: {out_dir.resolve()} | {dt_all:.1f}s")

if __name__ == "__main__":
    main()
