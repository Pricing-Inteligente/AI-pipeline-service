# LimpiezaGPU_mongo.py — Lee desde Mongo y escribe resultados limpios en la misma colección
# Depende de: pymongo, transformers, torch, pandas (no obligatorio aquí), tu LimpiezaGPU.py

import os, time, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from pymongo import MongoClient
from bson import ObjectId

import torch

# Reusamos tu pipeline
from LimpiezaGPU import (
    load_model, focus_html, preclean_html, parse_fields, html_to_md
)

# ---------- Utilidades ----------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def to_str_id(_id) -> str:
    try:
        return str(_id)
    except Exception:
        return f"{_id!r}"

def process_html(tok, model, html_raw: str) -> Dict[str, Any]:
    """
    Ejecuta el pipeline de limpieza sobre un HTML crudo.
    Retorna dict con:
      - body_cleaned_markdown
      - clean_fields (lo que parse_fields detectó + rellenado posterior)
      - clean_stats (duraciones, longitudes)
    """
    t0 = time.time()

    # 1) focus + pre-clean
    # SUGERENCIA: pasa todo el HTML, deja que focus_html recorte adentro
    raw_focus = focus_html(html_raw)
    cleaned = preclean_html(raw_focus)

    # 2) señales/heurísticas iniciales
    fields_initial = parse_fields(html_raw)  # usa el html crudo para señales VTEX-like

    # 3) LLM → Markdown
    md = html_to_md(tok, model, cleaned, fields_initial)

    t1 = time.time()
    return {
        "body_cleaned_markdown": md,
        "clean_fields": {k: v for k, v in fields_initial.items()},
        "clean_stats": {
            "started_at": now_iso(),
            "elapsed_sec": round(t1 - t0, 3),
            "len_html_raw": len(html_raw or ""),
            "len_cleaned": len(md or ""),
            "compression_ratio_chars": round((len(md or "") / max(1, len(html_raw or ""))), 4),
        }
    }

# ---------- Runner principal ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Limpieza de bodies directamente desde MongoDB")
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                    help="URI de Mongo (env MONGO_URI)")
    ap.add_argument("--db", default=os.getenv("MONGO_DB", "crawler_db"),
                    help="Nombre de la base (env MONGO_DB)")
    ap.add_argument("--col", default=os.getenv("MONGO_COL", "pages"),
                    help="Nombre de la colección (env MONGO_COL)")

    # Campos de tu colección (ajústalos si difieren)
    ap.add_argument("--body-field", default=os.getenv("BODY_FIELD", "text_raw"),
                    help="Campo del HTML crudo (por defecto text_raw)")
    ap.add_argument("--id-field", default=os.getenv("ID_FIELD", "_id"),
                    help="Campo identificador (por defecto _id)")

    # Filtro y control
    ap.add_argument("--query", default=os.getenv("QUERY", '{"body_cleaned_markdown":{"$exists":false}}'),
                    help='Filtro JSON para seleccionar docs (ej. \'{"body_cleaned_markdown":{"$exists":false}}\')')
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")),
                    help="Máximo de documentos a procesar (0 = sin límite)")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "200")),
                    help="Tamaño del batch del cursor (streaming)")
    ap.add_argument("--force", action="store_true",
                    help="Ignora el filtro de existencia de body_cleaned_markdown y procesa igual")
    ap.add_argument("--dry-run", action="store_true",
                    help="Ejecuta la limpieza pero NO guarda cambios en Mongo")
    ap.add_argument("--quiet", action="store_true",
                    help="Menos logs")
    args = ap.parse_args()

    # Conexión Mongo
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    col = db[args.col]

    # Carga modelo/tokenizer una sola vez
    tok, model = load_model()

    # Construir filtro
    import json
    try:
        user_query = json.loads(args.query) if args.query else {}
    except Exception as e:
        raise SystemExit(f"[ERROR] QUERY inválida: {e}")

    final_query = dict(user_query)
    if not args.force:
        # Asegura que si no se fuerza, solo se tomen docs sin limpieza previa
        final_query.setdefault("body_cleaned_markdown", {"$exists": False})

    projection = {args.body_field: 1, args.id_field: 1}

    total = 0
    ok = 0
    fail = 0
    t_global = time.time()

    cursor = col.find(final_query, projection=projection, no_cursor_timeout=True, batch_size=args.batch_size)

    try:
        for doc in cursor:
            if args.limit and total >= args.limit:
                break
            total += 1

            _id = doc.get(args.id_field, doc.get("_id"))
            _id_str = to_str_id(_id)

            html_raw = doc.get(args.body_field, "")
            if not isinstance(html_raw, str) or not html_raw.strip():
                if not args.quiet:
                    print(f"[SKIP] {_id_str} (sin {args.body_field} válido)")
                continue

            t0 = time.time()
            if not args.quiet:
                print(f"[RUN] {_id_str} …")

            try:
                result = process_html(tok, model, html_raw)
                result["clean_stats"]["finished_at"] = now_iso()

                if not args.dry_run:
                    # Escribe in-place
                    update_doc = {
                        "$set": {
                            "body_cleaned_markdown": result["body_cleaned_markdown"],
                            "clean_fields": result["clean_fields"],
                            "clean_stats": result["clean_stats"],
                        }
                    }
                    col.update_one({args.id_field: _id} if args.id_field != "_id" else {"_id": _id}, update_doc)

                if not args.quiet:
                    dt = time.time() - t0
                    print(f"[OK]  {_id_str}  ({dt:.1f}s)")
                ok += 1

            except Exception:
                fail += 1
                err = traceback.format_exc()
                if not args.quiet:
                    print(f"[FAIL] {_id_str}\n{err}")
                if not args.dry_run:
                    # Persistimos el error para inspección
                    col.update_one({args.id_field: _id} if args.id_field != "_id" else {"_id": _id},
                                   {"$set": {
                                       "clean_error": {
                                           "message": str(err).splitlines()[-1],
                                           "traceback": err,
                                           "at": now_iso()
                                       }
                                   }})

            # Libera memoria entre docs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        cursor.close()

    dt_global = time.time() - t_global
    print(f"\nHecho. Total: {total} | Éxitos: {ok} | Fallos: {fail} | Tiempo: {dt_global:.1f}s")

if __name__ == "__main__":
    main()
