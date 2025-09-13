# LimpiezaGPU_mongo_to_collection.py
# Lee HTML crudo desde una colección origen y guarda un documento NUEVO en una colección destino (clean_bodies)
# Requiere: pymongo, torch, transformers y tu LimpiezaGPU.py

import os, time, traceback
from datetime import datetime
from typing import Dict, Any

from pymongo import MongoClient, ASCENDING
from bson import ObjectId
import torch

from LimpiezaGPU import (
    load_model, focus_html, preclean_html, parse_fields, html_to_md
)

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def to_str_id(_id) -> str:
    try:
        return str(_id)
    except Exception:
        return f"{_id!r}"

def process_html(tok, model, html_raw: str) -> Dict[str, Any]:
    t0 = time.time()
    raw_focus = focus_html(html_raw)      # pasar HTML completo; recorta internamente
    cleaned = preclean_html(raw_focus)
    fields = parse_fields(html_raw)       # señales a partir del HTML original
    md = html_to_md(tok, model, cleaned, fields)
    t1 = time.time()

    return {
        "body_cleaned_markdown": md,
        "clean_fields": fields,
        "clean_stats": {
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_sec": round(t1 - t0, 3),
            "len_html_raw": len(html_raw or ""),
            "len_cleaned": len(md or ""),
            "compression_ratio_chars": round((len(md or "") / max(1, len(html_raw or ""))), 4),
        }
    }

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Limpia bodies desde una colección y los inserta en otra colección destino")
    # Conexión
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default=os.getenv("MONGO_DB", "crawler_db"))
    ap.add_argument("--src-col", default=os.getenv("SRC_COL", "pages"))
    ap.add_argument("--dst-col", default=os.getenv("DST_COL", "clean_bodies"))

    # Campos
    ap.add_argument("--body-field", default=os.getenv("BODY_FIELD", "text_raw"), help="Campo con el HTML crudo")
    ap.add_argument("--id-field", default=os.getenv("ID_FIELD", "_id"), help="Campo ID en la colección de origen")

    # Filtros / control
    ap.add_argument("--query", default=os.getenv("QUERY", "{}"),
                    help='Filtro JSON para la colección origen (ej. \'{"domain":"retail.com"}\')')
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")), help="Máximo de docs (0 = sin límite)")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "200")))
    ap.add_argument("--skip-existing", action="store_true",
                    help="Saltar si ya existe en destino un registro para ese source_id (por defecto: False)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Si existe en destino, sobreescribe (incompatible con --skip-existing)")
    ap.add_argument("--copy-meta", default=os.getenv("COPY_META", "url,domain"),
                    help="Lista de campos meta a copiar del doc origen (coma-separado). Ej: url,domain, Fecha")
    ap.add_argument("--include-raw", action="store_true",
                    help="Guardar también el HTML crudo en destino (ocupa más espacio)")
    ap.add_argument("--dry-run", action="store_true", help="No escribe en destino (solo simula)")
    ap.add_argument("--quiet", action="store_true", help="Menos logs")
    args = ap.parse_args()

    # Validaciones rápidas
    if args.skip_existing and args.overwrite:
        raise SystemExit("--skip-existing y --overwrite no pueden usarse juntos.")

    try:
        user_query = json.loads(args.query) if args.query else {}
    except Exception as e:
        raise SystemExit(f"[ERROR] QUERY inválida: {e}")

    # Conexión Mongo
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    src = db[args.src_col]
    dst = db[args.dst_col]

    # Índice recomendado en destino para evitar duplicados por source_id
    try:
        dst.create_index([("source_id", ASCENDING)], name="uniq_source_id", unique=True)
    except Exception:
        pass

    # Carga modelo 1 sola vez
    tok, model = load_model()

    projection = {args.body_field: 1, args.id_field: 1}
    # incluir también los metacampos a copiar si existen
    meta_fields = [x.strip() for x in (args.copy_meta or "").split(",") if x.strip()]
    for mf in meta_fields:
        projection[mf] = 1

    total = ok = fail = skipped = 0
    t_global = time.time()

    cursor = src.find(user_query, projection=projection, no_cursor_timeout=True, batch_size=args.batch_size)

    try:
        for doc in cursor:
            if args.limit and total >= args.limit:
                break
            total += 1

            src_id = doc.get(args.id_field, doc.get("_id"))
            src_id_str = to_str_id(src_id)
            html_raw = doc.get(args.body_field, "")

            if not isinstance(html_raw, str) or not html_raw.strip():
                if not args.quiet:
                    print(f"[SKIP] {src_id_str} (sin {args.body_field})")
                skipped += 1
                continue

            # Revisar existencia en destino
            existing = dst.find_one({"source_id": src_id})
            if existing and args.skip_existing:
                if not args.quiet:
                    print(f"[SKIP] {src_id_str} (ya existe en {args.dst_col})")
                skipped += 1
                continue

            if not args.quiet:
                print(f"[RUN]  {src_id_str} …")

            try:
                result = process_html(tok, model, html_raw)

                # Documento destino
                out_doc = {
                    "source_id": src_id,               # vínculo al doc original
                    "source_collection": args.src_col,
                    "source_db": args.db,
                    "created_at": now_iso(),
                    "body_cleaned_markdown": result["body_cleaned_markdown"],
                    "clean_fields": result["clean_fields"],
                    "clean_stats": result["clean_stats"]
                }

                # Copiar metadatos seleccionados si existen en origen
                for mf in meta_fields:
                    if mf in doc:
                        out_doc[mf] = doc[mf]

                # Opcional: guardar HTML crudo
                if args.include_raw:
                    out_doc["text_raw"] = html_raw

                if not args.dry_run:
                    if existing and args.overwrite:
                        dst.update_one({"_id": existing["_id"]}, {"$set": out_doc})
                    else:
                        dst.insert_one(out_doc)

                if not args.quiet:
                    print(f"[OK]   {src_id_str}")
                ok += 1

            except Exception:
                fail += 1
                err = traceback.format_exc()
                if not args.quiet:
                    print(f"[FAIL] {src_id_str}\n{err}")
                if not args.dry_run:
                    # registrar un doc de error separado o anexar un campo error en destino (opcional)
                    try:
                        dst.insert_one({
                            "source_id": src_id,
                            "source_collection": args.src_col,
                            "source_db": args.db,
                            "created_at": now_iso(),
                            "clean_error": {
                                "message": str(err).splitlines()[-1],
                                "traceback": err
                            }
                        })
                    except Exception:
                        pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        cursor.close()

    dt = time.time() - t_global
    print(f"\nHecho. Total: {total} | OK: {ok} | Fail: {fail} | Skip: {skipped} | Tiempo: {dt:.1f}s")

if __name__ == "__main__":
    main()
