import os
import json
import time
import traceback
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from datetime import datetime
from LimpiezaGPU import load_model, focus_html, preclean_html, parse_fields, html_to_md
import torch

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def to_str_id(_id):
    try:
        return str(_id)
    except Exception:
        return f"{_id!r}"

def process_html(tok, model, html_raw):
    t0 = time.time()
    raw_focus = focus_html(html_raw)
    cleaned = preclean_html(raw_focus)
    fields = parse_fields(html_raw)
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
    # Config desde .env o valores por defecto
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27018")
    DB_ORIGEN = os.getenv("MONGO_DB", "raw_productos")
    DB_DESTINO = os.getenv("CLEAN_DB", "clean_productos")  # puedes definir CLEAN_DB en .env
    BODY_FIELD = os.getenv("BODY_FIELD", "data.html_raw")
    ID_FIELD = os.getenv("ID_FIELD", "_id")
    QUERY = os.getenv("QUERY", '{"body_cleaned_markdown": {"$exists": false}}')

    try:
        query = json.loads(QUERY)
    except Exception as e:
        raise SystemExit(f"[ERROR] QUERY inválida: {e}")

    client = MongoClient(MONGO_URI)
    db_src = client[DB_ORIGEN]
    db_dst = client[DB_DESTINO]

    print(f"Procesando colecciones desde BD: {DB_ORIGEN}")
    print(f"Destino: {DB_DESTINO}")

    tok, model = load_model()

    for col_name in db_src.list_collection_names():
        print(f"\n🟡 Procesando colección: {col_name}")
        src_col = db_src[col_name]
        dst_col = db_dst[col_name]

        try:
            dst_col.create_index([("source_id", ASCENDING)], name="uniq_source_id", unique=True)
        except:
            pass

        cursor = src_col.find(query, no_cursor_timeout=True)
        total = ok = fail = skipped = 0
        t0 = time.time()

        try:
            for doc in cursor:
                total += 1
                src_id = doc.get(ID_FIELD, doc.get("_id"))
                src_id_str = to_str_id(src_id)
                html_raw = doc.get("data", {}).get("html_raw", "")

                if not isinstance(html_raw, str) or not html_raw.strip():
                    print(f"[SKIP] {src_id_str} (sin HTML)")
                    skipped += 1
                    continue

                existing = dst_col.find_one({"source_id": src_id})
                if existing:
                    print(f"[SKIP] {src_id_str} (ya existe en destino)")
                    skipped += 1
                    continue

                print(f"[RUN]  {src_id_str}")
                try:
                    result = process_html(tok, model, html_raw)

                    out_doc = {
                        "source_id": src_id,
                        "source_collection": col_name,
                        "source_db": DB_ORIGEN,
                        "created_at": now_iso(),
                        "body_cleaned_markdown": result["body_cleaned_markdown"],
                        "clean_fields": result["clean_fields"],
                        "clean_stats": result["clean_stats"]
                        
                    }

                    # Copiar metadatos adicionales si existen
                    for field in ["producto", "marca", "pais", "retail", "url", "fecha_scraping"]:
                        if field in doc:
                            out_doc[field] = doc[field]

                    dst_col.insert_one(out_doc)
                    print(f"[OK]   {src_id_str}")
                    ok += 1

                except Exception as e:
                    err = traceback.format_exc()
                    print(f"[FAIL] {src_id_str}\n{err}")
                    fail += 1
                    try:
                        dst_col.insert_one({
                            "source_id": src_id,
                            "source_collection": col_name,
                            "source_db": DB_ORIGEN,
                            "created_at": now_iso(),
                            "clean_error": {
                                "message": str(e),
                                "traceback": err
                            }
                        })
                    except:
                        pass

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            cursor.close()

        dt = time.time() - t0
        print(f"✔️ Colección {col_name}: Total={total} | OK={ok} | Skipped={skipped} | Fail={fail} | Tiempo={dt:.1f}s")

if __name__ == "__main__":
    main()
