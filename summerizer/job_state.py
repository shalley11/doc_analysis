import json
import redis
from typing import Dict

redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=True)


def _key(batch_id: str) -> str:
    return f"pdf:batch:{batch_id}"


def init_job(batch_id: str, rq_job_id: str):
    data = {
        "state": "queued",
        "progress": 0,
        "total_pages": 0,
        "processed_pages": 0,
        "chunk_count": 0,
        "milvus_indexed": False,
        "rq_job_id": rq_job_id,
        "pdfs": {}
    }
    redis_conn.set(_key(batch_id), json.dumps(data))


def init_pdf(batch_id: str, pdf_name: str, total_pages: int):
    data = json.loads(redis_conn.get(_key(batch_id)))
    data["pdfs"][pdf_name] = {
        "total_pages": total_pages,
        "processed_pages": 0,
        "progress": 0,
        "chunk_count": 0,
        "status": "running"
    }
    redis_conn.set(_key(batch_id), json.dumps(data))


def update_job(batch_id: str, **updates):
    data = json.loads(redis_conn.get(_key(batch_id)))
    data.update(updates)

    if data.get("total_pages", 0) > 0:
        data["progress"] = int(
            (data["processed_pages"] / data["total_pages"]) * 100
        )

    redis_conn.set(_key(batch_id), json.dumps(data))


def update_pdf(batch_id: str, pdf_name: str, **updates):
    data = json.loads(redis_conn.get(_key(batch_id)))
    pdf = data["pdfs"][pdf_name]
    pdf.update(updates)

    if pdf["total_pages"] > 0:
        pdf["progress"] = int(
            (pdf["processed_pages"] / pdf["total_pages"]) * 100
        )

    redis_conn.set(_key(batch_id), json.dumps(data))


def get_job(batch_id: str) -> Dict | None:
    data = redis_conn.get(_key(batch_id))
    return json.loads(data) if data else None
