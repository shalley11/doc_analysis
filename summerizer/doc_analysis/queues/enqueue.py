from redis import Redis
from rq import Queue

from doc_analysis.workers.pdf_ingestion import ingest_batch_pdfs

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("pdf_ingestion", connection=redis_conn)


def enqueue_pdf_ingestion(
    batch_id: str,
    filenames: list,
    paths: list,
    embedding_dim: int,
    ttl_seconds: int,
    start_page: int = None,
    end_page: int = None
):
    queue.enqueue(
        ingest_batch_pdfs,
        batch_id,
        filenames,
        paths,
        embedding_dim,
        ttl_seconds,
        start_page,
        end_page,
        job_timeout=1800
    )
