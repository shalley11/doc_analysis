from redis import Redis
from rq import Worker, Queue

from doc_analysis.logging_config import get_worker_logger

logger = get_worker_logger()

redis_conn = Redis(host="localhost", port=6379)

queues = [
    Queue("pdf_ingestion", connection=redis_conn)
]

if __name__ == "__main__":
    logger.info("RQ Worker starting...")
    logger.info(f"Listening on queues: {[q.name for q in queues]}")

    worker = Worker(queues)
    worker.work()
