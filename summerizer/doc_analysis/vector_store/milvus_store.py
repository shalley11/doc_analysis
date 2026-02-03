"""
Simple Milvus store for vector operations.
"""
from typing import Optional, List
from pymilvus import Collection, connections
from doc_analysis.config import MILVUS_HOST, MILVUS_PORT


def _ensure_connection():
    """Ensure Milvus connection exists."""
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


class MilvusStore:
    def __init__(self, collection):
        _ensure_connection()
        if isinstance(collection, str):
            self.col = Collection(collection)
            self.col.load()
        else:
            self.col = collection

    def insert_chunks(self, records):
        self.col.insert(records)
        self.col.flush()

    def query_chunks(self, pdf_name: Optional[str] = None, limit: int = 10000) -> List[dict]:
        """
        Query chunks from collection, optionally filtered by pdf_name.
        Returns chunks sorted by page_no and chunk_seq.
        """
        if pdf_name:
            expr = f'pdf_name == "{pdf_name}"'
        else:
            expr = ""

        results = self.col.query(
            expr=expr,
            output_fields=[
                "text",
                "pdf_id",
                "pdf_name",
                "page_no",
                "chunk_seq",
                "prev_chunk_id",
                "next_chunk_id",
                "content_type",
                "table_id",
                "table_part",
                "summary",
            ],
            limit=limit
        )

        # Sort by page_no, then chunk_seq
        results.sort(key=lambda x: (x.get("page_no", 0), x.get("chunk_seq", 0)))
        return results

    def get_pdf_names(self) -> List[str]:
        """Get list of unique PDF names in the collection."""
        results = self.col.query(
            expr="",
            output_fields=["pdf_name"],
            limit=10000
        )
        return list(set(r["pdf_name"] for r in results if r.get("pdf_name")))

    def search(self, embedding, k: int = 8):
        return self.col.search(
            data=[embedding],
            anns_field="embedding",
            param={
                "metric_type": "IP",
                "params": {"ef": 64}
            },
            limit=k,
            output_fields=[
                "text",
                "pdf_id",
                "pdf_name",
                "page_no",
                "chunk_seq",
                "prev_chunk_id",
                "next_chunk_id",
                "content_type",
                "word_count",
                "image_path",
                "table_id",
                "table_part",
                "table_total_parts",
                "summary",
            ]
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """Get a specific chunk by its ID."""
        try:
            results = self.col.query(
                expr=f'chunk_id == "{chunk_id}"',
                output_fields=[
                    "text",
                    "pdf_id",
                    "pdf_name",
                    "page_no",
                    "chunk_seq",
                    "prev_chunk_id",
                    "next_chunk_id",
                    "content_type",
                    "summary",
                ],
                limit=1
            )
            return results[0] if results else None
        except Exception:
            return None

    def get_context_chunks(self, chunk_id: str) -> dict:
        """
        Get a chunk with its prev and next chunks for context.
        Useful for RAG to provide surrounding context.
        """
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return {"current": None, "prev": None, "next": None}

        prev_chunk = None
        next_chunk = None

        if chunk.get("prev_chunk_id"):
            prev_chunk = self.get_chunk_by_id(chunk["prev_chunk_id"])

        if chunk.get("next_chunk_id"):
            next_chunk = self.get_chunk_by_id(chunk["next_chunk_id"])

        return {
            "current": chunk,
            "prev": prev_chunk,
            "next": next_chunk
        }

    def insert_summary(self, summary_chunk: dict) -> None:
        """
        Insert a summary chunk into the collection.
        Summary chunks have content_type='summary'.
        """
        # Get embedding dimension from existing collection
        zero_vector = [0.0] * 1024  # Default E5 dimension

        record = {
            "chunk_id": summary_chunk.get("chunk_id", ""),
            "embedding": summary_chunk.get("embedding", zero_vector),
            "text": summary_chunk.get("text", ""),
            "pdf_id": summary_chunk.get("pdf_id", "summary"),
            "pdf_name": summary_chunk.get("pdf_name", ""),
            "page_no": 0,
            "chunk_seq": -1,  # Special sequence for summaries
            "prev_chunk_id": "",
            "next_chunk_id": "",
            "content_type": "summary",
            "word_count": summary_chunk.get("word_count", 0),
            "image_path": "",
            "table_id": 0,
            "table_part": 0,
            "table_total_parts": 0,
            "summary": "",  # Document summaries don't have vision summaries
            "summary_embedding": zero_vector,
        }

        try:
            self.col.insert([record])
            self.col.flush()
        except Exception as e:
            raise e

    def get_summary(
        self,
        pdf_name: Optional[str] = None,
        summary_type: Optional[str] = None
    ) -> Optional[dict]:
        """
        Retrieve a stored summary from the collection.

        Args:
            pdf_name: PDF name or None for combined summary
            summary_type: brief, bulletwise, detailed, or executive

        Returns:
            Summary data or None if not found
        """
        try:
            expr_parts = ['content_type == "summary"']

            if pdf_name:
                expr_parts.append(f'pdf_name == "{pdf_name}"')
            else:
                expr_parts.append('pdf_name == "combined"')

            # Filter by summary_type using chunk_id pattern
            # chunk_id format: "summary_{pdf_name or 'all'}_{summary_type}"
            if summary_type:
                pdf_part = pdf_name if pdf_name else "all"
                expected_chunk_id = f"summary_{pdf_part}_{summary_type}"
                expr_parts.append(f'chunk_id == "{expected_chunk_id}"')

            expr = " and ".join(expr_parts)

            results = self.col.query(
                expr=expr,
                output_fields=["text", "pdf_name", "word_count", "chunk_id"],
                limit=10
            )

            if results:
                return results[0]

        except Exception:
            pass

        return None

    def get_all_summaries(self) -> List[dict]:
        """Get all stored summaries in the collection."""
        try:
            results = self.col.query(
                expr='content_type == "summary"',
                output_fields=["text", "pdf_name", "word_count", "chunk_id"],
                limit=100
            )
            return results
        except Exception:
            return []
