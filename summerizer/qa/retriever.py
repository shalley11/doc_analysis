"""Retriever module for similarity search against Milvus."""

from typing import List, Dict, Any, Optional
from embedding.embedding_client import EmbeddingClient
from vector_store.milvus_store import MilvusVectorStore


class Retriever:
    """
    Retrieves relevant chunks from Milvus based on query similarity.
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_client: EmbeddingClient
    ):
        self.vector_store = vector_store
        self.embedder = embedding_client

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        content_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on query.

        Args:
            session_id: The batch/session ID to search within
            query: User's question or search query
            top_k: Number of results to return
            content_type_filter: Optional filter for content type (text, table, image)

        Returns:
            List of chunks with metadata and similarity scores
        """
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]

        # Search Milvus
        # Fetch more results if filtering, to ensure we have enough after filter
        fetch_k = top_k * 3 if content_type_filter else top_k
        results = self.vector_store.search(session_id, query_embedding, fetch_k)

        # Convert Milvus results to list of dicts
        chunks = []
        for hit in results:
            chunk = {
                "chunk_id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text", ""),
                "content_type": hit.entity.get("content_type", "text"),
                "pdf_name": hit.entity.get("pdf_name", ""),
                "page_no": hit.entity.get("page_no", 0),
                "position": hit.entity.get("position", 0),
                "chunk_number": hit.entity.get("chunk_number", 0),
                "image_link": hit.entity.get("image_link", ""),
                "table_link": hit.entity.get("table_link", ""),
                "context_before_id": hit.entity.get("context_before_id", ""),
                "context_after_id": hit.entity.get("context_after_id", "")
            }
            chunks.append(chunk)

        # Apply content type filter if specified
        if content_type_filter:
            chunks = [c for c in chunks if c["content_type"] == content_type_filter]

        # Return top_k results
        return chunks[:top_k]

    def search_with_context(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        include_neighbors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with optional context expansion (include neighboring chunks).

        Args:
            session_id: The batch/session ID
            query: User's question
            top_k: Number of primary results
            include_neighbors: Whether to include before/after chunks

        Returns:
            List of chunks, potentially expanded with neighbors
        """
        primary_chunks = self.search(session_id, query, top_k)

        if not include_neighbors:
            return primary_chunks

        # For now, return primary chunks
        # Context expansion can be added later by fetching context_before_id/context_after_id
        return primary_chunks

    def get_all_chunks(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a session (for summarization).

        Args:
            session_id: The batch/session ID
            limit: Maximum chunks to return

        Returns:
            List of all chunks up to limit
        """
        # Use a generic query to get all chunks
        # We'll use a zero vector search with high limit
        dummy_query = "document content summary overview"
        return self.search(session_id, dummy_query, top_k=limit)
