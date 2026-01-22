from typing import List, Dict, Any
from embedding.embedding_client import EmbeddingClient
from vector_store.milvus_store import MilvusVectorStore


class ChunkIndexer:
    def __init__(self, embedder: EmbeddingClient, vector_store: MilvusVectorStore, embedding_dim: int):
        self.embedder = embedder
        self.vector_store = vector_store
        self.embedding_dim = embedding_dim

    def index_chunks(self, session_id: str, chunks: List[Dict], ttl_seconds: int):
        """
        Index chunks into Milvus vector store.
        Supports both legacy chunks and new multimodal chunks.
        """
        if not chunks:
            print("ChunkIndexer: No chunks to index, skipping.")
            return

        print(f"ChunkIndexer: Creating session collection for {session_id}...")
        self.vector_store.create_session_collection(
            session_id=session_id,
            dim=self.embedding_dim,
            ttl_seconds=ttl_seconds
        )

        texts = [c["text"] for c in chunks]
        print(f"ChunkIndexer: Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedder.embed(texts)

        if not embeddings:
            raise ValueError("No embeddings returned from embedding service")

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) doesn't match chunk count ({len(chunks)})"
            )

        # Validate embedding dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != self.embedding_dim:
                raise ValueError(
                    f"Embedding {i} has wrong dimension: expected {self.embedding_dim}, got {len(emb)}"
                )

        records = []
        for chunk, emb in zip(chunks, embeddings):
            record = self._create_record(chunk, emb)
            records.append(record)

        print(f"ChunkIndexer: Inserting {len(records)} records into Milvus...")
        self.vector_store.insert_chunks(session_id, records)
        print(f"ChunkIndexer: Successfully indexed {len(records)} chunks.")

        # Log content type breakdown
        type_counts = {}
        for chunk in chunks:
            ct = chunk.get("content_type", "text")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        print(f"ChunkIndexer: Content breakdown - {type_counts}")

    def _create_record(self, chunk: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """
        Create a Milvus record from a chunk.
        Handles both legacy and new multimodal chunk formats.
        """
        # Required fields
        record = {
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding,
            "text": chunk["text"],
            "pdf_name": chunk["pdf_name"],
            "page_no": chunk["page_no"],
            "chunk_number": chunk.get("chunk_number", 0),
        }

        # New multimodal fields (with defaults for legacy support)
        record["content_type"] = chunk.get("content_type", "text")
        record["position"] = int(chunk.get("position", 0))
        record["image_link"] = chunk.get("image_link", "") or ""
        record["table_link"] = chunk.get("table_link", "") or ""
        record["context_before_id"] = chunk.get("context_before_id", "") or ""
        record["context_after_id"] = chunk.get("context_after_id", "") or ""

        return record

    def index_multimodal_chunks(
        self,
        session_id: str,
        chunks: List[Dict[str, Any]],
        ttl_seconds: int
    ):
        """
        Index multimodal chunks (text, tables, images) into Milvus.
        This is the preferred method for new multimodal processing.
        """
        if not chunks:
            print("ChunkIndexer: No chunks to index, skipping.")
            return

        # Filter out chunks without content
        valid_chunks = [c for c in chunks if c.get("text")]
        if len(valid_chunks) != len(chunks):
            print(f"ChunkIndexer: Filtered out {len(chunks) - len(valid_chunks)} chunks without content")

        if not valid_chunks:
            print("ChunkIndexer: No valid chunks to index after filtering.")
            return

        # Use the standard index_chunks method
        self.index_chunks(session_id, valid_chunks, ttl_seconds)
