"""Q&A Service - Orchestrates retrieval and generation for document Q&A."""

from typing import List, Dict, Any, Optional
from qa.retriever import Retriever
from qa.generator import LLMGenerator, get_default_generator, create_generator
from qa.prompts import build_qa_prompt, build_summary_prompt
from qa.summarizer import Summarizer, SummaryScope


class QAService:
    """
    Main service for document Q&A with citations.

    Flow:
    1. User asks a question
    2. Query is embedded and similar chunks retrieved from Milvus
    3. Chunks are formatted with source info
    4. LLM generates answer with citations
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Optional[LLMGenerator] = None
    ):
        self.retriever = retriever
        self.generator = generator or get_default_generator()
        self.summarizer = Summarizer(self.generator)

    def ask(
        self,
        session_id: str,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question based on the ingested documents.

        Args:
            session_id: Batch/session ID of the ingested PDFs
            question: User's question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature (0-1)
            include_sources: Whether to include source chunks in response

        Returns:
            Dict with answer and optional sources
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.search(session_id, question, top_k)

        if not chunks:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "chunks_retrieved": 0
            }

        # Step 2: Build prompt with citations
        system_prompt, user_prompt = build_qa_prompt(question, chunks)

        # Step 3: Generate answer
        answer = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )

        # Step 4: Build response
        response = {
            "answer": answer,
            "chunks_retrieved": len(chunks)
        }

        if include_sources:
            response["sources"] = [
                {
                    "pdf_name": c["pdf_name"],
                    "page_no": c["page_no"] + 1,  # 1-indexed for display
                    "content_type": c["content_type"],
                    "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                    "score": round(c["score"], 4),
                    "image_link": c.get("image_link", ""),
                    "table_link": c.get("table_link", "")
                }
                for c in chunks
            ]

        return response

    def summarize(
        self,
        session_id: str,
        summary_type: str = "brief",
        max_chunks: int = 20,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a summary of the ingested documents.

        Args:
            session_id: Batch/session ID
            summary_type: "brief", "detailed", or "bullets"
            max_chunks: Maximum chunks to include in context
            temperature: LLM temperature

        Returns:
            Dict with summary and metadata
        """
        # Retrieve chunks for summarization
        chunks = self.retriever.get_all_chunks(session_id, limit=max_chunks)

        if not chunks:
            return {
                "summary": "No content found in the documents.",
                "chunks_used": 0
            }

        # Build summary prompt
        system_prompt, user_prompt = build_summary_prompt(chunks, summary_type)

        # Generate summary
        summary = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=2048
        )

        return {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(chunks)
        }

    def advanced_summarize(
        self,
        session_id: str,
        scope: str = "all",
        summary_format: str = "brief",
        num_topics: int = 5,
        max_chunks: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate summaries with different scopes.

        Args:
            session_id: Batch/session ID
            scope: "topic" (section-wise), "document" (per-PDF), "all" (combined)
            summary_format: "brief", "detailed", or "bullets"
            num_topics: Number of topics for topic-wise summary
            max_chunks: Maximum chunks to process
            temperature: LLM temperature

        Returns:
            Dict with summaries based on scope
        """
        # Get all chunks
        chunks = self.retriever.get_all_chunks(session_id, limit=max_chunks)

        if not chunks:
            return {
                "scope": scope,
                "error": "No content found in the documents.",
                "chunks_used": 0
            }

        if scope == SummaryScope.TOPIC or scope == "topic":
            # Need embeddings for clustering
            embeddings = self._get_chunk_embeddings(chunks)
            result = self.summarizer.summarize_by_topic(
                chunks=chunks,
                embeddings=embeddings,
                num_topics=num_topics,
                summary_format=summary_format,
                temperature=temperature
            )
        elif scope == SummaryScope.DOCUMENT or scope == "document":
            result = self.summarizer.summarize_by_document(
                chunks=chunks,
                summary_format=summary_format,
                temperature=temperature
            )
        else:  # "all"
            result = self.summarizer.summarize_all(
                chunks=chunks,
                summary_format=summary_format,
                temperature=temperature
            )

        return result

    def _get_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Get embeddings for chunks (for clustering)."""
        texts = [c.get("text", "") for c in chunks]
        return self.retriever.embedder.embed(texts)

    def chat(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Multi-turn conversation with document context.

        Args:
            session_id: Batch/session ID
            messages: List of {"role": "user"|"assistant", "content": "..."}
            top_k: Chunks to retrieve per turn
            temperature: LLM temperature

        Returns:
            Dict with response and sources
        """
        if not messages:
            return {"error": "No messages provided"}

        # Get the latest user message
        latest_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_message = msg.get("content", "")
                break

        if not latest_message:
            return {"error": "No user message found"}

        # Use ask() for the response
        return self.ask(
            session_id=session_id,
            question=latest_message,
            top_k=top_k,
            temperature=temperature,
            include_sources=True
        )


def create_qa_service(
    vector_store,
    embedding_client,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None
) -> QAService:
    """
    Factory function to create a QA service.

    Args:
        vector_store: MilvusVectorStore instance
        embedding_client: EmbeddingClient instance
        llm_provider: LLM provider ("ollama", "openai", "anthropic", "gemini")
        llm_model: Optional model name

    Returns:
        QAService instance
    """
    retriever = Retriever(vector_store, embedding_client)
    generator = create_generator(provider=llm_provider, model=llm_model)
    return QAService(retriever, generator)
