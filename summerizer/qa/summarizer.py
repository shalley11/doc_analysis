"""
Advanced Summarization Strategies for different summary scopes.

Supports:
1. Topic/Section-wise - Cluster chunks by topic, summarize each
2. Document-wise - Summary per PDF
3. All Documents - Combined summary using Map-Reduce
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from qa.generator import LLMGenerator, get_default_generator


class SummaryScope(str, Enum):
    """Summary scope options."""
    TOPIC = "topic"          # Section/topic-wise summary
    DOCUMENT = "document"    # Per-PDF summary
    ALL = "all"              # Combined summary of all docs


class SummaryFormat(str, Enum):
    """Output format options."""
    BRIEF = "brief"
    DETAILED = "detailed"
    BULLETS = "bullets"


@dataclass
class SectionSummary:
    """A single section/topic summary."""
    title: str
    summary: str
    chunk_count: int
    sources: List[Dict[str, Any]]


@dataclass
class DocumentSummary:
    """Summary for a single document."""
    pdf_name: str
    summary: str
    page_count: int
    chunk_count: int


# =============================================================================
# Prompts
# =============================================================================

SECTION_TITLE_PROMPT = """Based on the following text chunks, generate a short, descriptive title (3-6 words) that captures the main topic.

Text:
{text}

Respond with ONLY the title, nothing else."""

SECTION_SUMMARY_PROMPT = """Summarize the following content about "{title}".

Content:
{context}

Instructions:
- Be concise but comprehensive
- Include key points and details
- Use citations: [Source: PDF_NAME, Page X]
- Format: {format_instruction}"""

MAP_SUMMARY_PROMPT = """Summarize the following document section concisely. Preserve key facts and details.

Source: {pdf_name}, Pages {page_range}

Content:
{content}

Provide a concise summary (2-3 sentences) capturing the main points."""

REDUCE_SUMMARY_PROMPT = """Combine the following summaries into a coherent {format_type} summary.

Summaries:
{summaries}

Instructions:
- Create a unified, well-structured summary
- Preserve important details from each section
- Include citations where relevant: [Source: PDF_NAME, Page X]
- {format_instruction}"""

DOCUMENT_SUMMARY_PROMPT = """Summarize the following document content.

Document: {pdf_name}
Total Pages: {page_count}

Content:
{content}

Instructions:
- Provide a comprehensive summary of this document
- Include main topics, key findings, and important details
- Use citations: [Source: {pdf_name}, Page X]
- Format: {format_instruction}"""

FORMAT_INSTRUCTIONS = {
    "brief": "Provide a concise summary in 2-3 paragraphs.",
    "detailed": "Provide a comprehensive summary covering all major topics.",
    "bullets": "Provide the summary as organized bullet points."
}


class Summarizer:
    """
    Advanced summarizer supporting multiple summary strategies.
    """

    def __init__(self, generator: Optional[LLMGenerator] = None):
        self.generator = generator or get_default_generator()

    def summarize_by_topic(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        num_topics: int = 5,
        summary_format: str = "brief",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate topic/section-wise summaries using clustering.

        Args:
            chunks: List of text chunks with metadata
            embeddings: Pre-computed embeddings for chunks
            num_topics: Number of topics/sections to identify
            summary_format: brief, detailed, or bullets
            temperature: LLM temperature

        Returns:
            Dict with sections list and metadata
        """
        if not chunks:
            return {"sections": [], "total_chunks": 0}

        # Adjust num_topics based on chunk count
        num_topics = min(num_topics, len(chunks))

        # Cluster chunks by topic
        clusters = self._cluster_chunks(chunks, embeddings, num_topics)

        sections = []
        for cluster_id, cluster_chunks in clusters.items():
            if not cluster_chunks:
                continue

            # Generate title for this topic
            sample_text = " ".join([c["text"][:200] for c in cluster_chunks[:3]])
            title = self._generate_title(sample_text, temperature)

            # Format context with citations
            context = self._format_context(cluster_chunks)

            # Generate summary for this section
            summary = self._generate_section_summary(
                title, context, summary_format, temperature
            )

            sections.append({
                "title": title,
                "summary": summary,
                "chunk_count": len(cluster_chunks),
                "sources": [
                    {"pdf_name": c["pdf_name"], "page_no": c.get("page_no", 0) + 1}
                    for c in cluster_chunks[:5]
                ]
            })

        return {
            "scope": "topic",
            "sections": sections,
            "total_topics": len(sections),
            "total_chunks": len(chunks),
            "format": summary_format
        }

    def summarize_by_document(
        self,
        chunks: List[Dict[str, Any]],
        summary_format: str = "brief",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate per-document summaries using Map-Reduce.

        Args:
            chunks: List of all chunks
            summary_format: brief, detailed, or bullets
            temperature: LLM temperature

        Returns:
            Dict with document summaries
        """
        if not chunks:
            return {"documents": [], "total_chunks": 0}

        # Group chunks by PDF
        pdf_chunks = defaultdict(list)
        for chunk in chunks:
            pdf_name = chunk.get("pdf_name", "Unknown")
            pdf_chunks[pdf_name].append(chunk)

        documents = []
        for pdf_name, doc_chunks in pdf_chunks.items():
            # Sort by page and position
            doc_chunks.sort(key=lambda x: (x.get("page_no", 0), x.get("position", 0)))

            # Get page range
            pages = set(c.get("page_no", 0) for c in doc_chunks)
            page_count = max(pages) - min(pages) + 1 if pages else 0

            # Use Map-Reduce for this document
            summary = self._map_reduce_summary(
                doc_chunks, pdf_name, summary_format, temperature
            )

            documents.append({
                "pdf_name": pdf_name,
                "summary": summary,
                "page_count": page_count,
                "chunk_count": len(doc_chunks)
            })

        return {
            "scope": "document",
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "format": summary_format
        }

    def summarize_all(
        self,
        chunks: List[Dict[str, Any]],
        summary_format: str = "brief",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a combined summary of all documents using hierarchical Map-Reduce.

        Args:
            chunks: List of all chunks
            summary_format: brief, detailed, or bullets
            temperature: LLM temperature

        Returns:
            Dict with combined summary
        """
        if not chunks:
            return {"summary": "No content found.", "total_chunks": 0}

        # First, get per-document summaries
        doc_result = self.summarize_by_document(chunks, "brief", temperature)

        if not doc_result["documents"]:
            return {"summary": "No content found.", "total_chunks": 0}

        # If only one document, just reformat its summary
        if len(doc_result["documents"]) == 1:
            doc = doc_result["documents"][0]
            if summary_format == "brief":
                final_summary = doc["summary"]
            else:
                # Re-summarize with requested format
                doc_chunks = [c for c in chunks if c.get("pdf_name") == doc["pdf_name"]]
                final_summary = self._map_reduce_summary(
                    doc_chunks, doc["pdf_name"], summary_format, temperature
                )
        else:
            # Combine document summaries
            summaries_text = "\n\n".join([
                f"**{doc['pdf_name']}**:\n{doc['summary']}"
                for doc in doc_result["documents"]
            ])

            prompt = REDUCE_SUMMARY_PROMPT.format(
                format_type=summary_format,
                summaries=summaries_text,
                format_instruction=FORMAT_INSTRUCTIONS.get(summary_format, FORMAT_INSTRUCTIONS["brief"])
            )

            final_summary = self.generator.generate(
                system_prompt="You are a helpful assistant that combines document summaries.",
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=2048
            )

        # Collect all sources
        pdf_names = list(set(c.get("pdf_name", "Unknown") for c in chunks))

        return {
            "scope": "all",
            "summary": final_summary,
            "documents_included": pdf_names,
            "total_documents": len(pdf_names),
            "total_chunks": len(chunks),
            "format": summary_format
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _cluster_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        num_clusters: int
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster chunks using K-means on embeddings."""
        try:
            from sklearn.cluster import KMeans

            if len(embeddings) < num_clusters:
                num_clusters = len(embeddings)

            embeddings_array = np.array(embeddings)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)

            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(chunks[i])

            return clusters

        except ImportError:
            # Fallback: simple sequential grouping
            clusters = defaultdict(list)
            chunk_size = max(1, len(chunks) // num_clusters)
            for i, chunk in enumerate(chunks):
                cluster_id = min(i // chunk_size, num_clusters - 1)
                clusters[cluster_id].append(chunk)
            return clusters

    def _generate_title(self, text: str, temperature: float) -> str:
        """Generate a title for a topic cluster."""
        prompt = SECTION_TITLE_PROMPT.format(text=text[:1000])
        title = self.generator.generate(
            system_prompt="You are a helpful assistant that generates concise titles.",
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=50
        )
        return title.strip().strip('"\'')

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks with source citations."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            pdf_name = chunk.get("pdf_name", "Unknown")
            page_no = chunk.get("page_no", 0) + 1
            text = chunk.get("text", "")
            parts.append(f"[Source {i}: {pdf_name}, Page {page_no}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def _generate_section_summary(
        self,
        title: str,
        context: str,
        summary_format: str,
        temperature: float
    ) -> str:
        """Generate summary for a single section."""
        prompt = SECTION_SUMMARY_PROMPT.format(
            title=title,
            context=context[:8000],  # Limit context size
            format_instruction=FORMAT_INSTRUCTIONS.get(summary_format, FORMAT_INSTRUCTIONS["brief"])
        )
        return self.generator.generate(
            system_prompt="You are a helpful assistant that summarizes document sections.",
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=1024
        )

    def _map_reduce_summary(
        self,
        chunks: List[Dict[str, Any]],
        pdf_name: str,
        summary_format: str,
        temperature: float
    ) -> str:
        """Generate summary using Map-Reduce strategy."""
        # MAP: Summarize chunks in batches
        batch_size = 5
        intermediate_summaries = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            pages = sorted(set(c.get("page_no", 0) + 1 for c in batch))
            page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(min(pages))

            content = "\n\n".join([c.get("text", "") for c in batch])

            prompt = MAP_SUMMARY_PROMPT.format(
                pdf_name=pdf_name,
                page_range=page_range,
                content=content[:4000]
            )

            summary = self.generator.generate(
                system_prompt="You are a helpful assistant that summarizes document sections.",
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=300
            )
            intermediate_summaries.append(f"[Pages {page_range}]: {summary}")

        # REDUCE: Combine intermediate summaries
        if len(intermediate_summaries) == 1:
            # Single batch, just reformat
            combined = intermediate_summaries[0]
        else:
            combined = "\n\n".join(intermediate_summaries)

        # Final reduction
        prompt = REDUCE_SUMMARY_PROMPT.format(
            format_type=summary_format,
            summaries=combined,
            format_instruction=FORMAT_INSTRUCTIONS.get(summary_format, FORMAT_INSTRUCTIONS["brief"])
        )

        return self.generator.generate(
            system_prompt="You are a helpful assistant that creates coherent document summaries.",
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=2048
        )
