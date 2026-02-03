"""
Summary generation service using vision models (Gemma3).
Generates document-wise and corpus-wise summaries on demand.
"""

import os
from typing import List, Dict, Any, Optional
from pdf.vision_utils import VisionProcessor, Gemma3VisionModel

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gemma3:4b")
SUMMARY_TIMEOUT = int(os.environ.get("SUMMARY_TIMEOUT", "1800"))  # 30 minutes default


# =============================================================================
# Summary Prompts
# =============================================================================

DOCUMENT_SUMMARY_PROMPT = """You are analyzing a PDF document. Based on all the extracted content below, provide a comprehensive summary.

DOCUMENT: {pdf_name}
TOTAL CHUNKS: {chunk_count}
CONTENT TYPES: {content_types}

DOCUMENT CONTENT:
{document_chunks}

---

Generate a summary with the following structure:

## DOCUMENT OVERVIEW
- **Document Type:** (report, manual, research paper, presentation, policy, etc.)
- **Main Topic:** (1-2 sentences describing the primary subject)
- **Target Audience:** (who is this document intended for?)

## KEY SECTIONS
List the main sections/chapters identified with brief descriptions.

## MAIN FINDINGS & CONTENT
- List 5-7 key points, findings, or important information
- Include specific data, numbers, statistics, or percentages mentioned
- Note any conclusions, recommendations, or action items

## TABLES & FIGURES SUMMARY
{table_figure_summary}

## CRITICAL TAKEAWAYS
List 3-5 most important things a reader should know from this document.

Keep the summary factual and based only on the provided content. Be specific with numbers and data.
"""

CORPUS_SUMMARY_PROMPT = """You are analyzing a collection of PDF documents. Based on the individual document summaries below, provide a comprehensive overview of the entire corpus.

CORPUS INFORMATION:
- Total Documents: {doc_count}
- Document Names: {doc_names}

INDIVIDUAL DOCUMENT SUMMARIES:
{document_summaries}

---

Generate a corpus-wide summary with the following structure:

## CORPUS OVERVIEW
- **Collection Theme:** What is the overall domain or theme of these documents?
- **Document Types:** List the types of documents in this collection
- **Coverage:** What topics or areas does this collection cover?

## COMMON TOPICS & THEMES
List topics that appear across multiple documents:
- Topic 1: (appears in X documents) - brief description
- Topic 2: (appears in X documents) - brief description
(continue for major themes)

## KEY PATTERNS & TRENDS
- Identify 5-7 recurring themes or patterns across documents
- Note any trends in data or findings
- Highlight points of agreement across documents

## UNIQUE CONTRIBUTIONS
What unique information does each document contribute to the corpus?

## CONTRADICTIONS OR DIFFERENCES
Note any conflicting information or differing viewpoints between documents.

## AGGREGATE INSIGHTS
- Combined key statistics and data points from all documents
- Overall conclusions that can be drawn from the collection
- Identified gaps (what topics are missing or underrepresented?)

## RECOMMENDED READING
Suggest which documents to prioritize based on their content and importance.

Base your analysis only on the provided document summaries. Be specific and factual.
"""

BRIEF_SUMMARY_PROMPT = """Provide a brief 3-5 sentence summary of the following document content:

DOCUMENT: {pdf_name}

CONTENT:
{document_chunks}

Summary:"""

BULLET_SUMMARY_PROMPT = """Extract the key points from the following document as a bullet list (10-15 points):

DOCUMENT: {pdf_name}

CONTENT:
{document_chunks}

Key Points:"""


class SummaryService:
    """Service for generating document and corpus summaries using Gemma3."""

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        model: str = SUMMARY_MODEL,
        timeout: int = SUMMARY_TIMEOUT
    ):
        """Initialize the summary service with Gemma3 model."""
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.timeout = timeout
        self._llm = None

    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                import requests
                self._llm = {
                    "base_url": self.ollama_base_url,
                    "model": self.model,
                    "timeout": self.timeout
                }
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}")
                self._llm = None
        return self._llm

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation."""
        import requests

        llm = self._get_llm()
        if not llm:
            return "Error: LLM not available"

        try:
            response = requests.post(
                f"{llm['base_url']}/api/generate",
                json={
                    "model": llm["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4096
                    }
                },
                timeout=llm["timeout"]
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The document may be too large."
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_document_summary(
        self,
        chunks: List[Dict[str, Any]],
        pdf_name: str,
        summary_type: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Generate a summary for a single document.

        Args:
            chunks: List of chunks from the document
            pdf_name: Name of the PDF document
            summary_type: "brief", "detailed", or "bullets"

        Returns:
            Dict with summary and metadata
        """
        if not chunks:
            return {
                "pdf_name": pdf_name,
                "summary": "No content available for summarization.",
                "chunk_count": 0,
                "error": "No chunks found"
            }

        # Prepare document content
        content_types = {}
        table_summaries = []
        image_summaries = []
        text_content = []

        for chunk in chunks:
            ct = chunk.get("content_type", "text")
            content_types[ct] = content_types.get(ct, 0) + 1

            # Collect text
            text = chunk.get("text", "")
            if text:
                # Add section context if available
                section = chunk.get("section_hierarchy", [])
                if section:
                    section_str = " > ".join(section)
                    text_content.append(f"[{section_str}]\n{text}")
                else:
                    text_content.append(text)

            # Collect table summaries
            table_summary = chunk.get("table_summary", "")
            if table_summary:
                table_summaries.append(f"- {table_summary[:500]}")

            # Collect image summaries
            image_summary = chunk.get("image_summary", "")
            if image_summary:
                image_summaries.append(f"- {image_summary[:500]}")

        # Combine content (limit to prevent token overflow)
        document_chunks = "\n\n".join(text_content[:50])  # Limit chunks
        if len(document_chunks) > 30000:
            document_chunks = document_chunks[:30000] + "\n\n[Content truncated...]"

        # Prepare table/figure summary
        table_figure_summary = ""
        if table_summaries:
            table_figure_summary += f"**Tables ({len(table_summaries)}):**\n" + "\n".join(table_summaries[:10])
        if image_summaries:
            table_figure_summary += f"\n\n**Images ({len(image_summaries)}):**\n" + "\n".join(image_summaries[:10])
        if not table_figure_summary:
            table_figure_summary = "No tables or figures with summaries found."

        # Select prompt based on summary type
        if summary_type == "brief":
            prompt = BRIEF_SUMMARY_PROMPT.format(
                pdf_name=pdf_name,
                document_chunks=document_chunks
            )
        elif summary_type == "bullets":
            prompt = BULLET_SUMMARY_PROMPT.format(
                pdf_name=pdf_name,
                document_chunks=document_chunks
            )
        else:  # detailed
            prompt = DOCUMENT_SUMMARY_PROMPT.format(
                pdf_name=pdf_name,
                chunk_count=len(chunks),
                content_types=", ".join(f"{k}: {v}" for k, v in content_types.items()),
                document_chunks=document_chunks,
                table_figure_summary=table_figure_summary
            )

        # Generate summary
        print(f"Generating {summary_type} summary for {pdf_name} ({len(chunks)} chunks)...")
        summary = self._call_ollama(prompt)

        return {
            "pdf_name": pdf_name,
            "summary_type": summary_type,
            "summary": summary,
            "chunk_count": len(chunks),
            "content_types": content_types,
            "tables_count": len(table_summaries),
            "images_count": len(image_summaries)
        }

    def generate_corpus_summary(
        self,
        document_summaries: List[Dict[str, Any]],
        batch_id: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a summary across all documents in a corpus.

        Args:
            document_summaries: List of individual document summaries
            batch_id: Batch identifier

        Returns:
            Dict with corpus summary and metadata
        """
        if not document_summaries:
            return {
                "batch_id": batch_id,
                "summary": "No documents available for summarization.",
                "doc_count": 0,
                "error": "No document summaries found"
            }

        # Prepare document summaries text
        doc_names = [ds.get("pdf_name", "Unknown") for ds in document_summaries]

        summaries_text = ""
        for ds in document_summaries:
            summaries_text += f"\n### {ds.get('pdf_name', 'Unknown')}\n"
            summaries_text += f"Chunks: {ds.get('chunk_count', 0)}, "
            summaries_text += f"Types: {ds.get('content_types', {})}\n"
            summaries_text += f"Summary:\n{ds.get('summary', 'No summary available.')}\n"
            summaries_text += "-" * 50 + "\n"

        # Limit size
        if len(summaries_text) > 40000:
            summaries_text = summaries_text[:40000] + "\n\n[Content truncated...]"

        prompt = CORPUS_SUMMARY_PROMPT.format(
            doc_count=len(document_summaries),
            doc_names=", ".join(doc_names),
            document_summaries=summaries_text
        )

        # Generate corpus summary
        print(f"Generating corpus summary for {len(document_summaries)} documents...")
        summary = self._call_ollama(prompt)

        return {
            "batch_id": batch_id,
            "summary_type": "corpus",
            "summary": summary,
            "doc_count": len(document_summaries),
            "documents": doc_names,
            "total_chunks": sum(ds.get("chunk_count", 0) for ds in document_summaries)
        }


def get_summary_service() -> SummaryService:
    """Get a configured summary service instance."""
    return SummaryService()
