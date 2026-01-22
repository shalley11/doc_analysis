# chunking_utils.py
import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic hashing.
    - collapse whitespace
    - lowercase
    """
    return " ".join(text.split()).lower()


def deterministic_chunk_id(pdf_name: str, page_no: int, text: str, content_type: str = "text") -> str:
    """
    Generate deterministic chunk ID.
    Same content => same ID (idempotent).
    """
    normalized = normalize_text(text)
    raw = f"{pdf_name}:{page_no}:{content_type}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitter.
    Replace with spaCy/NLTK if needed.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def calculate_sentence_similarities(embeddings: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarity between consecutive sentence embeddings.

    Returns list of similarities where similarities[i] is the similarity
    between sentence i and sentence i+1.
    """
    if len(embeddings) < 2:
        return []

    similarities = []
    embeddings_np = [np.array(emb) for emb in embeddings]

    for i in range(len(embeddings_np) - 1):
        sim = cosine_similarity(embeddings_np[i], embeddings_np[i + 1])
        similarities.append(sim)

    return similarities


def find_semantic_breakpoints(
    similarities: List[float],
    threshold: float = 0.5,
    percentile_threshold: Optional[float] = None
) -> List[int]:
    """
    Find breakpoints where semantic similarity drops below threshold.

    Args:
        similarities: List of similarities between consecutive sentences
        threshold: Absolute similarity threshold (default 0.5)
        percentile_threshold: If set, use this percentile of similarities as threshold
                            (e.g., 25 means break at bottom 25% of similarities)

    Returns:
        List of indices where breaks should occur (break AFTER sentence at index)
    """
    if not similarities:
        return []

    # Use percentile-based threshold if specified
    if percentile_threshold is not None:
        threshold = float(np.percentile(similarities, percentile_threshold))

    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i)

    return breakpoints


class SemanticChunker:
    """
    Embedding-based semantic chunker that detects topic boundaries.

    Uses sentence embeddings to identify where semantic shifts occur,
    creating chunks that preserve topical coherence.
    """

    def __init__(
        self,
        embedding_client: Optional[Any] = None,
        similarity_threshold: float = 0.5,
        percentile_threshold: Optional[float] = 25,
        min_chunk_size: int = 50,
        max_chunk_size: int = 500,
        combine_threshold: float = 0.7,
        buffer_size: int = 1
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedding_client: EmbeddingClient instance for generating embeddings
            similarity_threshold: Minimum similarity to keep sentences together (0-1)
            percentile_threshold: If set, use bottom N percentile as breakpoints
            min_chunk_size: Minimum words per chunk (will merge small chunks)
            max_chunk_size: Maximum words per chunk (will split large chunks)
            combine_threshold: Similarity threshold for combining small chunks
            buffer_size: Number of sentences to consider for context (sliding window)
        """
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.combine_threshold = combine_threshold
        self.buffer_size = buffer_size

    def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for sentences, with buffering for context."""
        if not self.embedding_client or not sentences:
            return None

        try:
            # Create buffered sentences for better context
            if self.buffer_size > 1 and len(sentences) > 1:
                buffered = []
                for i in range(len(sentences)):
                    start = max(0, i - self.buffer_size // 2)
                    end = min(len(sentences), i + self.buffer_size // 2 + 1)
                    buffered.append(" ".join(sentences[start:end]))
                return self.embedding_client.embed(buffered)
            else:
                return self.embedding_client.embed(sentences)
        except Exception as e:
            logger.warning(f"Failed to get embeddings for semantic chunking: {e}")
            return None

    def _merge_small_chunks(
        self,
        chunks: List[List[str]],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[List[str]]:
        """Merge chunks that are too small with their most similar neighbor."""
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_words = sum(len(s.split()) for s in current_chunk)

            # If chunk is too small and not the last one, try to merge
            if current_words < self.min_chunk_size and i < len(chunks) - 1:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_chunk = current_chunk + next_chunk
                merged.append(merged_chunk)
                i += 2
            elif current_words < self.min_chunk_size and merged:
                # Merge with previous chunk if this is the last small chunk
                merged[-1] = merged[-1] + current_chunk
                i += 1
            else:
                merged.append(current_chunk)
                i += 1

        return merged

    def _split_large_chunks(self, chunks: List[List[str]]) -> List[List[str]]:
        """Split chunks that exceed max_chunk_size."""
        result = []

        for chunk in chunks:
            total_words = sum(len(s.split()) for s in chunk)

            if total_words <= self.max_chunk_size:
                result.append(chunk)
                continue

            # Split the chunk
            current_split = []
            current_words = 0

            for sentence in chunk:
                sentence_words = len(sentence.split())

                if current_words + sentence_words > self.max_chunk_size and current_split:
                    result.append(current_split)
                    current_split = []
                    current_words = 0

                current_split.append(sentence)
                current_words += sentence_words

            if current_split:
                result.append(current_split)

        return result

    def chunk_text(
        self,
        text: str,
        pdf_name: str,
        page_no: int,
        start_chunk_number: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic chunking on text.

        Args:
            text: The text to chunk
            pdf_name: Name of the source PDF
            page_no: Page number in the PDF
            start_chunk_number: Starting chunk number

        Returns:
            List of chunk dictionaries with metadata
        """
        sentences = sentence_split(text)

        if not sentences:
            return []

        # If only one sentence or embedding client unavailable, use simple chunking
        if len(sentences) <= 1 or not self.embedding_client:
            return self._fallback_chunking(
                text, pdf_name, page_no, start_chunk_number
            )

        # Get embeddings for sentences
        embeddings = self._get_sentence_embeddings(sentences)

        if embeddings is None:
            logger.info("Falling back to simple chunking (no embeddings)")
            return self._fallback_chunking(
                text, pdf_name, page_no, start_chunk_number
            )

        # Calculate similarities between consecutive sentences
        similarities = calculate_sentence_similarities(embeddings)

        # Find breakpoints
        breakpoints = find_semantic_breakpoints(
            similarities,
            threshold=self.similarity_threshold,
            percentile_threshold=self.percentile_threshold
        )

        # Group sentences into chunks based on breakpoints
        chunks_sentences = self._group_by_breakpoints(sentences, breakpoints)

        # Apply size constraints
        chunks_sentences = self._merge_small_chunks(chunks_sentences, embeddings)
        chunks_sentences = self._split_large_chunks(chunks_sentences)

        # Convert to chunk dictionaries
        return self._sentences_to_chunks(
            chunks_sentences, pdf_name, page_no, start_chunk_number
        )

    def _group_by_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[List[str]]:
        """Group sentences into chunks based on breakpoint indices."""
        if not breakpoints:
            return [sentences]

        chunks = []
        prev_idx = 0

        for bp in sorted(breakpoints):
            # Break AFTER the sentence at index bp
            chunk = sentences[prev_idx:bp + 1]
            if chunk:
                chunks.append(chunk)
            prev_idx = bp + 1

        # Add remaining sentences
        if prev_idx < len(sentences):
            chunks.append(sentences[prev_idx:])

        return chunks

    def _sentences_to_chunks(
        self,
        chunks_sentences: List[List[str]],
        pdf_name: str,
        page_no: int,
        start_chunk_number: int
    ) -> List[Dict[str, Any]]:
        """Convert grouped sentences to chunk dictionaries."""
        chunks = []
        chunk_number = start_chunk_number

        for sentence_group in chunks_sentences:
            chunk_text = " ".join(sentence_group)

            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text, "text"
                ),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": 0,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {
                    "chunking_method": "semantic",
                    "sentence_count": len(sentence_group)
                }
            })
            chunk_number += 1

        return chunks

    def _fallback_chunking(
        self,
        text: str,
        pdf_name: str,
        page_no: int,
        start_chunk_number: int
    ) -> List[Dict[str, Any]]:
        """Fallback to simple word-count based chunking."""
        sentences = sentence_split(text)
        chunks = []

        current_chunk = []
        current_word_count = 0
        chunk_number = start_chunk_number

        for sentence in sentences:
            words = sentence.split()

            if current_word_count + len(words) > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": deterministic_chunk_id(
                        pdf_name, page_no, chunk_text, "text"
                    ),
                    "content_type": "text",
                    "text": chunk_text,
                    "pdf_name": pdf_name,
                    "page_no": page_no,
                    "position": 0,
                    "chunk_number": chunk_number,
                    "image_link": "",
                    "table_link": "",
                    "context_before_id": "",
                    "context_after_id": "",
                    "metadata": {"chunking_method": "fallback"}
                })
                chunk_number += 1
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += len(words)

        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text, "text"
                ),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": 0,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {"chunking_method": "fallback"}
            })

        return chunks


def semantic_chunk_text(
    text: str,
    pdf_name: str,
    page_no: int,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict]:
    """
    Create semantic chunks with deterministic IDs and metadata.

    Args:
        text: The text to chunk
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        max_words: Maximum words per chunk (default 250)
        start_chunk_number: Starting chunk number for this page (default 0)

    Returns:
        List of chunk dictionaries with chunk_id, pdf_name, page_no, chunk_number, and text
    """
    sentences = sentence_split(text)
    chunks = []

    current_chunk = []
    current_word_count = 0
    chunk_number = start_chunk_number

    for sentence in sentences:
        words = sentence.split()

        if current_word_count + len(words) > max_words:
            chunk_text = " ".join(current_chunk)

            chunks.append({
                "chunk_id": deterministic_chunk_id(
                    pdf_name, page_no, chunk_text
                ),
                "pdf_name": pdf_name,
                "page_no": page_no,
                "chunk_number": chunk_number,
                "text": chunk_text
            })

            chunk_number += 1
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    # last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)

        chunks.append({
            "chunk_id": deterministic_chunk_id(
                pdf_name, page_no, chunk_text
            ),
            "pdf_name": pdf_name,
            "page_no": page_no,
            "chunk_number": chunk_number,
            "text": chunk_text
        })

    return chunks


def create_multimodal_chunks(
    blocks: List[Dict[str, Any]],
    pdf_name: str,
    page_no: int,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Create chunks from multimodal content blocks (text, tables, images).

    Each block becomes one or more chunks:
    - Text blocks: Split by sentence/word count
    - Table blocks: Single chunk with vision-generated summary
    - Image blocks: Single chunk with vision-generated description

    Args:
        blocks: List of content blocks with type, content, position, links
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        max_words: Maximum words per text chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    chunks = []
    chunk_number = start_chunk_number

    for block in blocks:
        block_type = block.get("type", "text")
        content = block.get("content", "")
        position = block.get("position", 0)
        image_link = block.get("image_link")
        table_link = block.get("table_link")

        if block_type == "text" and content:
            # Split text into smaller chunks if needed
            text_chunks = _chunk_text_content(
                content, pdf_name, page_no, position, max_words, chunk_number
            )
            chunks.extend(text_chunks)
            chunk_number += len(text_chunks)

        elif block_type == "table":
            # Table as single chunk with vision summary
            chunk_text = content if content else "Table content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": table_link or "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

        elif block_type == "image":
            # Image as single chunk with vision description
            chunk_text = content if content else "Image content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "image"),
                "content_type": "image",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": image_link or "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

    # Link context (previous and next chunks)
    chunks = _link_chunk_context(chunks)

    return chunks


def _chunk_text_content(
    text: str,
    pdf_name: str,
    page_no: int,
    position: int,
    max_words: int,
    start_chunk_number: int
) -> List[Dict[str, Any]]:
    """Split text content into chunks respecting word limits."""
    sentences = sentence_split(text)
    chunks = []

    current_chunk = []
    current_word_count = 0
    chunk_number = start_chunk_number
    sub_position = 0

    for sentence in sentences:
        words = sentence.split()

        if current_word_count + len(words) > max_words and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "text"),
                "content_type": "text",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position + (sub_position * 0.01),  # Sub-position for text splits
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": {}
            })

            chunk_number += 1
            sub_position += 1
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    # Last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "text"),
            "content_type": "text",
            "text": chunk_text,
            "pdf_name": pdf_name,
            "page_no": page_no,
            "position": position + (sub_position * 0.01),
            "chunk_number": chunk_number,
            "image_link": "",
            "table_link": "",
            "context_before_id": "",
            "context_after_id": "",
            "metadata": {}
        })

    return chunks


def _link_chunk_context(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Link each chunk to its previous and next chunk for context."""
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk["context_before_id"] = chunks[i - 1]["chunk_id"]
        if i < len(chunks) - 1:
            chunk["context_after_id"] = chunks[i + 1]["chunk_id"]

    return chunks


def process_page_to_chunks(
    page_result: Dict[str, Any],
    vision_processor: Optional[Any] = None,
    max_words: int = 250,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a page result (from pdf_utils.process_page_with_positions) into chunks.

    Args:
        page_result: Result from process_page_with_positions()
        vision_processor: Optional VisionProcessor for image/table descriptions
        max_words: Maximum words per text chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunks ready for embedding and indexing
    """
    pdf_name = page_result["metadata"]["pdf_name"]
    page_no = page_result["metadata"]["page_no"]
    blocks = page_result.get("blocks", [])

    # Process blocks with vision model if available
    if vision_processor and blocks:
        from vision_utils import VisionProcessor
        if isinstance(vision_processor, VisionProcessor):
            blocks = vision_processor.process_blocks(blocks)

    # Create chunks from blocks
    chunks = create_multimodal_chunks(
        blocks=blocks,
        pdf_name=pdf_name,
        page_no=page_no,
        max_words=max_words,
        start_chunk_number=start_chunk_number
    )

    return chunks


def create_semantic_multimodal_chunks(
    blocks: List[Dict[str, Any]],
    pdf_name: str,
    page_no: int,
    embedding_client: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    percentile_threshold: Optional[float] = 25,
    min_chunk_size: int = 50,
    max_chunk_size: int = 500,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Create semantically-aware chunks from multimodal content blocks.

    Uses embedding-based semantic chunking for text blocks to detect
    topic boundaries, while handling tables and images as single chunks.

    Args:
        blocks: List of content blocks with type, content, position, links
        pdf_name: Name of the source PDF
        page_no: Page number in the PDF
        embedding_client: EmbeddingClient for semantic chunking (optional)
        similarity_threshold: Minimum similarity to keep sentences together (0-1)
        percentile_threshold: Use bottom N percentile as breakpoints
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunk dictionaries ready for embedding and indexing
    """
    # Initialize semantic chunker
    semantic_chunker = SemanticChunker(
        embedding_client=embedding_client,
        similarity_threshold=similarity_threshold,
        percentile_threshold=percentile_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )

    chunks = []
    chunk_number = start_chunk_number

    for block in blocks:
        block_type = block.get("type", "text")
        content = block.get("content", "")
        position = block.get("position", 0)
        image_link = block.get("image_link")
        table_link = block.get("table_link")

        if block_type == "text" and content:
            # Use semantic chunking for text blocks
            text_chunks = semantic_chunker.chunk_text(
                text=content,
                pdf_name=pdf_name,
                page_no=page_no,
                start_chunk_number=chunk_number
            )

            # Update positions for each chunk
            for i, chunk in enumerate(text_chunks):
                chunk["position"] = position + (i * 0.01)

            chunks.extend(text_chunks)
            chunk_number += len(text_chunks)

        elif block_type == "table":
            # Table as single chunk with vision summary
            chunk_text = content if content else "Table content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "table"),
                "content_type": "table",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": "",
                "table_link": table_link or "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

        elif block_type == "image":
            # Image as single chunk with vision description
            chunk_text = content if content else "Image content"
            chunks.append({
                "chunk_id": deterministic_chunk_id(pdf_name, page_no, chunk_text, "image"),
                "content_type": "image",
                "text": chunk_text,
                "pdf_name": pdf_name,
                "page_no": page_no,
                "position": position,
                "chunk_number": chunk_number,
                "image_link": image_link or "",
                "table_link": "",
                "context_before_id": "",
                "context_after_id": "",
                "metadata": block.get("metadata", {})
            })
            chunk_number += 1

    # Link context (previous and next chunks)
    chunks = _link_chunk_context(chunks)

    return chunks


def process_page_to_semantic_chunks(
    page_result: Dict[str, Any],
    embedding_client: Optional[Any] = None,
    vision_processor: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    percentile_threshold: Optional[float] = 25,
    min_chunk_size: int = 50,
    max_chunk_size: int = 500,
    start_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a page result into semantically-aware chunks.

    This is the main entry point for semantic chunking of PDF pages.

    Args:
        page_result: Result from process_page_with_positions()
        embedding_client: EmbeddingClient for semantic chunking
        vision_processor: Optional VisionProcessor for image/table descriptions
        similarity_threshold: Minimum similarity to keep sentences together
        percentile_threshold: Use bottom N percentile as breakpoints
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
        start_chunk_number: Starting chunk number

    Returns:
        List of chunks ready for embedding and indexing
    """
    pdf_name = page_result["metadata"]["pdf_name"]
    page_no = page_result["metadata"]["page_no"]
    blocks = page_result.get("blocks", [])

    # Process blocks with vision model if available
    if vision_processor and blocks:
        from vision_utils import VisionProcessor
        if isinstance(vision_processor, VisionProcessor):
            blocks = vision_processor.process_blocks(blocks)

    # Create semantic chunks from blocks
    chunks = create_semantic_multimodal_chunks(
        blocks=blocks,
        pdf_name=pdf_name,
        page_no=page_no,
        embedding_client=embedding_client,
        similarity_threshold=similarity_threshold,
        percentile_threshold=percentile_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        start_chunk_number=start_chunk_number
    )

    return chunks
