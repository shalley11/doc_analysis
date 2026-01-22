"""Chunking module for text segmentation and indexing."""

from .chunking_utils import (
    semantic_chunk_text,
    create_multimodal_chunks,
    create_semantic_multimodal_chunks,
    SemanticChunker
)
from .chunk_indexer import ChunkIndexer

__all__ = [
    "semantic_chunk_text",
    "create_multimodal_chunks",
    "create_semantic_multimodal_chunks",
    "SemanticChunker",
    "ChunkIndexer"
]
