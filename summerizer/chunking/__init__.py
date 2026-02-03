"""Chunking module for text segmentation and indexing."""

from .chunking_utils import (
    semantic_chunk_text,
    create_multimodal_chunks,
    create_semantic_multimodal_chunks,
    SemanticChunker,
    # Structure-based chunking
    process_document_with_structure,
    chunk_paragraph_with_section,
    chunk_table_with_section,
    chunk_list_with_section,
    chunk_figure_with_section,
    get_section_prefix
)
from .chunk_indexer import ChunkIndexer

__all__ = [
    "semantic_chunk_text",
    "create_multimodal_chunks",
    "create_semantic_multimodal_chunks",
    "SemanticChunker",
    "ChunkIndexer",
    # Structure-based chunking
    "process_document_with_structure",
    "chunk_paragraph_with_section",
    "chunk_table_with_section",
    "chunk_list_with_section",
    "chunk_figure_with_section",
    "get_section_prefix"
]
