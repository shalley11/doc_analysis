"""Embedding module for text embeddings and similarity."""

from .embedding_client import EmbeddingClient
from .embedding_utils import E5LargeEmbedder

__all__ = [
    "EmbeddingClient",
    "E5LargeEmbedder"
]
