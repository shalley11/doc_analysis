"""
Milvus collection utilities.
"""
from typing import Optional

from pymilvus import connections, Collection, utility

from doc_analysis.vector_store.milvus_schema import get_chunk_schema


def create_temp_collection(
    collection_name: str,
    dim: int,
    ttl_seconds: Optional[int] = None,
):
    """
    Create a temporary Milvus collection for batch processing.

    Args:
        collection_name: Name of the collection
        dim: Embedding dimension
        ttl_seconds: Time-to-live in seconds (optional)

    Returns:
        Loaded Milvus Collection
    """
    connections.connect(alias="default", host="localhost", port="19530")

    # Drop existing collection if exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    schema = get_chunk_schema(dim)

    collection = Collection(
        name=collection_name,
        schema=schema
    )

    # Create index for main embedding
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200}
        }
    )

    # Create index for summary embedding (unified for table/image summaries)
    collection.create_index(
        field_name="summary_embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200}
        }
    )

    if ttl_seconds:
        collection.set_properties({
            "collection.ttl.seconds": ttl_seconds
        })

    collection.load()
    return collection
