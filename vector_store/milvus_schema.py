"""
Simple Milvus schema for document chunks.
"""
from pymilvus import FieldSchema, CollectionSchema, DataType


def get_chunk_schema(dim: int):
    """
    Get Milvus collection schema for chunks.

    Args:
        dim: Embedding dimension

    Returns:
        CollectionSchema for Milvus
    """
    fields = [
        # Primary key
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64
        ),

        # Dense embedding
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim
        ),

        # Chunk text
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535
        ),

        # PDF metadata
        FieldSchema(
            name="pdf_id",
            dtype=DataType.VARCHAR,
            max_length=64
        ),
        FieldSchema(
            name="pdf_name",
            dtype=DataType.VARCHAR,
            max_length=256
        ),

        # Location + ordering
        FieldSchema(
            name="page_no",
            dtype=DataType.INT64
        ),
        FieldSchema(
            name="chunk_seq",
            dtype=DataType.INT64
        ),

        # Context linking (for better RAG context)
        FieldSchema(
            name="prev_chunk_id",
            dtype=DataType.VARCHAR,
            max_length=64
        ),
        FieldSchema(
            name="next_chunk_id",
            dtype=DataType.VARCHAR,
            max_length=64
        ),

        # Content type: text, table, image, summary
        FieldSchema(
            name="content_type",
            dtype=DataType.VARCHAR,
            max_length=32
        ),

        # Word count
        FieldSchema(
            name="word_count",
            dtype=DataType.INT64
        ),

        # Image path (for image/table chunks)
        FieldSchema(
            name="image_path",
            dtype=DataType.VARCHAR,
            max_length=512
        ),

        # Table metadata (for split tables)
        FieldSchema(
            name="table_id",
            dtype=DataType.INT64
        ),
        FieldSchema(
            name="table_part",
            dtype=DataType.INT64
        ),
        FieldSchema(
            name="table_total_parts",
            dtype=DataType.INT64
        ),

        # Summary field (for table/image vision summaries)
        # Used when content_type is 'table' or 'image'
        FieldSchema(
            name="summary",
            dtype=DataType.VARCHAR,
            max_length=8192
        ),
        # Summary embedding (requires Milvus 2.4+)
        FieldSchema(
            name="summary_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim
        ),
    ]

    return CollectionSchema(fields=fields)
