from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from typing import List, Dict


class MilvusVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        alias: str = "default"
    ):
        """
        Correct connection method for pymilvus >= 2.6.x
        (Standalone / Docker Milvus via gRPC)
        """
        connections.connect(
            alias=alias,
            host=host,
            port=port
        )
        self.alias = alias

    def _collection_name(self, session_id: str) -> str:
        # Milvus collection names can only contain letters, numbers, and underscores
        safe_id = session_id.replace("-", "_")
        return f"tmp_chunks_{safe_id}"

    def create_session_collection(
        self,
        session_id: str,
        dim: int,
        ttl_seconds: int
    ):
        name = self._collection_name(session_id)

        if utility.has_collection(name, using=self.alias):
            return

        fields = [
            # Primary key
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True
            ),
            # Embedding vector
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim
            ),
            # Content type: text | table | image
            FieldSchema(
                name="content_type",
                dtype=DataType.VARCHAR,
                max_length=16
            ),
            # Text content or vision-generated summary
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            # Source document info
            FieldSchema(
                name="pdf_name",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="page_no",
                dtype=DataType.INT64
            ),
            # Position within page (for ordering)
            FieldSchema(
                name="position",
                dtype=DataType.INT64
            ),
            # Global chunk number
            FieldSchema(
                name="chunk_number",
                dtype=DataType.INT64
            ),
            # Links for explainability
            FieldSchema(
                name="image_link",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name="table_link",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            # Context linking
            FieldSchema(
                name="context_before_id",
                dtype=DataType.VARCHAR,
                max_length=64
            ),
            FieldSchema(
                name="context_after_id",
                dtype=DataType.VARCHAR,
                max_length=64
            ),
            # Section hierarchy (JSON array of section titles)
            FieldSchema(
                name="section_hierarchy",
                dtype=DataType.VARCHAR,
                max_length=2048
            ),
            # Heading level (0=body, 1=H1, 2=H2, 3=H3)
            FieldSchema(
                name="heading_level",
                dtype=DataType.INT64
            ),
            # Vision model metadata for tables
            FieldSchema(
                name="table_summary",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            # Vision model metadata for images
            FieldSchema(
                name="image_caption",
                dtype=DataType.VARCHAR,
                max_length=4096
            ),
            FieldSchema(
                name="image_summary",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Temporary semantic chunks"
        )

        collection = Collection(
            name=name,
            schema=schema,
            using=self.alias
        )

        collection.set_properties({
            "collection.ttl.seconds": ttl_seconds
        })

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200}
            }
        )

        collection.load()

    def insert_chunks(self, session_id: str, records: List[Dict]):
        collection = Collection(
            name=self._collection_name(session_id),
            using=self.alias
        )
        collection.insert(records)
        collection.flush()

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int
    ):
        collection = Collection(
            name=self._collection_name(session_id),
            using=self.alias
        )
        collection.load()

        res = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=[
                "text", "content_type", "pdf_name", "page_no",
                "position", "chunk_number", "image_link", "table_link",
                "context_before_id", "context_after_id",
                "section_hierarchy", "heading_level",
                "table_summary", "image_caption", "image_summary"
            ]
        )
        return res[0]

    def drop_session(self, session_id: str):
        name = self._collection_name(session_id)
        if utility.has_collection(name, using=self.alias):
            utility.drop_collection(name, using=self.alias)
