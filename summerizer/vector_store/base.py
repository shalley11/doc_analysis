from abc import ABC, abstractmethod
from typing import List, Dict


class VectorStore(ABC):

    @abstractmethod
    def create_session_collection(self, session_id: str, dim: int, ttl_seconds: int):
        pass

    @abstractmethod
    def insert_chunks(self, session_id: str, records: List[Dict]):
        pass

    @abstractmethod
    def search(self, session_id: str, query_embedding: List[float], top_k: int):
        pass

    @abstractmethod
    def drop_session(self, session_id: str):
        pass
