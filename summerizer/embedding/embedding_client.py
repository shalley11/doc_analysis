import requests
from typing import List


class EmbeddingClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            r = requests.post(
                f"{self.endpoint}/embed",
                json={"texts": texts},
                timeout=60
            )
            r.raise_for_status()
            result = r.json()

            if "embeddings" not in result:
                raise ValueError(f"Invalid response from embedding service: missing 'embeddings' key")

            embeddings = result["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )

            return embeddings

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to embedding service at {self.endpoint}. "
                f"Ensure the service is running. Error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Embedding service at {self.endpoint} timed out after 60s. "
                f"Consider reducing batch size. Error: {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Embedding service returned error: {e.response.status_code} - {e.response.text}"
            ) from e
