# embedding_utils.py
import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class E5LargeEmbedder:
    def __init__(self, cache_dir: str = "./models/e5-large", quantize: bool = True):
        """
        cache_dir: local folder to store model/tokenizer
        quantize: load INT8 weights if available for GPU
        """
        # Automatically detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.quantize = quantize
        self._is_quantized = False

        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            # Load from local cache if exists
            self.tokenizer = AutoTokenizer.from_pretrained(self.cache_dir, local_files_only=True)

            if self.quantize and self.device == "cuda":
                # Load 8-bit quantized model on GPU
                self.model = AutoModel.from_pretrained(
                    self.cache_dir, local_files_only=True, load_in_8bit=True, device_map="auto"
                )
                self._is_quantized = True
            else:
                # Standard FP32/FP16 model
                self.model = AutoModel.from_pretrained(self.cache_dir, local_files_only=True)

        except Exception:
            # First-time download
            print("Downloading e5-large model for the first time...")
            self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large", cache_dir=self.cache_dir)
            if self.quantize and torch.cuda.is_available():
                self.model = AutoModel.from_pretrained(
                    "intfloat/e5-large", cache_dir=self.cache_dir, load_in_8bit=True, device_map="auto"
                )
                self._is_quantized = True
            else:
                self.model = AutoModel.from_pretrained("intfloat/e5-large", cache_dir=self.cache_dir)
            print(f"Model downloaded and saved to {self.cache_dir}")

        # Only move to device if not quantized (quantized models are already on device via device_map)
        if not self._is_quantized:
            self.model.to(self.device)

        self.model.eval()
        print(f"Using device: {self.device}, Quantized: {self._is_quantized}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        with torch.no_grad():
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            model_output = self.model(**encodings)
            embeddings = self.mean_pooling(model_output, encodings['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
