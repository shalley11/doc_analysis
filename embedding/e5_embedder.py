import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Callable, Optional


# -------------------------------
# CPU PERFORMANCE TUNING
# -------------------------------
CPU_CORES = os.cpu_count() or 4
torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(1)

# Use local model path from env, fallback to HuggingFace name
_MODEL_PATH = os.environ.get("E5_MODEL_PATH", "intfloat/e5-large-v2")

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, local_files_only=True)
_model = AutoModel.from_pretrained(_MODEL_PATH, local_files_only=True)
_model.eval()  # important


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1)
    return summed / counts


def embed_passages(
    texts: List[str],
    batch_size: int = 8,
    progress_cb: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    CPU-optimized E5 embedding with batching + progress.

    Args:
        texts: list of passage-prefixed texts
        batch_size: CPU-optimal batch size (8–16)
        progress_cb: optional callback(current, total)

    Returns:
        List of normalized embeddings
    """

    all_embeddings = []
    total = len(texts)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            inputs = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            outputs = _model(**inputs)

            embeddings = _mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

            all_embeddings.extend(
                embeddings.cpu().numpy().tolist()
            )

            if progress_cb:
                progress_cb(end, total)

    return all_embeddings
