from typing import Any, List
import hashlib
import struct
import numpy as np

# Attempt real embeddings; fall back to deterministic stub if unavailable.
try:
    from sentence_transformers import SentenceTransformer as _ST

    _model = _ST("all-MiniLM-L6-v2")
    _USE_REAL = True
except Exception:
    _USE_REAL = False

DIM = 384


class EmbeddingService:
    def embed(self, input_data: Any) -> List[float]:
        text = str(input_data)
        if _USE_REAL:
            vec = _model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        return _hash_embed(text)


def _hash_embed(text: str) -> List[float]:
    """Deterministic stub: sha256-seeded pseudo-random unit vector."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()
