import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _namespace_dir(base_path: str, namespace: str) -> Path:
    return Path(base_path) / namespace / "flat"


def save(base_path: str, namespace: str, vectors: List[np.ndarray], ids: List[str], texts: List[str]) -> None:
    out_dir = _namespace_dir(base_path, namespace)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vectors:
        matrix = np.stack(vectors).astype(np.float32)
    else:
        matrix = np.empty((0,), dtype=np.float32)

    _atomic_write_npy(out_dir / "vectors.npy", matrix)
    _atomic_write_json(out_dir / "ids.json", ids)
    _atomic_write_json(out_dir / "texts.json", texts)


def load(base_path: str, namespace: str) -> Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
    out_dir = _namespace_dir(base_path, namespace)
    vec_path = out_dir / "vectors.npy"
    ids_path = out_dir / "ids.json"
    texts_path = out_dir / "texts.json"

    if not vec_path.exists() or not ids_path.exists():
        return None

    try:
        matrix = np.load(str(vec_path))
        ids: List[str] = json.loads(ids_path.read_text())
        texts: List[str] = json.loads(texts_path.read_text()) if texts_path.exists() else [""] * len(ids)
        vectors = [matrix[i] for i in range(len(matrix))] if matrix.ndim > 1 else []
        return vectors, ids, texts
    except Exception as exc:
        logger.warning("flat_store.load failed namespace=%s: %s", namespace, exc)
        return None


def _atomic_write_npy(path: Path, data: np.ndarray) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, data: object) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
