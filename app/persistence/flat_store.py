import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from app.persistence.errors import SchemaVersionError

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 1


def _namespace_dir(base_path: str, namespace: str) -> Path:
    return Path(base_path) / namespace / "flat"


def save(
    base_path: str,
    namespace: str,
    vectors: List[np.ndarray],
    ids: List[str],
    texts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> None:
    out_dir = _namespace_dir(base_path, namespace)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vectors:
        matrix = np.stack(vectors).astype(np.float32)
    else:
        matrix = np.empty((0,), dtype=np.float32)

    _atomic_write_npy(out_dir / "vectors.npy", matrix)
    _atomic_write_json(out_dir / "ids.json", {"schema_version": _SCHEMA_VERSION, "data": ids})
    _atomic_write_json(out_dir / "texts.json", {"schema_version": _SCHEMA_VERSION, "data": texts})
    _atomic_write_json(
        out_dir / "metadata.json",
        {"schema_version": _SCHEMA_VERSION, "data": metadata if metadata is not None else [{} for _ in ids]},
    )


def load(
    base_path: str,
    namespace: str,
) -> Optional[Tuple[List[np.ndarray], List[str], List[str], List[Dict[str, Any]]]]:
    out_dir = _namespace_dir(base_path, namespace)
    vec_path = out_dir / "vectors.npy"
    ids_path = out_dir / "ids.json"
    texts_path = out_dir / "texts.json"
    metadata_path = out_dir / "metadata.json"

    if not vec_path.exists() or not ids_path.exists():
        return None

    try:
        matrix = np.load(str(vec_path))
        ids = _load_versioned_json(ids_path, namespace)
        texts = _load_versioned_json(texts_path, namespace) if texts_path.exists() else [""] * len(ids)
        metadata = _load_versioned_json(metadata_path, namespace) if metadata_path.exists() else [{} for _ in ids]
        if len(metadata) != len(ids):
            metadata = [{} for _ in ids]
        vectors = [matrix[i] for i in range(len(matrix))] if matrix.ndim > 1 else []
        return vectors, ids, texts, metadata
    except SchemaVersionError:
        raise
    except Exception as exc:
        logger.warning("flat_store.load failed namespace=%s: %s", namespace, exc)
        return None


def _load_versioned_json(path: Path, namespace: str):
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "schema_version" in raw:
        version = raw.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise SchemaVersionError(
                f"flat_store schema mismatch for namespace='{namespace}': expected {_SCHEMA_VERSION}, got {version}"
            )
        return raw.get("data", [])
    logger.warning("flat_store namespace=%s loading legacy schema_version=0 from %s", namespace, path.name)
    return raw


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
