import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from app.persistence.errors import SchemaVersionError

if TYPE_CHECKING:
    from app.hybrid.bm25 import BM25Index

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 1


def _namespace_dir(base_path: str, namespace: str) -> Path:
    return Path(base_path) / namespace / "bm25"


def save(base_path: str, namespace: str, bm25: "BM25Index") -> None:
    out_dir = _namespace_dir(base_path, namespace)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": _SCHEMA_VERSION,
        "data": {
            "ids": bm25._ids,
            "doc_tokens": bm25._doc_tokens,
            "df": dict(bm25._df),
            "avg_dl": bm25._avg_dl,
        },
    }
    _atomic_write_json(out_dir / "index.json", data)


def load(base_path: str, namespace: str) -> Optional["BM25Index"]:
    from app.hybrid.bm25 import BM25Index

    index_path = _namespace_dir(base_path, namespace) / "index.json"
    if not index_path.exists():
        return None

    try:
        payload = json.loads(index_path.read_text())
        if isinstance(payload, dict) and "schema_version" in payload:
            version = payload.get("schema_version")
            if version != _SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"bm25_store schema mismatch for namespace='{namespace}': expected {_SCHEMA_VERSION}, got {version}"
                )
            data = payload.get("data", {})
        else:
            logger.warning("bm25_store namespace=%s loading legacy schema_version=0", namespace)
            data = payload
        bm25 = BM25Index()
        bm25._ids = data["ids"]
        bm25._doc_tokens = data["doc_tokens"]
        bm25._df = defaultdict(int, data["df"])
        bm25._avg_dl = data["avg_dl"]
        return bm25
    except SchemaVersionError:
        raise
    except Exception as exc:
        logger.warning("bm25_store.load failed namespace=%s: %s", namespace, exc)
        return None


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
