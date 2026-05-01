import math
import threading
import time
import uuid
from fastapi import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import config
from app.memory.namespace import resolve as resolve_namespace
from app.observability.logger import get_json_logger

logger = get_json_logger("observability.middleware")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    _rate_lock = threading.Lock()
    _rate_buckets: dict[str, dict[str, float]] = {}

    @staticmethod
    async def _extract_namespace(request: Request) -> str:
        if "namespace" in request.query_params:
            return resolve_namespace(request.query_params.get("namespace"))
        if request.method in {"POST", "PUT", "PATCH"} and "application/json" in request.headers.get("content-type", ""):
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    return resolve_namespace(payload.get("namespace"))
            except Exception:
                return resolve_namespace(None)
        return resolve_namespace(None)

    def _consume_rate_token(self, namespace: str) -> tuple[bool, int]:
        rate_cfg = config.get("rate_limit", {})
        rpm = max(1, int(rate_cfg.get("rpm", 60)))
        capacity = float(rpm)
        refill_per_sec = capacity / 60.0
        now = time.monotonic()

        with self._rate_lock:
            bucket = self._rate_buckets.get(namespace)
            if bucket is None:
                bucket = {"tokens": capacity, "last_refill": now}
                self._rate_buckets[namespace] = bucket
            elapsed = max(0.0, now - bucket["last_refill"])
            bucket["tokens"] = min(capacity, bucket["tokens"] + elapsed * refill_per_sec)
            bucket["last_refill"] = now
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True, 0
            missing = 1.0 - bucket["tokens"]
            retry_after = int(math.ceil(missing / refill_per_sec))
            return False, max(1, retry_after)

    async def dispatch(self, request: Request, call_next):
        rate_cfg = config.get("rate_limit", {})
        exempt = request.url.path.startswith(("/admin/health", "/docs", "/openapi.json", "/redoc"))
        if rate_cfg.get("enabled", False) and not exempt:
            namespace = await self._extract_namespace(request)
            allowed, retry_after = self._consume_rate_token(namespace)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={"detail": "Rate limit exceeded"},
                )

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        t0 = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "endpoint": request.url.path,
                "status_code": response.status_code,
                "total_ms": round(total_ms, 2),
            },
        )
        return response
