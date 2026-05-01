import hmac
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import config
from app.memory.namespace import resolve as resolve_namespace
from app.observability.logger import get_json_logger

logger = get_json_logger("auth.middleware")

_EXEMPT_PREFIXES = ("/admin/health", "/docs", "/openapi.json", "/redoc")


class AuthMiddleware(BaseHTTPMiddleware):
    def _extract_api_key(self, request: Request) -> str:
        return request.headers.get("X-API-Key", "").strip()

    async def _extract_namespace(self, request: Request) -> str:
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

    def _is_valid_key(self, key: str, valid_keys: set[str]) -> bool:
        return any(hmac.compare_digest(key, candidate) for candidate in valid_keys)

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
            return await call_next(request)

        auth_cfg = config.get("auth", {})
        if not auth_cfg.get("enabled", False):
            return await call_next(request)

        key = self._extract_api_key(request)
        if not key:
            return JSONResponse(status_code=401, content={"detail": "Missing API key"})

        api_keys = auth_cfg.get("api_keys", set())
        namespace_keys = auth_cfg.get("namespace_keys", {})
        if not self._is_valid_key(key, api_keys | set(namespace_keys.keys())):
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        namespace = await self._extract_namespace(request)
        scoped_namespaces: Optional[list[str]] = namespace_keys.get(key)
        if scoped_namespaces is not None and namespace not in scoped_namespaces:
            logger.warning(
                "auth_forbidden_namespace",
                extra={"namespace": namespace, "path": request.url.path},
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"API key not authorized for namespace '{namespace}'"},
            )

        return await call_next(request)
