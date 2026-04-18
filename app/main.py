import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.core.config import config
from app.core.factory import get_vector_store
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_vector_store(config)
    app.state.store = store
    store_type = config["vector_store"]["type"]
    logger.info("Vector store initialized: type=%s", store_type)
    yield


app = FastAPI(title="CurationService", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s -> %d  %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


app.include_router(router)
