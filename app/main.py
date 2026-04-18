import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import config
from app.core.factory import get_vector_store
from app.api.routes import router
from app.observability.middleware import ObservabilityMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_vector_store(config)
    app.state.store = store
    logger.info("Vector store initialized: type=%s", config["vector_store"]["type"])
    yield


app = FastAPI(title="CurationService", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
app.include_router(router)
