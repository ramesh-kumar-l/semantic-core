import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import config
from app.core.factory import get_vector_store
from app.api.routes import router
from app.observability.middleware import ObservabilityMiddleware
from app.hybrid.bm25 import BM25Index
from app.ranking.simple import SimpleRankingService
from app.memory.service import MemoryService
from app.learning.feedback_store import FeedbackStore
from app.learning.l2r_rule import RuleBasedL2R
from app.learning.l2r_model import L2RModel
from app.planner.store import PlannerStore
from app.planner.rule_based import RuleBasedPlanner
from app.planner.adaptive import AdaptivePlanner
from app.graph.factory import get_graph
from app.semantic.service import SemanticService
from app.linking.engine import LinkingEngine
from app.graph_query.engine import GraphQueryEngine
from app.graph_query.planner import GraphPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_vector_store(config)
    app.state.store = store
    app.state.bm25 = BM25Index()
    app.state.ranker = SimpleRankingService()
    app.state.memory = MemoryService()

    learning_cfg = config.get("learning", {})
    feedback_path = learning_cfg.get("feedback_path", "./data/feedback.jsonl")
    model_path = learning_cfg.get("model_path", "./models/l2r.pkl")

    feedback_store = FeedbackStore(path=feedback_path)
    app.state.feedback_store = feedback_store

    l2r_rule = RuleBasedL2R()
    if learning_cfg.get("enabled", False) and learning_cfg.get("rule_based", True):
        historical = feedback_store.load_all()
        if historical:
            l2r_rule.load_from_feedback(historical)
            logger.info("L2R rule engine bootstrapped from %d feedback events", len(historical))
    app.state.l2r_rule = l2r_rule

    l2r_model = L2RModel()
    if learning_cfg.get("enabled", False) and learning_cfg.get("ml_model", False):
        loaded = l2r_model.load(model_path)
        if loaded:
            logger.info("L2R ML model loaded from %s", model_path)
        else:
            logger.info("L2R ML model not found at %s — rule-based fallback active", model_path)
    app.state.l2r_model = l2r_model

    planner_cfg = config.get("planner", {})
    planner_store = PlannerStore(path=planner_cfg.get("store_path", "./data/planner_store.json"))
    app.state.planner_store = planner_store
    if planner_cfg.get("mode", "rule_based") == "adaptive":
        app.state.planner = AdaptivePlanner(store=planner_store)
        logger.info("Query planner initialized: mode=adaptive")
    else:
        app.state.planner = RuleBasedPlanner()
        logger.info("Query planner initialized: mode=rule_based")

    graph_cfg = config.get("graph", {})
    if graph_cfg.get("enabled", True):
        graph = get_graph(config)
        app.state.graph = graph
        app.state.semantic = SemanticService(graph)
        logger.info("Graph initialized: type=%s", graph_cfg.get("type", "sqlite"))

        linking_cfg = config.get("linking", {})
        threshold = linking_cfg.get("threshold", 0.5)
        app.state.linking_engine = LinkingEngine(threshold=threshold)
        app.state.graph_planner = GraphPlanner()
        app.state.graph_query_engine = GraphQueryEngine()
        logger.info("Linking engine and graph query engine initialized")
    else:
        app.state.graph = None
        app.state.semantic = None
        app.state.linking_engine = None
        app.state.graph_planner = None
        app.state.graph_query_engine = None

    logger.info("Vector store initialized: type=%s", config["vector_store"]["type"])
    yield


app = FastAPI(title="CurationService", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
app.include_router(router)
