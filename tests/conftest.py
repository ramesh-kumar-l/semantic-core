import os
import tempfile
import pytest

# Set paths to a temp dir before any app module is imported.
# conftest.py is processed first by pytest, so these os.environ calls
# run before test files are collected and app.core.config is read.
_tmp = tempfile.mkdtemp(prefix="scs_test_")
os.environ["PERSISTENCE_BASE_PATH"] = _tmp
os.environ["LEARNING_FEEDBACK_PATH"] = os.path.join(_tmp, "feedback.jsonl")
os.environ["LEARNING_MODEL_PATH"] = os.path.join(_tmp, "l2r.pkl")
os.environ["PLANNER_STORE_PATH"] = os.path.join(_tmp, "planner_store.json")
os.environ["GRAPH_PATH"] = os.path.join(_tmp, "graph.db")
os.environ["GRAPH_ENABLED"] = "false"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
