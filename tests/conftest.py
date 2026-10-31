"""Shared fixtures: a real graph, ChromaDB and SQL engine, with a scripted LLM.

SQLite stands in for MySQL and the hashing embedder for the ONNX model. The stub LLM
lets each test choose exactly what the model returns, including bad output.
"""

from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from fastapi.testclient import TestClient

from app.agent.workflow import Deps, build_graph
from app.cache_and_security import SemanticCache
from app.config import PrincipalConfig, Settings
from app.database import Repository, VectorStore
from app.embeddings import HashingEmbedder
from app.llm import OfflineLLM
from app.main import create_app
from app.retrieval import HybridRetriever
from scripts.seed_index import seed

DATA = Path(__file__).resolve().parent.parent / "data"
ANALYST = {"X-API-Key": "analyst-key"}
SERVICE = {"X-API-Key": "service-key"}
ADMIN = {"X-API-Key": "admin-key"}
PUBLIC_REVIEWER = {"X-API-Key": "public-key"}


class StubLLM(OfflineLLM):
    """Offline behaviour by default, overridable per test."""

    def __init__(self):
        self.classify_calls = 0
        self.synthesize_calls = 0
        self.last_sources = []
        self.classify_output = None
        self.synthesize_output = None
        self.classify_error = None
        self.synthesize_error = None

    def classify(self, ticket):
        self.classify_calls += 1
        if self.classify_error:
            raise self.classify_error
        return self.classify_output or super().classify(ticket)

    def synthesize(self, ticket, sources):
        self.synthesize_calls += 1
        self.last_sources = sources
        if self.synthesize_error:
            raise self.synthesize_error
        return self.synthesize_output or super().synthesize(ticket, sources)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="hashing",
        reranker_model=None,
        llm_provider="offline",
        cache_similarity_threshold=0.95,
        duplicate_min_score=0.60,
        min_evidence_score=0.40,
        confidence_threshold=0.45,
        conflict_dominance=0.60,
        api_keys={
            "analyst-key": PrincipalConfig(name="analyst", clearance="INTERNAL", can_review=True),
            "service-key": PrincipalConfig(name="itsm", clearance="INTERNAL", can_review=False),
            "admin-key": PrincipalConfig(name="secops", clearance="RESTRICTED", can_review=True),
            "public-key": PrincipalConfig(name="kiosk", clearance="PUBLIC", can_review=True),
        },
    )


@pytest.fixture
def deps(settings):
    repo = Repository(settings.database_url)
    vectors = VectorStore(settings)
    embedder = HashingEmbedder()
    deps = Deps(
        settings=settings,
        repo=repo,
        vectors=vectors,
        embedder=embedder,
        llm=StubLLM(),
        retriever=HybridRetriever(settings, None),
        cache=SemanticCache(vectors, settings, embedder.version),
    )
    deps.graph = build_graph(deps)
    seed(repo, vectors, embedder, DATA / "tickets.json", DATA / "knowledge.json")
    yield deps
    # Chroma has no public close, and each client otherwise holds its files until exit.
    vectors.client._system.stop()
    SharedSystemClient.clear_system_cache()
    repo.engine.dispose()


@pytest.fixture
def client(deps):
    with TestClient(create_app(deps)) as test_client:
        yield test_client


VPN_TICKET = {
    "ticket_id": "NEW-1",
    "title": "GlobalProtect VPN error GP-4012 expired client certificate",
    "description": "User cannot connect to the GlobalProtect VPN, error GP-4012 says the client certificate has expired.",
    "symptoms": "GP-4012 when connecting to VPN.",
    "affected_service": "VPN",
    "environment": "prod",
}


@pytest.fixture
def analyze(client):
    """Submit a ticket and return the completed run."""

    def submit(headers=ANALYST, **overrides):
        body = {**VPN_TICKET, **overrides}
        accepted = client.post("/api/v1/tickets/analyze", json=body, headers=headers)
        assert accepted.status_code == 202, accepted.text
        return client.get(accepted.json()["status_url"], headers=headers).json()

    return submit
