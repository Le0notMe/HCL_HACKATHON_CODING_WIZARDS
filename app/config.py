"""Application settings, loaded from the environment or a ``.env`` file.

Every threshold lives here so a run can record exactly which values produced
its outcome (R7). The defaults are starting values, not calibrated ones: tune
them against a labelled validation set before relying on them (R6).
"""

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrincipalConfig(BaseModel):
    """An API caller. ``clearance`` is the highest access level it may read."""

    name: str
    clearance: str = "INTERNAL"
    can_review: bool = False


class Settings(BaseSettings):
    """Runtime configuration. See ``.env.example`` for the documented variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infrastructure
    database_url: str = "mysql+pymysql://ticket_app:ticket_app@127.0.0.1:3306/ticket_assistant?charset=utf8mb4"
    chroma_path: str = "./chroma_data"
    chroma_host: str | None = None
    chroma_port: int = 8000

    # API callers: {"<api key>": {"name": ..., "clearance": ..., "can_review": ...}}
    # Empty means every request is rejected: authentication fails closed.
    api_keys: dict[str, PrincipalConfig] = Field(default_factory=dict)

    # Models
    llm_provider: str = "gemini"
    llm_model: str = "gemini-flash-latest"
    llm_fallback_models: list[str] = Field(default_factory=list)
    gemini_api_key: str | None = None
    gemini_base_url: str | None = None
    anthropic_api_key: str | None = None
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    embedding_provider: str = "onnx"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Semantic cache (R2)
    cache_similarity_threshold: float = 0.95
    cache_ttl_days: int = 30

    # Retrieval (R3)
    retrieval_candidates: int = 20
    retrieval_top_k: int = 5
    bm25_corpus_limit: int = 5000
    rrf_k: int = 60
    same_service_boost: float = 0.05
    environment_mismatch_penalty: float = 0.10
    duplicate_min_score: float = 0.75

    # Evidence gate and routing (R4, R6)
    min_evidence_score: float = 0.50
    min_supporting_sources: int = 2
    conflict_dominance: float = 0.60
    contradiction_confidence_cap: float = 0.49
    confidence_threshold: float = 0.70
    llm_confidence_weight: float = 0.6
    no_neighbour_penalty: float = 0.8
    underspecified_word_count: int = 12

    # Citation verification (R5)
    min_excerpt_chars: int = 12

    # Recovery of runs orphaned by a crash or restart (R8)
    stale_run_minutes: int = 30

    def threshold_snapshot(self) -> dict[str, float | int]:
        """Return the values that affect routing, for the audit record."""
        return {
            "cache_similarity_threshold": self.cache_similarity_threshold,
            "duplicate_min_score": self.duplicate_min_score,
            "min_evidence_score": self.min_evidence_score,
            "min_supporting_sources": self.min_supporting_sources,
            "conflict_dominance": self.conflict_dominance,
            "confidence_threshold": self.confidence_threshold,
            "llm_confidence_weight": self.llm_confidence_weight,
        }


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
