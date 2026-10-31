"""Hybrid retrieval (R3): vector + BM25, fused, reranked and context-adjusted.

Three separate channels are searched, because they answer different questions:

- open tickets: is this the same live issue? (duplicates)
- resolved tickets: how was this fixed before? (precedents)
- approved knowledge: what is the documented fix?

Every query carries the caller's access filter, so a record above their clearance
never becomes a candidate.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import chromadb
from rank_bm25 import BM25Okapi

from app.cache_and_security import combine_where
from app.config import Settings
from app.schemas import SourceType

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Scores query/document pairs from 0 to 1."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """ONNX cross-encoder. Its raw logits are mapped to 0-1 with a sigmoid."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 / (1.0 + math.exp(-value)) for value in self._model.rerank(query, documents)]


def build_reranker(settings: Settings) -> Reranker | None:
    """Create the configured reranker. None means candidates keep their cosine score."""
    return CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None


@dataclass
class Candidate:
    """A retrieved record. ``score`` is the value used for gating and ranking."""

    source_id: str
    source_type: SourceType
    document: str
    metadata: dict[str, Any]
    vector_score: float = 0.0
    score: float = 0.0
    matched_on: list[str] = field(default_factory=list)
    rrf: float = 0.0

    def to_state(self) -> dict[str, Any]:
        """Return a plain dict for the graph state and the audit log."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "title": self.metadata.get("title", ""),
            "status": self.metadata.get("status"),
            "category": self.metadata.get("category") or None,
            "subcategory": self.metadata.get("subcategory") or None,
            "resolution_code": self.metadata.get("resolution_code") or None,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "matched_on": self.matched_on,
        }


def tokenize(text: str) -> list[str]:
    """Lower-case tokens that keep identifiers such as ``err-1042`` intact."""
    return re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class HybridRetriever:
    """Runs one retrieval channel from query to final ranked candidates."""

    def __init__(self, settings: Settings, reranker: Reranker | None) -> None:
        self._settings = settings
        self._reranker = reranker

    def search(
        self,
        collection: chromadb.Collection,
        source_type: SourceType,
        query_text: str,
        embedding: list[float],
        where: list[dict[str, Any]],
        ticket: dict[str, Any],
    ) -> list[Candidate]:
        """Return the channel's top candidates, best first.

        The BM25 corpus is fetched per query through the same filter as the vector
        search, so access scoping cannot drift between the two. That costs one scan
        per query, capped at ``bm25_corpus_limit``.
        """
        scope = combine_where(where)
        candidates: dict[str, Candidate] = {}
        self._vector_channel(collection, source_type, embedding, scope, candidates)
        self._keyword_channel(collection, source_type, query_text, scope, candidates)
        if not candidates:
            return []

        shortlist = sorted(candidates.values(), key=lambda c: c.rrf, reverse=True)
        shortlist = shortlist[: self._settings.retrieval_candidates]
        self._fill_missing_cosine(collection, embedding, shortlist)
        self._score(query_text, shortlist)
        for candidate in shortlist:
            candidate.score = self._adjust_for_context(candidate, ticket)
        shortlist.sort(key=lambda c: c.score, reverse=True)
        return shortlist[: self._settings.retrieval_top_k]

    def _vector_channel(
        self,
        collection: chromadb.Collection,
        source_type: SourceType,
        embedding: list[float],
        scope: dict[str, Any],
        candidates: dict[str, Candidate],
    ) -> None:
        found = collection.query(
            query_embeddings=[embedding],
            n_results=self._settings.retrieval_candidates,
            where=scope,
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0])
        for rank, (source_id, document, metadata, distance) in enumerate(rows, start=1):
            candidate = Candidate(source_id, source_type, document, dict(metadata))
            candidate.vector_score = 1.0 - float(distance)
            candidate.rrf += 1.0 / (self._settings.rrf_k + rank)
            candidate.matched_on.append("vector")
            candidates[source_id] = candidate

    def _keyword_channel(
        self,
        collection: chromadb.Collection,
        source_type: SourceType,
        query_text: str,
        scope: dict[str, Any],
        candidates: dict[str, Candidate],
    ) -> None:
        limit = self._settings.bm25_corpus_limit
        corpus = collection.get(where=scope, limit=limit, include=["documents", "metadatas"])
        if not corpus["ids"]:
            return
        if len(corpus["ids"]) == limit:
            logger.warning("BM25 corpus reached its limit of %d documents", limit)
        index = BM25Okapi([tokenize(document) for document in corpus["documents"]])
        scores = index.get_scores(tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        matching = [i for i in ranked if scores[i] > 0][: self._settings.retrieval_candidates]
        for rank, position in enumerate(matching, start=1):
            source_id = corpus["ids"][position]
            candidate = candidates.get(source_id)
            if candidate is None:
                candidate = Candidate(
                    source_id, source_type, corpus["documents"][position], dict(corpus["metadatas"][position])
                )
                candidates[source_id] = candidate
            candidate.rrf += 1.0 / (self._settings.rrf_k + rank)
            candidate.matched_on.append("bm25")

    @staticmethod
    def _fill_missing_cosine(
        collection: chromadb.Collection, embedding: list[float], shortlist: list[Candidate]
    ) -> None:
        """Compute cosine for keyword-only hits, so every candidate has a semantic score."""
        missing = [c for c in shortlist if "vector" not in c.matched_on]
        if not missing:
            return
        stored = collection.get(ids=[c.source_id for c in missing], include=["embeddings"])
        vectors = dict(zip(stored["ids"], stored["embeddings"]))
        for candidate in missing:
            candidate.vector_score = _cosine(embedding, [float(v) for v in vectors[candidate.source_id]])

    def _score(self, query_text: str, shortlist: list[Candidate]) -> None:
        """Score semantically, so a keyword match alone cannot rank a candidate highly."""
        if self._reranker is None:
            for candidate in shortlist:
                candidate.score = candidate.vector_score
            return
        scores = self._reranker.score(query_text, [c.document for c in shortlist])
        for candidate, score in zip(shortlist, scores):
            candidate.score = score
            candidate.matched_on.append("rerank")

    def _adjust_for_context(self, candidate: Candidate, ticket: dict[str, Any]) -> float:
        """Apply soft service and environment signals.

        These are adjustments, not filters: reporters often name the wrong service,
        and a hard filter would discard the right match.
        """
        score = candidate.score
        if candidate.metadata.get("service") == ticket["affected_service"]:
            score += self._settings.same_service_boost
        environment = candidate.metadata.get("environment")
        if ticket.get("environment") and environment not in (None, "unspecified", ticket["environment"]):
            score -= self._settings.environment_mismatch_penalty
        return max(0.0, min(1.0, score))
