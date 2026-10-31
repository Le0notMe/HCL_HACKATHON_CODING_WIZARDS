"""Redaction, access scoping (R1) and the scoped semantic cache (R2).

Redaction runs in the request path, before anything is stored, embedded or sent
to a model, so raw ticket text never reaches MySQL, ChromaDB, the audit log or
the LLM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.config import PrincipalConfig, Settings
from app.database import VectorStore, epoch_now
from app.schemas import AccessLevel, TicketIn

logger = logging.getLogger(__name__)


def _luhn_valid(number: str) -> bool:
    """Return True if a digit string passes the card-number checksum."""
    digits = [int(char) for char in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


# Order matters: secrets first, so a token is not half-consumed by a broader pattern.
_PATTERNS: list[tuple[str, re.Pattern[str], Callable[[re.Match[str]], bool] | None]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), None),
    ("JWT", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b"), None),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), None),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[\w.~+/=-]{12,}", re.I), None),
    (
        "SECRET",
        re.compile(
            r"(?<![\w-])(?:password|passwd|pwd|passphrase|secret|api[_-]?key|access[_-]?token|token)"
            r"(?:\s+is\s+|\s*[:=]\s*)['\"]?[^\s'\",;]{4,}['\"]?",
            re.I,
        ),
        None,
    ),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), None),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), lambda match: _luhn_valid(match.group())),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?|\d{3,4}[ -])\d{3}[ -]?\d{4}\b"), None),
    ("PHONE", re.compile(r"(?<![\w-])\+\d{10,14}\b"), None),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), None),
]

_INJECTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|prior|above|earlier|all)\b.{0,20}\b(?:instructions?|rules?|prompts?)\b",
        r"\byou are now\b",
        r"\b(?:system|developer) prompt\b",
        r"\b(?:reveal|print|show)\b.{0,30}\b(?:instructions?|prompt)\b",
        r"</?untrusted_data>",
    )
]


@dataclass(frozen=True)
class RedactionReport:
    """Counts of what was masked, safe to log and audit, plus the injection flag."""

    counts: dict[str, int]
    injection_suspected: bool

    def to_dict(self) -> dict[str, Any]:
        return {"redactions": self.counts, "injection_suspected": self.injection_suspected}


def redact_text(text: str) -> tuple[str, Counter[str]]:
    """Replace PII and secrets with ``[REDACTED_<TYPE>]`` markers.

    This is pattern-based. It catches structured values but not names or free-form
    personal details, which need an NER model or data-loss-prevention service.
    """
    counts: Counter[str] = Counter()
    for label, pattern, is_match in _PATTERNS:

        def replace(match: re.Match[str]) -> str:
            if is_match is not None and not is_match(match):
                return match.group()
            counts[label] += 1
            return f"[REDACTED_{label}]"

        text = pattern.sub(replace, text)
    return text, counts


def looks_like_injection(text: str) -> bool:
    """Flag text that tries to instruct the model. Flagged tickets are escalated."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def prepare_ticket(ticket: TicketIn) -> tuple[dict[str, Any], RedactionReport]:
    """Return the redacted ticket as column values, with a report of what changed."""
    counts: Counter[str] = Counter()
    original_text = " ".join(filter(None, (ticket.title, ticket.description, ticket.symptoms)))
    cleaned: dict[str, str | None] = {}
    for field in ("title", "description", "symptoms"):
        value = getattr(ticket, field)
        cleaned[field], found = redact_text(value) if value else (None, Counter())
        counts.update(found)
    prepared = {
        "ticket_id": ticket.ticket_id,
        "title": cleaned["title"][:255],
        "description": cleaned["description"],
        "symptoms": cleaned["symptoms"],
        "affected_service": ticket.affected_service,
        "environment": ticket.environment,
        "assignment_group": ticket.assignment_group,
        "status": "OPEN",
        "access_rank": ticket.access_level.rank,
    }
    return prepared, RedactionReport(dict(counts), looks_like_injection(original_text))


def payload_fingerprint(prepared: dict[str, Any]) -> str:
    """Hash the redacted ticket, to detect a reused ``ticket_id`` with new content."""
    keys = ("title", "description", "symptoms", "affected_service", "environment", "access_rank")
    canonical = json.dumps({key: prepared[key] for key in keys}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def ticket_document(ticket: dict[str, Any]) -> str:
    """Build the text that is embedded and indexed for a ticket (FR-02)."""
    parts = [
        ticket["title"],
        ticket["description"],
        ticket.get("symptoms"),
        f"Service: {ticket['affected_service']}",
    ]
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    name: str
    clearance: AccessLevel
    can_review: bool


def authenticate(api_key: str | None, keys: dict[str, PrincipalConfig]) -> Principal | None:
    """Resolve an API key in constant time. Returns None if it is not recognised."""
    if not api_key:
        return None
    found: PrincipalConfig | None = None
    for candidate, config in keys.items():
        if hmac.compare_digest(candidate.encode(), api_key.encode()):
            found = config
    if found is None:
        return None
    return Principal(found.name, AccessLevel(found.clearance), found.can_review)


def combine_where(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Chroma filter. ``$and`` needs at least two operands."""
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass(frozen=True)
class CacheHit:
    """A semantic cache match pointing at the approved run that produced it."""

    source_run_id: str
    similarity: float


class SemanticCache:
    """Cache of analyst-approved analyses, scoped so that a hit is safe to reuse.

    An entry only matches a ticket with the same service, environment and clearance.
    Entries expire, and are tied to the embedding model so a model change invalidates
    them. The full analysis stays in ``analysis_runs``; an entry only points to it.
    """

    def __init__(self, vectors: VectorStore, settings: Settings, embedding_version: str) -> None:
        self._collection = vectors.cache
        self._threshold = settings.cache_similarity_threshold
        self._ttl_seconds = settings.cache_ttl_days * 86_400
        self._embedding_version = embedding_version

    def lookup(
        self, embedding: list[float], service: str, environment: str | None, clearance_rank: int
    ) -> CacheHit | None:
        """Return the nearest in-scope entry above the threshold, if there is one."""
        where = combine_where(
            [
                {"service": service},
                {"environment": environment or "unspecified"},
                {"clearance_rank": clearance_rank},
                {"embedding_version": self._embedding_version},
                {"expires_at": {"$gt": epoch_now()}},
            ]
        )
        matches = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where=where,
            include=["distances", "metadatas"],
        )
        if not matches["ids"][0]:
            return None
        similarity = 1.0 - float(matches["distances"][0][0])
        if similarity <= self._threshold:
            return None
        return CacheHit(matches["metadatas"][0][0]["source_run_id"], round(similarity, 4))

    def store(
        self,
        run_id: str,
        embedding: list[float],
        service: str,
        environment: str | None,
        clearance_rank: int,
    ) -> None:
        """Add an approved run. Called only after an analyst marks it useful."""
        self._collection.upsert(
            ids=[run_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "source_run_id": run_id,
                    "service": service,
                    "environment": environment or "unspecified",
                    "clearance_rank": clearance_rank,
                    "embedding_version": self._embedding_version,
                    "expires_at": epoch_now() + self._ttl_seconds,
                }
            ],
        )

    def invalidate(self, run_id: str) -> None:
        """Remove the entry for a run, after an analyst marks it incorrect."""
        self._collection.delete(ids=[run_id])
