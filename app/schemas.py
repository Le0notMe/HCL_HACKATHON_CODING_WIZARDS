"""Pydantic contracts for the API, the LLM responses and the stored analysis."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_CHARS = 20_000
MAX_CHAT_MESSAGE_CHARS = 4_000
MAX_CHAT_MESSAGES = 30

TAXONOMY: dict[str, list[str]] = {
    "Access Management": ["VPN Authentication", "Password Reset", "Single Sign-On", "Permissions"],
    "Application Failure": ["Crash", "Performance", "Integration Error"],
    "Configuration": ["Network Configuration", "Certificate", "Software Configuration"],
    "Infrastructure": ["Server", "Storage", "Network Outage"],
    "Email and Collaboration": ["Mailbox", "Calendar", "Messaging"],
}
PRIORITIES = ("P1", "P2", "P3", "P4")
INTENTS = ("restore_access", "fix_incident", "change_configuration", "information_request")


class AccessLevel(str, Enum):
    """Sensitivity of a record, and clearance of a caller."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"

    @property
    def rank(self) -> int:
        """Return an integer so levels can be compared and range-filtered."""
        return {"PUBLIC": 0, "INTERNAL": 1, "RESTRICTED": 2}[self.value]


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Outcome(str, Enum):
    RECOMMEND = "RECOMMEND"
    CLARIFY = "CLARIFY"
    ESCALATE = "ESCALATE"


class FeedbackTarget(str, Enum):
    MATCH = "MATCH"
    RECOMMENDATION = "RECOMMENDATION"


class Verdict(str, Enum):
    USEFUL = "USEFUL"
    INCORRECT = "INCORRECT"
    UNRESOLVED = "UNRESOLVED"


class SourceType(str, Enum):
    TICKET = "TICKET"
    KNOWLEDGE = "KNOWLEDGE"


class ChatKind(str, Enum):
    ANSWER = "ANSWER"  # Backed by at least one verified citation.
    FOLLOW_UP = "FOLLOW_UP"  # A greeting or a question back, which claims no fix.
    NO_ANSWER = "NO_ANSWER"  # No reply could be trusted, so an analyst should take over.


class StrictModel(BaseModel):
    """Base for request bodies: unknown fields are errors, strings are trimmed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TicketIn(StrictModel):
    """Incoming ticket. ``ticket_id`` is the idempotency key."""

    ticket_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9._-]+$")
    title: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=3, max_length=MAX_TEXT_CHARS)
    symptoms: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    affected_service: str = Field(min_length=1, max_length=128)
    environment: str | None = Field(default=None, max_length=32)
    assignment_group: str | None = Field(default=None, max_length=64)
    access_level: AccessLevel = AccessLevel.INTERNAL

    @field_validator("environment")
    @classmethod
    def normalise_environment(cls, value: str | None) -> str | None:
        """Lower-case the environment so 'PROD' and 'prod' filter identically."""
        return value.lower() if value else None


class FeedbackIn(StrictModel):
    """An analyst verdict on one match, or on the recommendation as a whole."""

    target_type: FeedbackTarget
    target_ref: str = Field(
        min_length=1,
        max_length=64,
        description="A source id for MATCH, or 'recommendation' for RECOMMENDATION.",
    )
    verdict: Verdict
    comment: str | None = Field(default=None, max_length=2_000)


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)


class ChatRequest(StrictModel):
    """A conversation so far, ending with the user's new message. The server keeps no history."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)

    @field_validator("messages")
    @classmethod
    def must_end_with_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("The last message must come from the user")
        return value


# Structured LLM responses. Field descriptions are sent to the model as part of the schema.


class ClassificationOutput(BaseModel):
    """FR-01: the model's classification of one ticket."""

    category: str = Field(description="One category from the taxonomy.")
    subcategory: str = Field(description="One subcategory of the chosen category.")
    priority: str = Field(description="One of P1, P2, P3 or P4.")
    intent: str = Field(description="One intent from the allowed list.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0 in the category.")
    rationale: str = Field(description="One sentence citing the ticket fields used.")


class EvidenceClaim(BaseModel):
    """FR-04: one source supporting a resolution path."""

    source_id: str = Field(description="The id of a supplied source. Never invent one.")
    excerpt: str = Field(description="A verbatim quote copied from that source's text.")
    reason: str = Field(description="Why this excerpt supports the resolution path.")


class ResolutionPath(BaseModel):
    """FR-03: one candidate resolution with its evidence."""

    summary: str
    steps: list[str]
    evidence: list[EvidenceClaim]


class SynthesisOutput(BaseModel):
    """The model's grounded resolution proposal."""

    resolution_paths: list[ResolutionPath] = Field(
        description="Paths supported by the sources. Empty if they support none."
    )
    evidence_conflict: bool = Field(
        description="True if the sources recommend incompatible resolutions."
    )
    clarifying_questions: list[str] = Field(
        description="Questions for the reporter, if missing detail blocks a recommendation."
    )


class ChatOutput(BaseModel):
    """The model's reply to one chat turn."""

    answer: str = Field(description="The reply to the user, as plain text.")
    is_resolution: bool = Field(
        description="True if the reply tells the user how to fix or diagnose something."
    )
    evidence: list[EvidenceClaim] = Field(description="Sources supporting the reply.")
    evidence_conflict: bool = Field(
        description="True if the sources recommend incompatible resolutions."
    )


# Stored analysis, returned by GET /runs/{run_id}.


class Classification(BaseModel):
    category: str
    subcategory: str
    priority: str
    intent: str
    confidence: float
    llm_confidence: float
    neighbour_agreement: float | None
    rationale: str


class MatchOut(BaseModel):
    """A retrieved source shown to the analyst, whatever the outcome (R4)."""

    source_id: str
    source_type: SourceType
    title: str
    score: float
    vector_score: float
    matched_on: list[str]
    status: str | None = None


class VerifiedEvidence(BaseModel):
    source_id: str
    source_type: SourceType
    excerpt: str
    reason: str


class VerifiedPath(BaseModel):
    summary: str
    steps: list[str]
    evidence: list[VerifiedEvidence]


class CitationReport(BaseModel):
    """R5: the result of deterministic citation verification."""

    claims_total: int = 0
    claims_verified: int = 0
    rejected: list[dict[str, str]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """The complete analysis of one ticket."""

    outcome: Outcome
    confidence: float
    confidence_breakdown: dict[str, float]
    reasons: list[str]
    classification: Classification | None
    duplicates: list[MatchOut]
    similar_cases: list[MatchOut]
    recommendation: list[VerifiedPath]
    clarifying_questions: list[str]
    citations: CitationReport
    cache_hit: bool = False
    cache_source_run_id: str | None = None


class AnalyzeAccepted(BaseModel):
    """Response to POST /tickets/analyze."""

    run_id: str
    ticket_id: str
    status: RunStatus
    deduplicated: bool
    status_url: str


class FeedbackOut(BaseModel):
    feedback_id: int
    target_type: FeedbackTarget
    target_ref: str
    verdict: Verdict
    reviewer_id: str
    comment: str | None
    created_at: datetime


class RunOut(BaseModel):
    run_id: str
    ticket_id: str
    status: RunStatus
    outcome: Outcome | None
    confidence: float | None
    cache_hit: bool
    result: AnalysisResult | None
    error: str | None
    feedback: list[FeedbackOut]
    created_at: datetime
    completed_at: datetime | None


class ChatReply(BaseModel):
    """Response to POST /chat. ``related`` lists retrieved sources, whatever the outcome."""

    kind: ChatKind
    answer: str
    evidence: list[VerifiedEvidence]
    related: list[MatchOut]
    citations: CitationReport


class TicketOut(BaseModel):
    """A ticket search result. All free text is already redacted."""

    ticket_id: str
    title: str
    affected_service: str
    environment: str | None
    category: str | None
    subcategory: str | None
    priority: str | None
    status: str
    assignment_group: str | None
    created_at: datetime


class TicketSearchOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[TicketOut]
