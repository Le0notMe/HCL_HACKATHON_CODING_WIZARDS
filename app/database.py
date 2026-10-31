"""MySQL models and repository, plus the ChromaDB collections.

MySQL is the system of record. ChromaDB is a derived index that can be rebuilt
from it, joined on ``ticket_id`` / ``article_id``. Only redacted text is written
to either store.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    DDL,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.schemas import ChatKind, FeedbackIn, FeedbackTarget, Outcome, RunStatus, Verdict

LongText = Text().with_variant(mysql.MEDIUMTEXT(), "mysql")
BigIntPk = BigInteger().with_variant(Integer, "sqlite")

OPEN_STATUSES = ("OPEN", "IN_PROGRESS")
TICKET_COLLECTION = "ticket_embeddings"
KNOWLEDGE_COLLECTION = "knowledge_embeddings"
CACHE_COLLECTION = "semantic_cache"


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime, as MySQL stores it."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    metadata = MetaData()


class Ticket(Base):
    """Master ticket record. Free text is redacted before it reaches this table."""

    __tablename__ = "tickets"
    __table_args__ = (
        Index("idx_tickets_filter", "affected_service", "environment", "status"),
        Index("idx_tickets_category", "category", "subcategory"),
    )

    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(LongText)
    symptoms: Mapped[str | None] = mapped_column(LongText)
    affected_service: Mapped[str] = mapped_column(String(128))
    environment: Mapped[str | None] = mapped_column(String(32))
    category: Mapped[str | None] = mapped_column(String(64))
    subcategory: Mapped[str | None] = mapped_column(String(64))
    priority: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(32), default="OPEN")
    assignment_group: Mapped[str | None] = mapped_column(String(64))
    resolution_notes: Mapped[str | None] = mapped_column(LongText)
    resolution_code: Mapped[str | None] = mapped_column(String(64))
    access_rank: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


class KnowledgeArticle(Base):
    """Approved knowledge. Only rows with ``approved`` set may be used as evidence."""

    __tablename__ = "knowledge_articles"

    article_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(LongText)
    affected_service: Mapped[str | None] = mapped_column(String(128))
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    access_rank: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AnalysisRun(Base):
    """One analysis of one ticket.

    The unique ``ticket_id`` makes it the idempotency record. ``clearance_rank``
    is the access level the run retrieved at, so its stored evidence can only be
    read back at that level or above.
    """

    __tablename__ = "analysis_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.ticket_id"), unique=True)
    payload_hash: Mapped[str] = mapped_column(String(64))
    requested_by: Mapped[str] = mapped_column(String(64))
    clearance_rank: Mapped[int] = mapped_column(Integer)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.QUEUED)
    outcome: Mapped[Outcome | None] = mapped_column(Enum(Outcome))
    confidence: Mapped[float | None] = mapped_column(Float)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_source_run_id: Mapped[str | None] = mapped_column(String(36))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    embedding_version: Mapped[str] = mapped_column(String(128))
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    feedback: Mapped[list[AnalystFeedback]] = relationship(order_by="AnalystFeedback.feedback_id")


class AuditEvent(Base):
    """Append-only trail, written by each node as it completes (R7).

    Database triggers, defined below, reject UPDATE and DELETE on this table.
    """

    __tablename__ = "audit_events"
    __table_args__ = (Index("idx_audit_run", "run_id", "event_id"),)

    event_id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.run_id"))
    step: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AnalystFeedback(Base):
    """FR-06: a reviewer's verdict. Feedback is added, never edited."""

    __tablename__ = "analyst_feedback"

    feedback_id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.run_id"), index=True)
    target_type: Mapped[FeedbackTarget] = mapped_column(Enum(FeedbackTarget))
    target_ref: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[Verdict] = mapped_column(Enum(Verdict))
    reviewer_id: Mapped[str] = mapped_column(String(64))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChatEvent(Base):
    """Append-only trail of chat turns. Conversations have no run, so they have their own table.

    ``payload`` holds only the redacted question and metadata about the reply.
    """

    __tablename__ = "chat_events"

    event_id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)
    requested_by: Mapped[str] = mapped_column(String(64), index=True)
    clearance_rank: Mapped[int] = mapped_column(Integer)
    kind: Mapped[ChatKind] = mapped_column(Enum(ChatKind))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


for table in (AuditEvent.__table__, ChatEvent.__table__):
    message = f"{table.name} is append-only"
    for operation in ("UPDATE", "DELETE"):
        name = f"{table.name}_no_{operation.lower()}"
        event.listen(
            table,
            "after_create",
            DDL(
                f"CREATE TRIGGER {name} BEFORE {operation} ON {table.name} FOR EACH ROW "
                f"SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = '{message}'"
            ).execute_if(dialect="mysql"),
        )
        event.listen(
            table,
            "after_create",
            DDL(
                f"CREATE TRIGGER {name} BEFORE {operation} ON {table.name} "
                f"BEGIN SELECT RAISE(ABORT, '{message}'); END"
            ).execute_if(dialect="sqlite"),
        )


class IdempotencyConflict(Exception):
    """The same ``ticket_id`` was submitted again with different content."""


class Repository:
    """All SQL access. Each method runs in its own short transaction."""

    def __init__(self, database_url: str) -> None:
        options: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            options = {"connect_args": {"check_same_thread": False}, "poolclass": NullPool}
        self.engine = create_engine(database_url, **options)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        """Create any missing tables, along with the audit triggers."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session that commits on success and rolls back on error."""
        with self._sessions() as session, session.begin():
            yield session

    def create_run(
        self,
        ticket: dict[str, Any],
        payload_hash: str,
        requested_by: str,
        clearance_rank: int,
        versions: dict[str, Any],
    ) -> tuple[AnalysisRun, bool]:
        """Create the run for a ticket, or return the existing one.

        Returns ``(run, created)``. The unique constraint on ``ticket_id`` makes
        this safe under concurrent webhook retries. A FAILED run with the same
        content is requeued, so a transient failure can be retried by resubmitting.

        Raises:
            IdempotencyConflict: the ``ticket_id`` was used for different content.
        """
        run_id = str(uuid.uuid4())
        try:
            with self.session() as session:
                self._ensure_ticket(session, ticket)
                session.add(
                    AnalysisRun(
                        run_id=run_id,
                        ticket_id=ticket["ticket_id"],
                        payload_hash=payload_hash,
                        requested_by=requested_by,
                        clearance_rank=clearance_rank,
                        **versions,
                    )
                )
        except IntegrityError:
            return self._existing_run(ticket["ticket_id"], payload_hash)
        return self.get_run(run_id), True

    @staticmethod
    def _ensure_ticket(session: Session, ticket: dict[str, Any]) -> None:
        """Insert the ticket unless it already exists, as seeded tickets do."""
        if session.get(Ticket, ticket["ticket_id"]) is None:
            session.add(Ticket(**ticket))
            session.flush()

    def _existing_run(self, ticket_id: str, payload_hash: str) -> tuple[AnalysisRun, bool]:
        """Resolve a duplicate submission to its run, requeueing it if it had failed."""
        with self.session() as session:
            run = session.scalar(select(AnalysisRun).where(AnalysisRun.ticket_id == ticket_id))
            if run is None:
                raise IdempotencyConflict(f"Could not create a run for {ticket_id}")
            if run.payload_hash != payload_hash:
                raise IdempotencyConflict(f"{ticket_id} was already submitted with different content")
            requeued = session.execute(
                update(AnalysisRun)
                .where(AnalysisRun.run_id == run.run_id, AnalysisRun.status == RunStatus.FAILED)
                .values(status=RunStatus.QUEUED, error=None)
            )
            run_id = run.run_id
        return self.get_run(run_id), requeued.rowcount == 1

    def get_run(self, run_id: str) -> AnalysisRun | None:
        """Return a run with its feedback, or None."""
        with self.session() as session:
            run = session.get(AnalysisRun, run_id)
            if run is not None:
                _ = run.feedback
            return run

    def mark_running(self, run_id: str) -> bool:
        """Claim a queued run. Returns False if another worker already has it."""
        with self.session() as session:
            claimed = session.execute(
                update(AnalysisRun)
                .where(AnalysisRun.run_id == run_id, AnalysisRun.status == RunStatus.QUEUED)
                .values(status=RunStatus.RUNNING)
            )
            return claimed.rowcount == 1

    def complete_run(
        self,
        run_id: str,
        outcome: Outcome,
        confidence: float,
        result: dict[str, Any],
        cache_source_run_id: str | None,
    ) -> None:
        """Store the final analysis."""
        with self.session() as session:
            session.execute(
                update(AnalysisRun)
                .where(AnalysisRun.run_id == run_id)
                .values(
                    status=RunStatus.COMPLETED,
                    outcome=outcome,
                    confidence=confidence,
                    result=result,
                    cache_hit=cache_source_run_id is not None,
                    cache_source_run_id=cache_source_run_id,
                    completed_at=utcnow(),
                )
            )

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed. The message must not contain ticket text."""
        with self.session() as session:
            session.execute(
                update(AnalysisRun)
                .where(AnalysisRun.run_id == run_id)
                .values(status=RunStatus.FAILED, error=error[:255], completed_at=utcnow())
            )

    def fail_stale_runs(self, older_than_minutes: int) -> int:
        """Fail runs orphaned by a restart so that resubmitting can retry them."""
        cutoff = utcnow() - timedelta(minutes=older_than_minutes)
        with self.session() as session:
            stale = session.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]),
                    AnalysisRun.created_at < cutoff,
                )
                .values(status=RunStatus.FAILED, error="Interrupted before completion", completed_at=utcnow())
            )
            return stale.rowcount

    def append_audit_event(self, run_id: str, step: str, payload: dict[str, Any]) -> None:
        """Append one audit event, committed immediately so it survives a later failure."""
        with self.session() as session:
            session.add(AuditEvent(run_id=run_id, step=step, payload=payload))

    def add_chat_event(
        self, requested_by: str, clearance_rank: int, kind: ChatKind, payload: dict[str, Any]
    ) -> None:
        """Record one chat turn."""
        with self.session() as session:
            session.add(
                ChatEvent(requested_by=requested_by, clearance_rank=clearance_rank, kind=kind, payload=payload)
            )

    def list_chat_events(self) -> list[ChatEvent]:
        """Return the chat trail, oldest first."""
        with self.session() as session:
            return list(session.scalars(select(ChatEvent).order_by(ChatEvent.event_id)))

    def list_audit_events(self, run_id: str) -> list[AuditEvent]:
        """Return a run's audit trail, oldest first."""
        with self.session() as session:
            query = select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.event_id)
            return list(session.scalars(query))

    def add_feedback(self, run_id: str, feedback: FeedbackIn, reviewer_id: str) -> None:
        """Record a verdict and its audit event in a single transaction (FR-06, FR-08)."""
        with self.session() as session:
            session.add(
                AnalystFeedback(
                    run_id=run_id,
                    target_type=feedback.target_type,
                    target_ref=feedback.target_ref,
                    verdict=feedback.verdict,
                    reviewer_id=reviewer_id,
                    comment=feedback.comment,
                )
            )
            session.add(
                AuditEvent(
                    run_id=run_id,
                    step="review",
                    payload={**feedback.model_dump(mode="json"), "reviewer_id": reviewer_id},
                )
            )

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        """Return a ticket, or None."""
        with self.session() as session:
            return session.get(Ticket, ticket_id)

    def get_tickets(self, ticket_ids: Iterable[str], clearance_rank: int) -> dict[str, Ticket]:
        """Fetch tickets by id, enforcing access again at the system of record."""
        with self.session() as session:
            query = select(Ticket).where(
                Ticket.ticket_id.in_(list(ticket_ids)), Ticket.access_rank <= clearance_rank
            )
            return {ticket.ticket_id: ticket for ticket in session.scalars(query)}

    def get_articles(self, article_ids: Iterable[str], clearance_rank: int) -> dict[str, KnowledgeArticle]:
        """Fetch approved articles by id, enforcing access at the system of record."""
        with self.session() as session:
            query = select(KnowledgeArticle).where(
                KnowledgeArticle.article_id.in_(list(article_ids)),
                KnowledgeArticle.approved.is_(True),
                KnowledgeArticle.access_rank <= clearance_rank,
            )
            return {article.article_id: article for article in session.scalars(query)}

    def search_tickets(
        self,
        clearance_rank: int,
        filters: dict[str, Any],
        text: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[Ticket]]:
        """FR-07: filter tickets the caller may see. Returns ``(total, page)``."""
        conditions = [Ticket.access_rank <= clearance_rank]
        conditions += [getattr(Ticket, field) == value for field, value in filters.items()]
        if text:
            conditions.append(Ticket.title.contains(text, autoescape=True))
        if created_from:
            conditions.append(Ticket.created_at >= created_from)
        if created_to:
            conditions.append(Ticket.created_at <= created_to)
        with self.session() as session:
            total = session.scalar(select(func.count()).select_from(Ticket).where(*conditions))
            query = (
                select(Ticket)
                .where(*conditions)
                .order_by(Ticket.created_at.desc(), Ticket.ticket_id)
                .limit(limit)
                .offset(offset)
            )
            return total or 0, list(session.scalars(query))

    def upsert_seed_ticket(self, values: dict[str, Any]) -> None:
        """Insert or replace a seeded ticket."""
        with self.session() as session:
            session.merge(Ticket(**values))

    def upsert_seed_article(self, values: dict[str, Any]) -> None:
        """Insert or replace a seeded knowledge article."""
        with self.session() as session:
            session.merge(KnowledgeArticle(**values))


class VectorStore:
    """The ChromaDB collections, all in cosine space.

    Chroma defaults to L2 distance, so the space is set explicitly. Embeddings are
    always computed by the caller: Chroma's own embedding function is disabled.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.chroma_host:
            self.client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.tickets = self._collection(TICKET_COLLECTION)
        self.knowledge = self._collection(KNOWLEDGE_COLLECTION)
        self.cache = self._collection(CACHE_COLLECTION)

    def _collection(self, name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def index_ticket(self, ticket: Ticket, document: str, embedding: list[float]) -> None:
        """Add or refresh a ticket. Metadata carries every field used for filtering."""
        self.tickets.upsert(
            ids=[ticket.ticket_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[
                {
                    "ticket_id": ticket.ticket_id,
                    "title": ticket.title,
                    "service": ticket.affected_service,
                    "environment": ticket.environment or "unspecified",
                    "category": ticket.category or "",
                    "subcategory": ticket.subcategory or "",
                    "status": ticket.status,
                    "is_open": ticket.status in OPEN_STATUSES,
                    "assignment_group": ticket.assignment_group or "",
                    "resolution_code": ticket.resolution_code or "",
                    "access_rank": ticket.access_rank,
                    "created_at": int(ticket.created_at.replace(tzinfo=timezone.utc).timestamp()),
                }
            ],
        )

    def index_article(self, article: KnowledgeArticle, document: str, embedding: list[float]) -> None:
        """Add or refresh an approved knowledge article."""
        self.knowledge.upsert(
            ids=[article.article_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[
                {
                    "ticket_id": article.article_id,
                    "title": article.title,
                    "service": article.affected_service or "",
                    "approved": article.approved,
                    "access_rank": article.access_rank,
                }
            ],
        )


def epoch_now() -> int:
    """Return the current time in seconds, as stored in cache metadata."""
    return int(time.time())
