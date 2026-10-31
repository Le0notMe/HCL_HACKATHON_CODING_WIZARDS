"""FastAPI gateway.

Run with ``uvicorn app.main:app``. Blocking database and vector calls are moved to
the thread pool, so the event loop stays free. Analysis runs after the 202 response
in a FastAPI background task.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.agent.chat import answer_chat
from app.agent.workflow import (
    Deps,
    approve_for_cache,
    build_deps,
    run_analysis,
    update_cache_from_feedback,
)
from app.cache_and_security import Principal, authenticate, payload_fingerprint, prepare_ticket
from app.config import get_settings
from app.database import AnalysisRun, IdempotencyConflict
from app.schemas import (
    AnalysisResult,
    AnalyzeAccepted,
    ChatReply,
    ChatRequest,
    FeedbackIn,
    FeedbackOut,
    FeedbackTarget,
    RunOut,
    RunStatus,
    TicketIn,
    TicketOut,
    TicketSearchOut,
)

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
RECOMMENDATION_REF = "recommendation"


def get_deps(request: Request) -> Deps:
    """Return the dependencies created at startup."""
    return request.app.state.deps


def get_principal(
    api_key: Annotated[str | None, Security(api_key_header)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> Principal:
    """Authenticate the caller. With no keys configured, every request is rejected."""
    principal = authenticate(api_key, deps.settings.api_keys)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")
    return principal


DepsParam = Annotated[Deps, Depends(get_deps)]
PrincipalParam = Annotated[Principal, Depends(get_principal)]


def _visible_run(deps: Deps, run_id: str, principal: Principal) -> AnalysisRun:
    """Load a run the caller may read.

    Stored evidence was retrieved at the run's clearance, so reading it needs at least
    that clearance. A run the caller cannot read looks identical to a missing one.
    """
    run = deps.repo.get_run(run_id)
    if run is None or run.clearance_rank > principal.clearance.rank:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


def _run_response(run: AnalysisRun) -> RunOut:
    return RunOut(
        run_id=run.run_id,
        ticket_id=run.ticket_id,
        status=run.status,
        outcome=run.outcome,
        confidence=run.confidence,
        cache_hit=run.cache_hit,
        result=AnalysisResult.model_validate(run.result) if run.result else None,
        error=run.error,
        feedback=[FeedbackOut.model_validate(item, from_attributes=True) for item in run.feedback],
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


def _check_feedback_target(run: AnalysisRun, feedback: FeedbackIn) -> None:
    """Reject feedback that does not refer to something this run actually showed."""
    if run.status != RunStatus.COMPLETED or not run.result:
        raise HTTPException(status.HTTP_409_CONFLICT, "Run has not completed")
    result = AnalysisResult.model_validate(run.result)
    if feedback.target_type == FeedbackTarget.RECOMMENDATION:
        valid = feedback.target_ref == RECOMMENDATION_REF and bool(result.recommendation)
    else:
        matches = {match.source_id for match in result.duplicates + result.similar_cases}
        valid = feedback.target_ref in matches
    if not valid:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "target_ref is not part of this run")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build dependencies, unless a test injected them, and recover orphaned runs."""
    if getattr(app.state, "deps", None) is None:
        app.state.deps = await run_in_threadpool(build_deps, get_settings())
    deps: Deps = app.state.deps
    await run_in_threadpool(deps.repo.create_schema)
    orphaned = await run_in_threadpool(deps.repo.fail_stale_runs, deps.settings.stale_run_minutes)
    if orphaned:
        logger.warning("Marked %d interrupted runs as failed", orphaned)
    yield


def create_app(deps: Deps | None = None) -> FastAPI:
    """Create the application. Tests pass their own dependencies."""
    app = FastAPI(
        title="Intelligent IT Ticket Analysis and Resolution Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.deps = deps

    @app.exception_handler(OperationalError)
    async def database_unavailable(request: Request, exc: OperationalError) -> JSONResponse:
        logger.error("Database unavailable: %s", type(exc.orig).__name__)
        return JSONResponse({"detail": "Database unavailable"}, status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.get("/health", tags=["operations"])
    async def health(deps: DepsParam) -> dict[str, str]:
        """Report whether the database is reachable."""

        def ping() -> None:
            with deps.repo.session() as session:
                session.execute(text("SELECT 1"))

        await run_in_threadpool(ping)
        return {"status": "ok"}

    @app.post("/api/v1/tickets/analyze", status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
    async def analyze_ticket(
        ticket: TicketIn,
        background_tasks: BackgroundTasks,
        deps: DepsParam,
        principal: PrincipalParam,
    ) -> AnalyzeAccepted:
        """Accept a ticket for analysis (FR-01 to FR-05, FR-08).

        ``ticket_id`` is the idempotency key. Resubmitting identical content returns
        the existing run without starting another, reusing an id for different content
        returns 409, and resubmitting a failed run retries it.
        """
        if ticket.access_level.rank > principal.clearance.rank:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Ticket is above your clearance")

        prepared, report = prepare_ticket(ticket)
        try:
            run, created = await run_in_threadpool(
                deps.repo.create_run,
                prepared,
                payload_fingerprint(prepared),
                principal.name,
                principal.clearance.rank,
                deps.run_versions(),
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

        if created:
            await run_in_threadpool(
                deps.repo.append_audit_event,
                run.run_id,
                "input",
                {"ticket": {k: v for k, v in prepared.items()}, "security": report.to_dict()},
            )
            background_tasks.add_task(run_analysis, deps, run.run_id, prepared, run.clearance_rank, report)
        return AnalyzeAccepted(
            run_id=run.run_id,
            ticket_id=run.ticket_id,
            status=run.status,
            deduplicated=not created,
            status_url=f"/api/v1/runs/{run.run_id}",
        )

    @app.get("/api/v1/runs/{run_id}", tags=["analysis"])
    async def get_run(run_id: str, deps: DepsParam, principal: PrincipalParam) -> RunOut:
        """Return a run with its outcome, evidence and feedback (FR-04, FR-08)."""
        return _run_response(await run_in_threadpool(_visible_run, deps, run_id, principal))

    @app.post("/api/v1/runs/{run_id}/feedback", status_code=status.HTTP_201_CREATED, tags=["analysis"])
    async def submit_feedback(
        run_id: str, feedback: FeedbackIn, deps: DepsParam, principal: PrincipalParam
    ) -> RunOut:
        """Record an analyst verdict (FR-06).

        Approving a recommendation makes it eligible for the semantic cache, so this
        endpoint needs the ``can_review`` permission. An INCORRECT verdict removes the
        run from the cache.
        """
        if not principal.can_review:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewer permission required")

        def record() -> AnalysisRun:
            run = _visible_run(deps, run_id, principal)
            _check_feedback_target(run, feedback)
            update_cache_from_feedback(deps, run, feedback)
            deps.repo.add_feedback(run_id, feedback, principal.name)
            approve_for_cache(deps, run, feedback)
            return deps.repo.get_run(run_id)

        return _run_response(await run_in_threadpool(record))

    @app.post("/api/v1/chat", tags=["chat"])
    async def chat(request: ChatRequest, deps: DepsParam, principal: PrincipalParam) -> ChatReply:
        """Answer the last message of a conversation from the caller's visible sources.

        The client sends the whole conversation on each turn. A reply that claims a fix
        needs a verified citation.
        """
        return await run_in_threadpool(answer_chat, deps, request.messages, principal)

    @app.get("/api/v1/tickets/search", tags=["search"])
    async def search_tickets(
        deps: DepsParam,
        principal: PrincipalParam,
        q: Annotated[str | None, Query(max_length=200, description="Text to find in the title")] = None,
        application: Annotated[str | None, Query(max_length=128)] = None,
        category: Annotated[str | None, Query(max_length=64)] = None,
        ticket_status: Annotated[str | None, Query(alias="status", max_length=32)] = None,
        assignment_group: Annotated[str | None, Query(max_length=64)] = None,
        environment: Annotated[str | None, Query(max_length=32)] = None,
        priority: Annotated[str | None, Query(max_length=8)] = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> TicketSearchOut:
        """Search and filter the tickets visible to the caller (FR-07)."""
        supplied = {
            "affected_service": application,
            "category": category,
            "status": ticket_status,
            "assignment_group": assignment_group,
            "environment": environment.lower() if environment else None,
            "priority": priority,
        }
        filters = {field: value for field, value in supplied.items() if value is not None}
        total, tickets = await run_in_threadpool(
            deps.repo.search_tickets,
            principal.clearance.rank,
            filters,
            q,
            created_from,
            created_to,
            limit,
            offset,
        )
        return TicketSearchOut(
            total=total,
            limit=limit,
            offset=offset,
            items=[TicketOut.model_validate(ticket, from_attributes=True) for ticket in tickets],
        )

    return app


app = create_app()
