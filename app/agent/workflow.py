"""The revised analysis pipeline, as a LangGraph state machine.

::

    cache_check --hit--------------------------------------> finalize
        | miss
        v
    retrieve -> classify -> evidence_gate --weak--> route -> finalize
                                 | strong             ^
                                 v                    |
                             synthesize --------------+

Every node appends its own audit event as it completes (R7), so a run that fails
midway still leaves a trail up to the failing step. The graph has no tools and no
write access to any ITSM system: it only analyzes and recommends.

A model failure does not fail the run. It is recorded and the ticket escalates to
a human, in keeping with the "fail towards a human" principle.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.cache_and_security import RedactionReport, SemanticCache, ticket_document
from app.config import Settings
from app.database import AnalysisRun, Repository, VectorStore
from app.embeddings import Embedder, build_embedder
from app.llm import PROMPT_VERSION, LLMClient, LLMError, build_llm
from app.retrieval import Candidate, HybridRetriever, build_reranker
from app.schemas import (
    INTENTS,
    PRIORITIES,
    TAXONOMY,
    AnalysisResult,
    CitationReport,
    Classification,
    ClassificationOutput,
    FeedbackIn,
    FeedbackTarget,
    MatchOut,
    Outcome,
    ResolutionPath,
    RunStatus,
    SourceType,
    Verdict,
    VerifiedEvidence,
    VerifiedPath,
)

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "What is the exact error message or code?",
    "When did the problem start, and did anything change beforehand?",
    "How many users are affected, and in which environment?",
]


class AnalysisState(TypedDict, total=False):
    """Shared graph state. All values are plain data, so they can be audited."""

    run_id: str
    ticket: dict[str, Any]
    clearance: int
    security: dict[str, Any]
    embedding: list[float]
    cached_result: dict[str, Any] | None
    cache_source_run_id: str | None
    duplicates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    classification: dict[str, Any] | None
    gate: dict[str, Any]
    sources: list[dict[str, Any]]
    verified_paths: list[dict[str, Any]]
    citations: dict[str, Any]
    evidence_conflict: bool
    clarifying_questions: list[str]
    errors: list[str]
    outcome: str
    confidence: float
    confidence_breakdown: dict[str, float]
    reasons: list[str]


@dataclass
class Deps:
    """Everything the workflow needs, injected so tests can substitute parts."""

    settings: Settings
    repo: Repository
    vectors: VectorStore
    embedder: Embedder
    llm: LLMClient
    retriever: HybridRetriever
    cache: SemanticCache
    graph: CompiledStateGraph | None = None

    def run_versions(self) -> dict[str, Any]:
        """Return what a run must record to be reproducible (R7)."""
        return {
            "model_version": self.llm.version,
            "prompt_version": PROMPT_VERSION,
            "embedding_version": self.embedder.version,
            "thresholds": self.settings.threshold_snapshot(),
        }


def build_deps(settings: Settings) -> Deps:
    """Construct the production dependency set."""
    repo = Repository(settings.database_url)
    vectors = VectorStore(settings)
    embedder = build_embedder(settings)
    deps = Deps(
        settings=settings,
        repo=repo,
        vectors=vectors,
        embedder=embedder,
        llm=build_llm(settings),
        retriever=HybridRetriever(settings, build_reranker(settings)),
        cache=SemanticCache(vectors, settings, embedder.version),
    )
    deps.graph = build_graph(deps)
    return deps


# Pure decision logic, kept free of I/O so it can be tested directly.


def normalise_for_match(text: str) -> str:
    """Normalise only what cannot change meaning: unicode form, whitespace and case."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def verify_citations(
    paths: list[ResolutionPath],
    sources: dict[str, dict[str, Any]],
    min_excerpt_chars: int,
) -> tuple[list[VerifiedPath], CitationReport]:
    """R5: keep only evidence whose excerpt really occurs in its cited source.

    This check is deterministic and does not trust the model. An unknown source id,
    a too-short excerpt or text not found in the source removes that claim, and a
    path left with no verified evidence is removed entirely.
    """
    report = CitationReport()
    verified_paths = []
    for path in paths:
        verified: list[VerifiedEvidence] = []
        for claim in path.evidence:
            report.claims_total += 1
            source = sources.get(claim.source_id)
            excerpt = normalise_for_match(claim.excerpt)
            if source is None:
                failure = "unknown_source"
            elif len(excerpt) < min_excerpt_chars:
                failure = "excerpt_too_short"
            elif excerpt not in normalise_for_match(source["text"]):
                failure = "excerpt_not_in_source"
            else:
                report.claims_verified += 1
                verified.append(
                    VerifiedEvidence(
                        source_id=claim.source_id,
                        source_type=source["source_type"],
                        excerpt=claim.excerpt,
                        reason=claim.reason,
                    )
                )
                continue
            report.rejected.append({"source_id": claim.source_id, "failure": failure})
        if verified:
            verified_paths.append(VerifiedPath(summary=path.summary, steps=path.steps, evidence=verified))
    return verified_paths, report


def neighbour_agreement(category: str, evidence: list[dict[str, Any]]) -> float | None:
    """R6: the score-weighted share of labelled neighbours in the same category.

    The classifier never sees these labels, which keeps this an independent second
    opinion. Returns None when no neighbour is labelled.
    """
    labelled = [item for item in evidence if item.get("category")]
    total = sum(item["score"] for item in labelled)
    if not labelled or total <= 0:
        return None
    return sum(item["score"] for item in labelled if item["category"] == category) / total


def blend_confidence(llm_confidence: float, agreement: float | None, settings: Settings) -> float:
    """Combine model confidence with neighbour agreement, or discount it if there is none."""
    if agreement is None:
        return llm_confidence * settings.no_neighbour_penalty
    weight = settings.llm_confidence_weight
    return weight * llm_confidence + (1.0 - weight) * agreement


def taxonomy_violation(output: ClassificationOutput) -> str | None:
    """Return a description of the first value outside the taxonomy, if any."""
    if output.category not in TAXONOMY:
        return "category"
    if output.subcategory not in TAXONOMY[output.category]:
        return "subcategory"
    if output.priority not in PRIORITIES:
        return "priority"
    if output.intent not in INTENTS:
        return "intent"
    return None


def assess_evidence(evidence: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """R4: decide whether the evidence justifies calling the LLM.

    Supporting sources are those above the score floor. Their resolution codes, where
    the data has them, reveal contradictions without a model: if no single fix accounts
    for most of the sources, they disagree.
    """
    supporting = [item for item in evidence if item["score"] >= settings.min_evidence_score]
    codes = Counter(item["resolution_code"] for item in supporting if item.get("resolution_code"))
    labelled = sum(codes.values())
    conflict = len(codes) > 1 and codes.most_common(1)[0][1] / labelled < settings.conflict_dominance
    strong = len(supporting) >= settings.min_supporting_sources
    return {
        "strong": strong,
        "conflict": conflict,
        "passed": strong and not conflict,
        "top_score": evidence[0]["score"] if evidence else 0.0,
        "supporting_sources": len(supporting),
        "resolution_codes": dict(codes),
    }


def is_underspecified(ticket: dict[str, Any], settings: Settings) -> bool:
    """Return True if the reporter gave too little detail to diagnose the ticket."""
    return not ticket.get("symptoms") and len(ticket["description"].split()) < settings.underspecified_word_count


def decide_route(state: AnalysisState, settings: Settings) -> dict[str, Any]:
    """R4: choose RECOMMEND, CLARIFY or ESCALATE, and the overall confidence.

    Confidence is the lowest of the classification, retrieval and citation signals, so
    one weak signal cannot be hidden by two strong ones. Rules run from most to least
    serious, and a recommendation must pass all of them.
    """
    gate = state["gate"]
    classification = state.get("classification")
    citations = state.get("citations") or {}
    breakdown = {
        "classification": classification["confidence"] if classification else 0.0,
        "retrieval": gate["top_score"],
    }
    if citations.get("claims_total"):
        breakdown["grounding"] = citations["claims_verified"] / citations["claims_total"]
    confidence = min(breakdown.values())
    conflict = gate["conflict"] or state.get("evidence_conflict", False)
    if conflict:
        confidence = min(confidence, settings.contradiction_confidence_cap)
    questions = state.get("clarifying_questions") or []
    errors = state.get("errors") or []

    def decision(outcome: Outcome, *reasons: str) -> dict[str, Any]:
        return {
            "outcome": outcome.value,
            "confidence": round(confidence, 4),
            "confidence_breakdown": {key: round(value, 4) for key, value in breakdown.items()},
            "reasons": list(reasons),
            "clarifying_questions": questions if outcome == Outcome.CLARIFY else [],
        }

    if state["security"]["injection_suspected"]:
        return decision(Outcome.ESCALATE, "Ticket text resembles an attempt to instruct the model")
    if errors:
        return decision(Outcome.ESCALATE, *errors)
    if conflict:
        return decision(Outcome.ESCALATE, "Sources recommend incompatible resolutions")
    if not gate["strong"]:
        if is_underspecified(state["ticket"], settings):
            questions = questions or DEFAULT_QUESTIONS
            return decision(Outcome.CLARIFY, "Too little detail and too little matching evidence")
        return decision(Outcome.ESCALATE, "Too little matching evidence for a recommendation")
    if not state.get("verified_paths"):
        if questions:
            return decision(Outcome.CLARIFY, "The sources need more ticket detail to apply")
        return decision(Outcome.ESCALATE, "No resolution survived citation verification")
    if confidence < settings.confidence_threshold:
        return decision(
            Outcome.ESCALATE,
            f"Confidence {confidence:.2f} is below the threshold of {settings.confidence_threshold:.2f}",
        )
    return decision(Outcome.RECOMMEND, "All confidence and evidence checks passed")


def hydrate_sources(deps: Deps, evidence: list[dict[str, Any]], clearance: int) -> list[dict[str, Any]]:
    """Fetch the full text of the evidence from MySQL, the system of record."""
    ticket_ids = [e["source_id"] for e in evidence if e["source_type"] == SourceType.TICKET.value]
    article_ids = [e["source_id"] for e in evidence if e["source_type"] == SourceType.KNOWLEDGE.value]
    tickets = deps.repo.get_tickets(ticket_ids, clearance)
    articles = deps.repo.get_articles(article_ids, clearance)
    sources = []
    for item in evidence:
        source_id = item["source_id"]
        ticket = tickets.get(source_id)
        article = articles.get(source_id)
        if ticket is not None and ticket.resolution_notes:
            parts = (ticket.title, ticket.description, ticket.symptoms, ticket.resolution_notes)
            text = "\n".join(part for part in parts if part)
            resolution = ticket.resolution_notes
        elif article is not None:
            text = f"{article.title}\n{article.content}"
            resolution = article.content
        else:
            continue
        sources.append(
            {
                "source_id": source_id,
                "source_type": item["source_type"],
                "score": item["score"],
                "title": item["title"],
                "text": text,
                "resolution": resolution,
            }
        )
    return sources


# Graph nodes.


def _audited(deps: Deps, step: str, node: Callable[[AnalysisState], dict[str, Any]]):
    """Wrap a node so it appends an audit event on completion or failure (R7)."""

    def wrapper(state: AnalysisState) -> dict[str, Any]:
        try:
            update = node(state)
        except Exception as exc:
            deps.repo.append_audit_event(state["run_id"], f"{step}.error", {"error": type(exc).__name__})
            raise
        payload = {key: value for key, value in update.items() if key not in ("embedding", "sources")}
        deps.repo.append_audit_event(state["run_id"], step, payload)
        return update

    return wrapper


def build_graph(deps: Deps) -> CompiledStateGraph:
    """Compile the workflow against a set of dependencies."""
    settings = deps.settings

    def cache_check(state: AnalysisState) -> dict[str, Any]:
        ticket = state["ticket"]
        embedding = deps.embedder.embed(ticket_document(ticket))
        miss = {"embedding": embedding, "cached_result": None, "cache_source_run_id": None}
        if state["security"]["injection_suspected"]:
            return miss
        hit = deps.cache.lookup(
            embedding, ticket["affected_service"], ticket.get("environment"), state["clearance"]
        )
        if hit is None:
            return miss
        source = deps.repo.get_run(hit.source_run_id)
        if source is None or source.outcome != Outcome.RECOMMEND or not source.result:
            deps.cache.invalidate(hit.source_run_id)
            return miss
        return {
            "embedding": embedding,
            "cached_result": source.result,
            "cache_source_run_id": hit.source_run_id,
            "reasons": [f"Matched approved analysis at similarity {hit.similarity}"],
        }

    def retrieve(state: AnalysisState) -> dict[str, Any]:
        ticket = state["ticket"]
        document = ticket_document(ticket)
        access = {"access_rank": {"$lte": state["clearance"]}}
        not_self = {"ticket_id": {"$ne": ticket["ticket_id"]}}

        def search(collection, source_type: SourceType, *conditions: dict[str, Any]) -> list[Candidate]:
            return deps.retriever.search(
                collection, source_type, document, state["embedding"], [access, *conditions], ticket
            )

        open_tickets = search(deps.vectors.tickets, SourceType.TICKET, {"is_open": True}, not_self)
        resolved = search(deps.vectors.tickets, SourceType.TICKET, {"is_open": False}, not_self)
        knowledge = search(deps.vectors.knowledge, SourceType.KNOWLEDGE, {"approved": True})
        evidence = sorted(resolved + knowledge, key=lambda item: item.score, reverse=True)
        return {
            "duplicates": [c.to_state() for c in open_tickets if c.score >= settings.duplicate_min_score],
            "evidence": [c.to_state() for c in evidence[: settings.retrieval_top_k]],
        }

    def classify(state: AnalysisState) -> dict[str, Any]:
        try:
            output = deps.llm.classify(state["ticket"])
        except LLMError as exc:
            return {"classification": None, "errors": [f"Classification unavailable: {exc}"]}
        violation = taxonomy_violation(output)
        if violation:
            return {"classification": None, "errors": [f"Classification had an invalid {violation}"]}
        llm_confidence = min(1.0, max(0.0, output.confidence))
        agreement = neighbour_agreement(output.category, state.get("evidence", []))
        classification = Classification(
            category=output.category,
            subcategory=output.subcategory,
            priority=output.priority,
            intent=output.intent,
            confidence=round(blend_confidence(llm_confidence, agreement, settings), 4),
            llm_confidence=llm_confidence,
            neighbour_agreement=None if agreement is None else round(agreement, 4),
            rationale=output.rationale,
        )
        return {"classification": classification.model_dump(), "errors": []}

    def evidence_gate(state: AnalysisState) -> dict[str, Any]:
        return {"gate": assess_evidence(state.get("evidence", []), settings)}

    def synthesize(state: AnalysisState) -> dict[str, Any]:
        sources = hydrate_sources(deps, state["evidence"], state["clearance"])
        try:
            output = deps.llm.synthesize(state["ticket"], sources)
        except LLMError as exc:
            return {"sources": sources, "errors": [*state.get("errors", []), f"Synthesis unavailable: {exc}"]}
        by_id = {source["source_id"]: source for source in sources}
        verified, report = verify_citations(output.resolution_paths, by_id, settings.min_excerpt_chars)
        return {
            "sources": sources,
            "verified_paths": [path.model_dump(mode="json") for path in verified],
            "citations": report.model_dump(),
            "evidence_conflict": output.evidence_conflict,
            "clarifying_questions": output.clarifying_questions,
        }

    def route(state: AnalysisState) -> dict[str, Any]:
        return decide_route(state, settings)

    def finalize(state: AnalysisState) -> dict[str, Any]:
        source_run_id = state.get("cache_source_run_id")
        if source_run_id:
            result = AnalysisResult.model_validate(state["cached_result"]).model_copy(
                update={
                    "cache_hit": True,
                    "cache_source_run_id": source_run_id,
                    "duplicates": [],
                    "reasons": state["reasons"],
                }
            )
        else:
            outcome = Outcome(state["outcome"])
            classification = state.get("classification")
            result = AnalysisResult(
                outcome=outcome,
                confidence=state["confidence"],
                confidence_breakdown=state["confidence_breakdown"],
                reasons=state["reasons"],
                classification=Classification(**classification) if classification else None,
                duplicates=[MatchOut(**item) for item in state.get("duplicates", [])],
                similar_cases=[MatchOut(**item) for item in state.get("evidence", [])],
                recommendation=state.get("verified_paths", []) if outcome == Outcome.RECOMMEND else [],
                clarifying_questions=state.get("clarifying_questions", []),
                citations=CitationReport(**(state.get("citations") or {})),
            )
        deps.repo.complete_run(
            state["run_id"], result.outcome, result.confidence, result.model_dump(mode="json"), source_run_id
        )
        _index_open_ticket(deps, state)
        return {"outcome": result.outcome.value, "confidence": result.confidence, "cache_hit": result.cache_hit}

    graph = StateGraph(AnalysisState)
    nodes = {
        "cache_check": cache_check,
        "retrieve": retrieve,
        "classify": classify,
        "evidence_gate": evidence_gate,
        "synthesize": synthesize,
        "route": route,
        "finalize": finalize,
    }
    for name, node in nodes.items():
        graph.add_node(name, _audited(deps, name, node))

    graph.add_edge(START, "cache_check")
    graph.add_conditional_edges(
        "cache_check",
        lambda state: "hit" if state.get("cache_source_run_id") else "miss",
        {"hit": "finalize", "miss": "retrieve"},
    )
    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        lambda state: "pass" if state["gate"]["passed"] and not state.get("errors") else "stop",
        {"pass": "synthesize", "stop": "route"},
    )
    graph.add_edge("synthesize", "route")
    graph.add_edge("route", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _index_open_ticket(deps: Deps, state: AnalysisState) -> None:
    """Make an analyzed open ticket findable as a duplicate of later tickets."""
    row = deps.repo.get_ticket(state["ticket"]["ticket_id"])
    if row is not None and row.status == "OPEN":
        deps.vectors.index_ticket(row, ticket_document(state["ticket"]), state["embedding"])


def run_analysis(
    deps: Deps,
    run_id: str,
    ticket: dict[str, Any],
    clearance: int,
    report: RedactionReport,
) -> None:
    """Background entry point for one run. Never raises.

    The run is claimed atomically first, so a retried webhook cannot start a second
    worker. Unexpected errors mark the run FAILED, which allows a resubmission to
    retry it. The stored error holds only the exception type, never ticket text.
    """
    if not deps.repo.mark_running(run_id):
        logger.info("Run %s was already claimed", run_id)
        return
    try:
        deps.graph.invoke(
            {"run_id": run_id, "ticket": ticket, "clearance": clearance, "security": report.to_dict()}
        )
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        deps.repo.fail_run(run_id, f"Internal error: {type(exc).__name__}")


def update_cache_from_feedback(deps: Deps, run: AnalysisRun, feedback: FeedbackIn) -> None:
    """Keep the cache consistent with analyst review (R2, R7).

    Any INCORRECT verdict removes the run's entry and the entry it was served from.
    Removal happens before the feedback is stored, so a failure here cannot leave a
    rejected answer being served. Approval is handled by :func:`approve_for_cache`.
    """
    if feedback.verdict == Verdict.INCORRECT:
        deps.cache.invalidate(run.run_id)
        if run.cache_source_run_id:
            deps.cache.invalidate(run.cache_source_run_id)


def approve_for_cache(deps: Deps, run: AnalysisRun, feedback: FeedbackIn) -> None:
    """Add a run to the cache if an analyst approved its recommendation.

    Cache hits are not re-added, so one approval cannot extend its own expiry, and a
    run with any INCORRECT verdict is never added. A failure here only costs a future
    cache hit, so it is logged rather than raised.
    """
    approved = (
        feedback.target_type == FeedbackTarget.RECOMMENDATION
        and feedback.verdict == Verdict.USEFUL
        and run.status == RunStatus.COMPLETED
        and run.outcome == Outcome.RECOMMEND
        and not run.cache_hit
        and not any(item.verdict == Verdict.INCORRECT for item in run.feedback)
    )
    if not approved:
        return
    try:
        row = deps.repo.get_ticket(run.ticket_id)
        ticket = {
            "title": row.title,
            "description": row.description,
            "symptoms": row.symptoms,
            "affected_service": row.affected_service,
        }
        embedding = deps.embedder.embed(ticket_document(ticket))
        deps.cache.store(run.run_id, embedding, row.affected_service, row.environment, run.clearance_rank)
    except Exception:
        logger.exception("Could not add run %s to the cache", run.run_id)
