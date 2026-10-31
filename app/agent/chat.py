"""Conversational troubleshooting, built on the same safeguards as the analysis pipeline.

Each turn is redacted, retrieved under the caller's clearance and citation-checked. A
reply that claims a fix is shown only if at least one citation is verified. Otherwise
the caller gets a fixed message and the related sources, so the chat fails towards a
human just as the pipeline does. The model has no tools and the server keeps no history.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from app.agent.workflow import Deps, assess_evidence, hydrate_sources, verify_citations
from app.cache_and_security import Principal, looks_like_injection, redact_text
from app.llm import PROMPT_VERSION, LLMError
from app.retrieval import Candidate
from app.schemas import (
    ChatKind,
    ChatMessage,
    ChatReply,
    CitationReport,
    MatchOut,
    ResolutionPath,
    SourceType,
)

logger = logging.getLogger(__name__)

# Follow-ups such as "that did not work" retrieve nothing alone, so recent turns are included.
QUERY_USER_MESSAGES = 3

INJECTION_REPLY = (
    "That message looks like an attempt to change how I work, so I can't act on it. "
    "Please describe the IT problem, or contact an analyst."
)
CONFLICT_REPLY = (
    "The sources I found recommend different fixes, so I won't pick one. "
    "Please contact an analyst, and share the related cases below with them."
)
UNVERIFIED_REPLY = (
    "I couldn't back up a fix with a verified source, so I won't suggest one. "
    "Please contact an analyst, and share the related cases below with them."
)
UNAVAILABLE_REPLY = (
    "The AI model is unavailable right now, so please try again shortly. "
    "If this is urgent, contact an analyst. Any related cases I found are listed below."
)


def retrieve_evidence(deps: Deps, query: str, clearance: int) -> list[dict[str, Any]]:
    """Search past fixes and approved knowledge, under the caller's clearance."""
    embedding = deps.embedder.embed(query)
    access = {"access_rank": {"$lte": clearance}}
    context = {"affected_service": "", "environment": None}

    def search(collection, source_type: SourceType, condition: dict[str, Any]) -> list[Candidate]:
        return deps.retriever.search(collection, source_type, query, embedding, [access, condition], context)

    resolved = search(deps.vectors.tickets, SourceType.TICKET, {"is_open": False})
    knowledge = search(deps.vectors.knowledge, SourceType.KNOWLEDGE, {"approved": True})
    ranked = sorted(resolved + knowledge, key=lambda item: item.score, reverse=True)
    return [candidate.to_state() for candidate in ranked[: deps.settings.retrieval_top_k]]


def _reply_from_model(
    deps: Deps, messages: list[dict[str, str]], supporting: list[dict[str, Any]], clearance: int
) -> tuple[ChatKind, str, list[Any], CitationReport]:
    """Call the model and keep its reply only if it passes the citation check."""
    sources = hydrate_sources(deps, supporting, clearance)
    try:
        output = deps.llm.chat(messages, sources)
    except LLMError as exc:
        logger.warning("Chat reply unavailable: %s", exc)
        return ChatKind.NO_ANSWER, UNAVAILABLE_REPLY, [], CitationReport()

    by_id = {source["source_id"]: source for source in sources}
    claims = ResolutionPath(summary=output.answer, steps=[], evidence=output.evidence)
    verified, report = verify_citations([claims], by_id, deps.settings.min_excerpt_chars)
    evidence = verified[0].evidence if verified else []
    if output.evidence_conflict:
        return ChatKind.NO_ANSWER, CONFLICT_REPLY, [], report
    if evidence:
        return ChatKind.ANSWER, output.answer, evidence, report
    if output.is_resolution or output.evidence:
        return ChatKind.NO_ANSWER, UNVERIFIED_REPLY, [], report
    return ChatKind.FOLLOW_UP, output.answer, [], report


def answer_chat(deps: Deps, messages: list[ChatMessage], principal: Principal) -> ChatReply:
    """Answer the last user message, and record the turn in the audit trail."""
    counts: Counter[str] = Counter()
    redacted = []
    for message in messages:
        content, found = redact_text(message.content)
        counts.update(found)
        redacted.append({"role": message.role, "content": content})

    injection = looks_like_injection(messages[-1].content)
    user_turns = [message["content"] for message in redacted if message["role"] == "user"]
    query = "\n".join(user_turns[-QUERY_USER_MESSAGES:])
    clearance = principal.clearance.rank
    evidence = retrieve_evidence(deps, query, clearance)
    supporting = [item for item in evidence if item["score"] >= deps.settings.min_evidence_score]

    report = CitationReport()
    verified: list[Any] = []
    if injection:
        kind, answer = ChatKind.NO_ANSWER, INJECTION_REPLY
    elif assess_evidence(evidence, deps.settings)["conflict"]:
        kind, answer = ChatKind.NO_ANSWER, CONFLICT_REPLY
    else:
        kind, answer, verified, report = _reply_from_model(deps, redacted, supporting, clearance)

    deps.repo.add_chat_event(
        requested_by=principal.name,
        clearance_rank=clearance,
        kind=kind,
        payload={
            "question": redacted[-1]["content"],
            "security": {"redactions": dict(counts), "injection_suspected": injection},
            "sources": [item["source_id"] for item in supporting],
            "citations": report.model_dump(),
            "model_version": deps.llm.version,
            "prompt_version": PROMPT_VERSION,
        },
    )
    return ChatReply(
        kind=kind,
        answer=answer,
        evidence=verified,
        related=[MatchOut(**item) for item in supporting],
        citations=report,
    )
