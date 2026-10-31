# System Architecture & Technical Design: Intelligent IT Ticket Analysis and Resolution Assistant

| | |
|---|---|
| **Programme** | HCLTech Future Ready AI Engineer Hackathon |
| **Domain** | Enterprise IT / Service Management |
| **Document type** | Technical design |
| **Status** | The workflow revision in Section 5 was adopted and implemented, except for the R9 evaluation harness. Sections 2 and 3 record the original design, so where they differ, the [README](README.md) and the code describe the current system. |

## Contents

1. [Executive Summary & Objective](#1-executive-summary--objective)
2. [High-Level Architecture & Tech Stack](#2-high-level-architecture--tech-stack)
3. [End-to-End Workflow (Up to Semantic Similarity)](#3-end-to-end-workflow-up-to-semantic-similarity)
4. [Functional Requirements Matrix](#4-functional-requirements-matrix)
5. [Design Review: Recommended Workflow Revision](#5-design-review-recommended-workflow-revision)

---

## 1. Executive Summary & Objective

### 1.1 Overview

The Intelligent IT Ticket Analysis and Resolution Assistant is an agentic Retrieval-Augmented Generation (RAG) platform for enterprise IT support. It automates three tasks that slow analysts down and produce inconsistent outcomes when done by hand:

- **Ticket classification:** assigning category, subcategory, priority and intent from a defined taxonomy.
- **Duplicate detection:** finding related historical tickets by meaning rather than shared keywords.
- **Resolution synthesis:** drafting resolution steps grounded in how similar tickets were actually resolved.

### 1.2 Core Goal

Reduce ticket handling time, ensure evidence-backed outcomes and maintain strict traceability, **without performing automated production changes**. The system analyzes and recommends. A human analyst makes every decision.

### 1.3 Design Principles

| Principle | What it means in this system |
|---|---|
| **Evidence first** | No recommendation is shown without a source reference, an excerpt and a stated reason. |
| **Fail towards a human** | Weak evidence suppresses the recommendation and escalates the ticket. |
| **Read-only by design** | The system holds no credentials or tools that can change an ITSM or production system. |
| **Fully traceable** | Every prediction can be traced to its input, evidence, model version and reviewer decision. |

### 1.4 Scope

| In scope | Out of scope |
|---|---|
| Classification with confidence scores | Automatic ticket closure |
| Duplicate and similar-ticket retrieval | Privileged remediation |
| Evidence-backed resolution recommendations | Production ITSM changes |
| Human escalation, analyst feedback, search and filtering | |
| Audit logging for evaluation | |

### 1.5 Assumptions

- Seed data is synthetic or de-identified, and any generated labels are identified as such in the dataset.
- The LLM and embedding providers are configuration choices. This design does not depend on a specific vendor.
- All numeric thresholds in this document are starting values, to be tuned against a labelled validation set.

---

## 2. High-Level Architecture & Tech Stack

### 2.1 System Context

```text
 +------------------+                     +---------------------------+
 |  ITSM webhook /  |                     |   Analyst Review Console  |
 |  API client      |                     |   (Streamlit / Chainlit)  |
 +--------+---------+                     +-------------+-------------+
          |  ticket payload                             |  search, review,
          v                                             v  feedback
 +----------------------------------------------------------------------+
 |                     FastAPI Gateway + Pydantic                       |
 |      request validation | schema enforcement | response contracts    |
 +-----------------+-----------------------------------+----------------+
                   |                                   |
                   v                                   |
 +-----------------------------------+                 |
 |     Semantic Cache Check          |                 |
 |     cosine similarity > 0.95      |                 |
 +--------+-----------------+--------+                 |
     hit  |                 | miss                     |
          |                 v                          |
          |   +-------------------------------+        |
          |   |   LangGraph Orchestrator      |        |
          |   |   classify > retrieve >       |        |
          |   |   hydrate > synthesize >      |        |
          |   |   guardrail > audit           |        |
          |   +---+-----------------+---------+        |
          |       |                 |                  |
          v       v                 v                  v
 +-------------------------+   +--------------------------------------+
 |  ChromaDB               |   |  MySQL                               |
 |  - ticket_embeddings    |   |  - tickets (system of record)        |
 |  - semantic_cache       |   |  - analysis_runs                     |
 |                         |   |  - audit_events                      |
 |                         |   |  - analyst_feedback                  |
 +-------------------------+   +--------------------------------------+
                  |
                  v
          +---------------+
          |  LLM provider |  (called only from LangGraph nodes)
          +---------------+
```

### 2.2 Tech Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Backend API gateway | **FastAPI + Pydantic** | Strict request validation, schema enforcement and typed responses |
| Orchestration core | **LangGraph** | Multi-step agentic state machine with shared, inspectable state |
| Semantic and vector layer | **ChromaDB** | Dense vector embeddings, similarity search and high-threshold semantic caching |
| Relational storage and audit | **MySQL** | Master ticket records, relational filtering and audit logging |
| Frontend interface | **Streamlit or Chainlit** | Analyst review dashboards |

**Frontend note:** Streamlit is the better fit here. The analyst's work is a review screen with filters, tables, evidence panels and feedback controls, which is Streamlit's strength. Chainlit is built for chat interactions.

### 2.3 Data Ownership

MySQL is the system of record. ChromaDB holds a derived index that can be rebuilt from MySQL at any time, and the two stores are joined on `ticket_id`.

| ChromaDB collection | Contents | Metadata for filtering |
|---|---|---|
| `ticket_embeddings` | One embedding per historical ticket | `ticket_id`, `service`, `environment`, `category`, `status`, `assignment_group`, `created_at` |
| `semantic_cache` | Embeddings of tickets with pre-approved resolutions | `cache_id`, `run_id`, `approved_at` |

ChromaDB collections default to L2 distance, so both collections must be explicitly created in cosine space. Cosine similarity is then computed as `1 - distance`.

### 2.4 Core MySQL Schema

```sql
CREATE TABLE tickets (
    ticket_id        VARCHAR(32) PRIMARY KEY,
    title            VARCHAR(255) NOT NULL,
    description      TEXT NOT NULL,
    symptoms         TEXT,
    affected_service VARCHAR(128) NOT NULL,
    environment      VARCHAR(32),
    category         VARCHAR(64),
    subcategory      VARCHAR(64),
    priority         VARCHAR(8),
    status           VARCHAR(32),
    assignment_group VARCHAR(64),
    resolution_notes TEXT,
    created_at       DATETIME NOT NULL,
    resolved_at      DATETIME,
    INDEX idx_filter (affected_service, environment, status)
);

CREATE TABLE analysis_runs (
    run_id         CHAR(36) PRIMARY KEY,
    ticket_id      VARCHAR(32) NOT NULL,
    cache_hit      BOOLEAN NOT NULL,
    outcome        ENUM('RECOMMENDED', 'ESCALATED') NOT NULL,
    confidence     DECIMAL(4, 3),
    model_version  VARCHAR(64),
    prompt_version VARCHAR(32),
    created_at     DATETIME NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
);

CREATE TABLE audit_events (
    event_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id     CHAR(36) NOT NULL,
    step       VARCHAR(32) NOT NULL,
    payload    JSON NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES analysis_runs (run_id)
);

CREATE TABLE analyst_feedback (
    feedback_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id      CHAR(36) NOT NULL,
    target_type ENUM('MATCH', 'RECOMMENDATION') NOT NULL,
    target_ref  VARCHAR(64) NOT NULL,
    verdict     ENUM('USEFUL', 'INCORRECT', 'UNRESOLVED') NOT NULL,
    reviewer_id VARCHAR(64) NOT NULL,
    comment     TEXT,
    created_at  DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES analysis_runs (run_id)
);
```

`audit_events` is append-only. The application never updates or deletes its rows.

### 2.5 API Surface

| Method | Endpoint | Purpose | Requirements |
|---|---|---|---|
| `POST` | `/api/v1/tickets/analyze` | Run the pipeline on an incoming ticket | FR-01 to FR-05, FR-08 |
| `GET` | `/api/v1/runs/{run_id}` | Fetch a run with its evidence and outcome | FR-04, FR-08 |
| `POST` | `/api/v1/runs/{run_id}/feedback` | Record an analyst verdict | FR-06, FR-08 |
| `GET` | `/api/v1/tickets/search` | Search and filter tickets | FR-07 |

---

## 3. End-to-End Workflow (Up to Semantic Similarity)

### 3.1 Pipeline Overview

```text
           ITSM webhook / API payload
                      |
                      v
       [1] Ingestion & Validation (Pydantic)
                      |
                      v
       [2] Semantic Cache Check -- similarity > 0.95 --+
                      |                                |
                 cache miss                        cache hit
                      |                                |
   . . . . . . . . . .|. . . . . . . . . . . . .       |
   .  LangGraph       v                        .       |
   .   [3] Classification (FR-01)              .       |
   .                  |                        .       |
   .                  v                        .       |
   .   [4] Vector Similarity Retrieval (FR-02) .       |
   .                  |                        .       |
   .                  v                        .       |
   .   [5] Hydration & Synthesis (FR-03/04)    .       |
   .                  |                        .       |
   .                  v                        .       |
   .   [6] Guardrail Evaluation (FR-05)        .       |
   .        |                     |            .       |
   .   conf >= 0.70          conf < 0.70       .       |
   .   strong evidence       or weak evidence  .       |
   .        |                     |            .       |
   .        v                     v            .       |
   .   Recommendation      Human escalation    .       |
   .        |                     |            .       |
   . . . . .|. . . . . . . . . . .|. . . . . . .       |
            |                     |                    |
            v                     v                    v
       [7] Audit Logging to MySQL (FR-08) <------------+
                      |
                      v
           Analyst Review Console
```

### 3.2 Step 1: Ingestion & Validation

An incoming ITSM webhook or API payload hits the FastAPI endpoint and is validated by Pydantic. Malformed requests are rejected with HTTP 422 before any processing takes place.

```python
from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=3)
    symptoms: str | None = None
    affected_service: str = Field(min_length=1, max_length=128)
    environment: str | None = None
```

### 3.3 Step 2: Semantic Cache Check

The ticket's text is embedded and queried against the `semantic_cache` collection with a strict threshold of **cosine similarity > 0.95**.

| Result | Behaviour |
|---|---|
| **Cache hit** | Bypasses the LLM pipeline and returns the pre-approved resolution immediately, reducing latency and cost. |
| **Cache miss** | Proceeds to full LangGraph orchestration. |

A resolution counts as pre-approved only after an analyst has marked it useful (FR-06). The cache is therefore filled by human review and never directly by model output.

The ticket embedding is computed once at this step and carried in the graph state, so Step 4 can reuse it.

### 3.4 LangGraph State

Steps 3 to 7 run as a state machine over a single shared state object.

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict, total=False):
    run_id: str
    ticket: dict
    embedding: list[float]
    classification: dict
    similar_tickets: list[dict]
    hydrated_cases: list[dict]
    recommendation: dict
    confidence: float
    outcome: str


builder = StateGraph(TicketState)
builder.add_node("classify", classify_ticket)
builder.add_node("retrieve", retrieve_similar)
builder.add_node("hydrate_and_synthesize", synthesize_resolution)
builder.add_node("guardrail", evaluate_guardrail)
builder.add_node("audit", write_audit_log)

builder.add_edge(START, "classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "hydrate_and_synthesize")
builder.add_edge("hydrate_and_synthesize", "guardrail")
builder.add_edge("guardrail", "audit")
builder.add_edge("audit", END)

graph = builder.compile()
```

### 3.5 Step 3: Automated Classification (FR-01)

An LLM node categorizes the ticket against the defined taxonomy and returns structured output. It computes a confidence score through logprobs where the provider supports them, or otherwise through prompt self-evaluation.

```json
{
  "category": "Access Management",
  "subcategory": "VPN Authentication",
  "priority": "P2",
  "intent": "restore_access",
  "confidence": 0.91
}
```

Predicted values are validated against the taxonomy, and any value outside it is rejected.

### 3.6 Step 4: Vector Duplicate & Similarity Retrieval (FR-02)

The text embedding is queried against `ticket_embeddings` to retrieve ranked historical cases. Metadata filters on `service` and `environment` restrict the search to operationally relevant tickets, which prevents superficial keyword matches from a different system.

```python
ticket = state["ticket"]
where = {"service": ticket["affected_service"]}
if ticket.get("environment"):
    where = {"$and": [where, {"environment": ticket["environment"]}]}

results = ticket_collection.query(
    query_embeddings=[state["embedding"]],
    n_results=10,
    where=where,
)
```

The text embedded for each ticket combines title, description, symptoms and affected service, as FR-02 requires. Each result carries its `ticket_id` and similarity score into the graph state.

### 3.7 Step 5: Data Hydration & Resolution Synthesis (FR-03, FR-04)

The orchestrator fetches the full resolution notes from MySQL using the retrieved `ticket_id` values. It then prompts the LLM to synthesize actionable steps strictly grounded in the retrieved history, with explicit source excerpts and reasoning.

```json
{
  "resolution_paths": [
    {
      "summary": "Reset the expired VPN certificate profile",
      "steps": [
        "Confirm the certificate expiry date in the VPN portal",
        "Re-issue the user's certificate profile",
        "Ask the user to reconnect and confirm access"
      ],
      "evidence": [
        {
          "source_ticket_id": "INC-10482",
          "excerpt": "Certificate profile had expired; re-issuing it restored access.",
          "reason": "Same service, same error code and the same symptom pattern."
        }
      ]
    }
  ]
}
```

The prompt instructs the model to use only the supplied cases, and to say so when they do not support a resolution.

### 3.8 Step 6: Guardrail Evaluation & Fallback (FR-05)

The system evaluates aggregated confidence metrics. The aggregate takes the lowest signal, so one weak result cannot be hidden by two strong ones.

```python
CONFIDENCE_THRESHOLD = 0.70
MIN_SIMILARITY = 0.60
MIN_SUPPORTING_CASES = 2


def evaluate_guardrail(state: TicketState) -> dict:
    similar = state.get("similar_tickets", [])
    supporting = [t for t in similar if t["similarity"] >= MIN_SIMILARITY]
    retrieval_conf = max((t["similarity"] for t in similar), default=0.0)
    confidence = min(state["classification"]["confidence"], retrieval_conf)

    weak_evidence = len(supporting) < MIN_SUPPORTING_CASES
    if confidence < CONFIDENCE_THRESHOLD or weak_evidence:
        return {
            "confidence": confidence,
            "recommendation": None,
            "outcome": "ESCALATED",
        }
    return {"confidence": confidence, "outcome": "RECOMMENDED"}
```

| Condition | Outcome |
|---|---|
| Confidence ≥ 0.70 and evidence is strong | Recommendation is shown to the analyst with its evidence |
| Confidence < 0.70 or evidence is weak | Automated recommendation is suppressed and the ticket is flagged for human escalation |

### 3.9 Step 7: Audit Logging (FR-08)

Every step is logged into MySQL to provide a fully compliant audit trail, including cache hits.

| Audited item | Stored in |
|---|---|
| Input ticket | `tickets`, `audit_events` (`step = input`) |
| Prediction | `audit_events` (`step = classification`) |
| Retrieved evidence | `audit_events` (`step = retrieval`) |
| Recommendation | `audit_events` (`step = recommendation`) |
| Reviewer decision | `analyst_feedback` |

---

## 4. Functional Requirements Matrix

The status column shows coverage by the workflow in Section 3. Items marked **Partial** are closed by the revision in Section 5.

| ID | Capability | Technical module | Stack | Status |
|---|---|---|---|---|
| FR-01 | Ticket classification | `classify` node, taxonomy validation | LangGraph, LLM, Pydantic | Covered |
| FR-02 | Duplicate and similarity detection | `retrieve` node, `ticket_embeddings` collection | LangGraph, ChromaDB | Partial (R3) |
| FR-03 | Resolution recommendation | `hydrate_and_synthesize` node | LangGraph, MySQL, LLM | Partial (R3) |
| FR-04 | Evidence and traceability | Evidence schema in synthesis output, evidence panel | Pydantic, LLM, Streamlit | Partial (R5) |
| FR-05 | Confidence and fallback | `guardrail` node, escalation queue | LangGraph, Streamlit | Partial (R4) |
| FR-06 | Analyst feedback | `POST /runs/{run_id}/feedback`, review controls | FastAPI, MySQL, Streamlit | Partial (R7) |
| FR-07 | Search and filtering | `GET /tickets/search`, metadata filters | FastAPI, MySQL, ChromaDB | Covered |
| FR-08 | Audit-friendly output | `audit` node, `audit_events`, `analyst_feedback` | LangGraph, MySQL | Partial (R7) |

---

## 5. Design Review: Recommended Workflow Revision

The workflow in Section 3 demonstrates the core flow, but a review against the problem brief found gaps in cache safety, data protection, retrieval coverage and the required evaluation evidence. This section proposes a revised pipeline. If it is adopted, it replaces Section 3.

### 5.1 Revised Pipeline

```text
 OFFLINE   Seed tickets + approved KB > redact > embed > index in ChromaDB
 ---------------------------------------------------------------------------
 ONLINE    (every node appends to the audit log as it completes)

 [1] Ingest & validate       202 Accepted, idempotent on ticket_id
          |
          v
 [2] Redact & scope          mask PII/secrets, resolve the caller's access
          |
          v
 [3] Scoped semantic cache   same service + environment + access level
          |          \
          | miss      \ hit: return the stored analysis and its evidence,
          v            \     then go to [9]
 [4] Retrieve & rerank       open tickets  > duplicate candidates
          |                  closed tickets + KB > resolution evidence
          v
 [5] Classify                LLM + neighbour labels as a second opinion
          |
          v
 [6] Evidence gate --------- weak evidence: skip the LLM, go to [8]
          |
          v
 [7] Synthesize + verify     every excerpt must exist in its cited source
          |
          v
 [8] Route                   RECOMMEND | CLARIFY | ESCALATE
          |
          v
 [9] Analyst review          useful / incorrect / unresolved
          |
          +---> approved results enter the cache; verdicts feed evaluation
```

### 5.2 Recommended Changes

| # | Change | Problem in the Section 3 workflow | Brief reference |
|---|---|---|---|
| **R1** | Add a **redaction and access-scoping step** before the first embedding. Filter on access level during retrieval. Treat ticket text as untrusted data in every prompt. | Raw ticket text, which may contain PII or passwords, reaches the embedding model, the vector store and the LLM. Any retrieved ticket can be shown to any user. A crafted ticket could also inject instructions into the prompt. | Scope: sensitive content and access restrictions |
| **R2** | **Scope the cache** by service, environment and access level, and store the **full analysis** (classification, evidence, reason). Expire entries, and invalidate them when an analyst marks the source incorrect or when the embedding model changes. | "VPN down in prod" and "VPN down in dev" can exceed 0.95 similarity and still need different fixes. A hit also skips classification and evidence, so it breaks FR-01 and FR-04 for that ticket. A stale fix would be served indefinitely. | FR-01, FR-04, FR-05 |
| **R3** | **Retrieve before classifying**. Use hybrid search (vector + BM25) with a cross-encoder reranker. Search open tickets for duplicates, and resolved tickets plus **approved knowledge** for resolutions. Use filters only for fields the ticket states explicitly. | Duplicates and similar tickets are different things: a duplicate is the same live issue, and a similar ticket is a precedent. Knowledge articles are missing altogether. Strict filters discard valid matches when a user names the wrong service, and vector search alone can still return superficial matches. | FR-02, FR-03, Evaluation: duplicate retrieval |
| **R4** | Use **two gates and three outcomes**. An evidence gate before synthesis avoids a wasted LLM call. The final route is recommend, **clarify** or escalate, and contradictory sources force a low-confidence result. Escalated tickets still show their similar cases. | Weak evidence is detected only after paying for synthesis. There is no clarification path and no contradiction check, and both are in the evaluation guidance. Suppressing everything also hides useful matches from the analyst. | FR-05, Evaluation: confidence and fallback |
| **R5** | Add **deterministic citation verification**: each excerpt must appear in its cited source, or that claim is removed. | The LLM can invent an excerpt or a ticket ID, and nothing would catch it. Prompt instructions alone do not guarantee grounding. | FR-04, Evaluation: traceability |
| **R6** | Use **neighbour-label agreement** as a second confidence signal, and calibrate thresholds on validation data. | LLM self-evaluation is often overconfident, and logprobs are not available from every provider. A fixed 0.70 means little until it is measured against real outcomes. | FR-01, FR-05 |
| **R7** | Write **audit events from each node as it completes**, recording model, prompt and threshold versions. Make **analyst review an explicit final step** whose verdicts fill the cache. | A single audit step at the end loses the trail if a run fails midway. The reviewer decision is required by FR-08 but arrives after the pipeline ends, so it needs its own step. | FR-06, FR-08 |
| **R8** | Return **202 Accepted** with a `run_id`, process in the background and treat `ticket_id` as an **idempotency key**. Add an **offline indexing pipeline**. | LLM calls can outlast webhook timeouts, and ITSM webhooks retry, which creates duplicate runs. The workflow also never says how ChromaDB gets populated. | Prototype reliability |
| **R9** | Add an **evaluation harness** over a labelled dataset (see 5.3). | The brief requires an evaluation report and three worked examples. The workflow does not produce them. | Expected submission, Evaluation |

### 5.3 Evaluation Plan

| Dimension | Metrics |
|---|---|
| Classification quality | Macro-F1 per field, plus a confusion matrix highlighting operationally important classes such as P1 vs P2 |
| Duplicate retrieval quality | Recall@5, MRR and nDCG@5, including a test set of keyword-similar but unrelated tickets |
| Resolution usefulness | Analyst ratings for relevance, actionability and evidence support |
| Confidence and fallback | Share of weak or contradictory test tickets correctly routed to clarify or escalate |
| Traceability | Share of claims with a verified citation (target: 100%) |
| Semantic cache | Precision of cache hits, since a wrong hit is the costliest failure in the pipeline |

### 5.4 Delivery Priority

| Priority | Changes | Rationale |
|---|---|---|
| **Must have** | R1, R2, R4, R5, R7, R9, plus the offline indexing in R8 | Directly required by the brief, prevents wrong answers from reaching analysts, or is needed for retrieval to work at all |
| **Should have** | R3, R6 | Improves retrieval and confidence quality. Separating duplicates from precedents and adding knowledge articles matters most. The reranker can follow. |
| **If time permits** | The rest of R8 | The background task and idempotency check are small. A full queue is unnecessary for a prototype. |
