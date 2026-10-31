# Intelligent IT Ticket Analysis and Resolution Assistant

An AI assistant for IT support teams, built for the HCLTech Future Ready AI Engineer Hackathon. It classifies incoming tickets, finds duplicates and past fixes, recommends resolutions backed by checked evidence, and answers troubleshooting questions in a chat.

The assistant only analyzes and recommends. It has no tools or credentials that can change an ITSM or production system, so a person always makes the final decision.

## Features

- **Chat:** describe a problem, ask follow-up questions, and get fixes that cite past tickets and approved knowledge.
- **Ticket analysis:** each ticket gets a category, priority, possible duplicates, similar cases and an outcome of `RECOMMEND`, `CLARIFY` or `ESCALATE`.
- **Hybrid search:** vector and BM25 keyword search are combined and then reranked, so both meaning and exact error codes count.
- **Citation checking:** every quote from the model must appear in the source it cites. A fix that fails this check is never shown.
- **Failing towards a person:** weak evidence, conflicting sources or a model outage leads to escalation, with the related cases still shown.
- **Privacy and access:** passwords, tokens, emails and similar values are masked before storage or any model call. Each API key only sees records at or below its clearance.
- **Learning from analysts:** an approved recommendation can answer near-identical tickets from a cache. Marking it incorrect removes it.
- **Audit trail:** each pipeline step and chat turn is recorded in append-only tables.
- **Model resilience:** Gemini is the default, with automatic fallback when a model is overloaded. Claude and an offline demo mode are also available.

## How it works

Tickets go through a LangGraph pipeline:

```text
POST /tickets/analyze   validate > authorize > redact > idempotency check > 202
                                                                             |
                                                       background task       v
cache_check ---- hit ------------------------------------------------> finalize
    | miss                                                                   ^
    v                                                                        |
retrieve > classify > evidence_gate -- weak or conflicting --> route --------+
                           | strong                              ^
                           v                                     |
                       synthesize + verify citations ------------+

POST /runs/{id}/feedback   a USEFUL verdict adds to the cache, INCORRECT removes
```

Chat uses the same safeguards on every turn:

```text
POST /chat   authorize > redact > check for injection > retrieve within clearance
                                                                 |
                                                                 v
                         audit < verify citations < Gemini, with no tools
```

| Chat reply | Meaning |
|---|---|
| `ANSWER` | A reply backed by at least one verified citation. |
| `FOLLOW_UP` | A greeting or a question back, which claims no fix. |
| `NO_ANSWER` | No reply could be trusted, so the user is pointed to an analyst and the related cases. |

## Tech stack

| Area | Technology |
|---|---|
| API | FastAPI, Pydantic |
| Workflow | LangGraph |
| Models | Google Gemini by default; Claude and an offline mode are optional |
| Vector store | ChromaDB |
| Database | MySQL, through SQLAlchemy |
| Retrieval | ONNX `all-MiniLM-L6-v2` embeddings, BM25, ONNX cross-encoder reranker |
| Frontend | Streamlit |
| Tests | pytest |

## Project structure

```text
app/
  main.py                FastAPI gateway and endpoints
  schemas.py             Pydantic contracts, taxonomy, structured LLM output
  config.py              Settings and thresholds
  cache_and_security.py  Redaction, API keys, scoped semantic cache
  database.py            MySQL models and repository, ChromaDB collections
  retrieval.py           Hybrid search: vector + BM25, reranking
  embeddings.py          Local embedding providers
  llm.py                 LLM interface; Gemini, Claude and offline adapters
  agent/workflow.py      LangGraph pipeline, citation checks, routing
  agent/chat.py          Grounded chat, using the same safeguards
frontend/
  streamlit_app.py       Analyst console
  api_client.py          HTTP client for the API
scripts/seed_index.py    Offline indexing
data/                    Synthetic seed tickets and knowledge articles
tests/                   Unit and end-to-end tests
IT_Ticket_Assistant_Technical_Design.md   Original design and workflow review
```

## Getting started

You need Python 3.12, Docker and [uv](https://docs.astral.sh/uv/). The commands below are for Windows PowerShell.

> **Do not build the virtual environment from Anaconda's Python on Windows.** Its older C++ runtime DLLs crash `chromadb` and `onnxruntime` with an access violation (exit code 139). The `--python-preference only-managed` flag below uses a standalone Python instead.

```powershell
# 1. Create the environment
uv venv --python-preference only-managed --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\activate

# 2. Start MySQL on port 3307
docker compose up -d --wait

# 3. Create your settings file
copy .env.example .env
```

Edit `.env`:

- Replace the placeholder keys in `API_KEYS` with long random values.
- Set `GEMINI_API_KEY` to a key from [Google AI Studio](https://aistudio.google.com/apikey), or set `LLM_PROVIDER=offline` to try the pipeline without one.

```powershell
# 4. Load the sample data. The first run downloads the embedding model.
python -m scripts.seed_index

# 5. Start the API. The first run downloads the reranker.
uvicorn app.main:app --port 8000
```

In a second terminal, start the console:

```powershell
.venv\Scripts\activate
streamlit run frontend/streamlit_app.py
```

Open <http://localhost:8501> and paste your analyst key from `API_KEYS` into the sidebar. The interactive API docs are at <http://127.0.0.1:8000/docs>.

To skip pasting the key or to use another API address, set `TICKET_API_KEY` and `API_BASE_URL` before starting the console. These are read from the shell, not from `.env`.

## Using the console

| Page | What it does |
|---|---|
| **Chat** | Talk through a problem. Each reply lists its verified sources and related cases. |
| **Analyze a ticket** | Submit a ticket or a sample, read the outcome and evidence, and record feedback. |
| **Look up a run** | Reopen an earlier result by run ID. |
| **Search tickets** | Filter by text, application, category, status, priority, environment and date. |

The console only calls the API, so authentication, redaction and clearance are all enforced there.

## API

Every request except `/health` needs an `X-API-Key` header with a key from `API_KEYS`.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Answer the last message of a conversation. |
| `POST` | `/api/v1/tickets/analyze` | Accept a ticket. Returns `202` with a `run_id` and `status_url`. |
| `GET` | `/api/v1/runs/{run_id}` | Poll for the outcome, evidence and feedback. |
| `POST` | `/api/v1/runs/{run_id}/feedback` | Record a verdict. Needs `can_review`. |
| `GET` | `/api/v1/tickets/search` | Filter by `q`, `application`, `category`, `status`, `assignment_group`, `environment`, `priority` and dates. |
| `GET` | `/health` | Check database connectivity. |

Analyze a ticket:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/tickets/analyze `
  -H "X-API-Key: replace-with-analyst-key" -H "Content-Type: application/json" `
  -d '{\"ticket_id\":\"INC-70001\",\"title\":\"GlobalProtect VPN fails with GP-4012\",\"description\":\"Laptop cannot reach the VPN. GlobalProtect shows GP-4012, client certificate expired.\",\"symptoms\":\"GP-4012 on connect\",\"affected_service\":\"VPN\",\"environment\":\"prod\"}'
```

Chat. The client sends the whole conversation on each turn, ending with a user message:

```json
{"messages": [{"role": "user", "content": "VPN shows GP-4012, my certificate expired"}]}
```

Approve a recommendation, which makes it eligible for the cache:

```json
{"target_type": "RECOMMENDATION", "target_ref": "recommendation", "verdict": "USEFUL"}
```

To rate a single match, set `target_type` to `MATCH` and `target_ref` to its source id. `ticket_id` is an idempotency key: identical content returns the existing run, different content returns `409`, and resubmitting a failed run retries it.

## Tests

```powershell
python -m pytest tests
```

The suite needs no API key or MySQL. It runs the real graph and ChromaDB, with SQLite, a hashing embedder and a scripted LLM. The frontend tests drive the real Streamlit page against the API over HTTP. The Gemini and Claude adapters run through their real SDKs against a local stub server.

## Design decisions

| Decision | Reason |
|---|---|
| Redaction runs in the request path | Raw text never reaches MySQL, ChromaDB, the audit log or the LLM. The trade-off is that the stored input is the redacted version. |
| The classifier never sees neighbour labels | Showing them would make neighbour agreement circular. Kept apart, it is an independent second opinion. |
| Overall confidence is the lowest signal | One weak signal cannot be averaged away by two strong ones. |
| LLM failures escalate instead of failing the run | An analyst still gets the retrieved cases. |
| Chat uses the API's safeguards, not a direct model call | Every turn is redacted, retrieved under the caller's clearance and audited in `chat_events`. A reply claiming a fix is shown only with a verified citation. |
| The server keeps no chat history | `chat_events` stores only each redacted question and metadata. The trade-off is that a conversation is lost when the page is closed. |
| Service and environment adjust scores instead of filtering | Reporters often name the wrong service, and a hard filter would hide the right match. |
| A cache hit needs an exact clearance match | A hit may contain evidence retrieved at that clearance. |
| A cached result drops its duplicates list | Duplicates are about current open tickets, so stored ones would be stale. |
| Analysis uses a FastAPI background task | It needs no extra infrastructure. If the process dies, the run is marked failed at the next startup after `STALE_RUN_MINUTES`, and resubmitting retries it. |
| Gemini is the default LLM, behind an interface | `LLMClient` in `app/llm.py` is the only code a new provider has to implement. Gemini safety blocks escalate to a human. |
| `.env.example` pins a model and lists fallbacks | The newest Flash models, which `gemini-flash-latest` follows, can return `503` under high demand. On a `503` or `429`, the next model in `LLM_FALLBACK_MODELS` is tried. A run records the configured model, and the fallback used appears only in the API log. |

## Known limitations

- **No evaluation report yet.** The R9 evaluation harness from the design is not built, so accuracy has not been measured.
- **Thresholds are uncalibrated.** They are starting values, and need tuning on labelled data. The reranker saturates near 1.0 on close matches, so it separates good from bad matches better than it ranks good ones.
- **Seed data is synthetic** and only 13 tickets, so results here do not predict accuracy on real tickets.
- **Redaction is pattern-based.** It catches emails, phone numbers, IP addresses, cards, SSNs, tokens and passwords. It does not catch names or free-form personal details, which need an NER model or a DLP service. Add one before using real data.
- **The free Gemini tier is for synthetic data only.** Google's terms let it use content sent through unpaid services to improve its products. Use a billing-enabled project for real tickets.
- **Injection detection is a heuristic.** The main defenses are that ticket text is encoded as data and the model has no tools. The pattern check is an extra signal.
- **Chat citations are checked, but the wording around them is not.** A verified quote shows that the source exists and says that text, not that every sentence of the reply follows from it.
- **BM25 scans per query.** That keeps it behind the same access filter as vector search, but it is capped at `BM25_CORPUS_LIMIT` documents and needs a dedicated index at scale.
- **Conflict detection needs `resolution_code`.** Without codes, it relies on the model reporting a conflict.
- **Cache invalidation is per run.** Marking a run incorrect removes that run's entry and its source entry, not other entries citing the same evidence. The TTL bounds those.
- **API keys are plain configuration values.** This suits a prototype. Production needs SSO for analysts and a secret manager.
- **One uvicorn worker with embedded ChromaDB.** For more workers, run a Chroma server and set `CHROMA_HOST`.
