# PolicyPilot AI

**AI-Powered Support Ticket Decision Assistant** — turning customer complaints into policy-backed decisions.

A customer-support agent submits a ticket; the system retrieves the relevant company policy rules, asks Gemini to apply them, validates the structured answer, stores it, and shows the recommendation with its confidence, reasoning, and the exact policy files it relied on.

---

## Contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Running it](#running-it)
- [API reference](#api-reference)
- [Database schema](#database-schema)
- [Retrieval design](#retrieval-design)
- [Decision design](#decision-design)
- [Tests](#tests)
- [Evaluation](#evaluation)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)

---

## How it works

```
Streamlit  ──HTTP──▶  FastAPI  ──▶  retrieval (local vectors)  ──▶  policy rules
                         │                                              │
                         │◀─────────────────────────────────────────────┘
                         ▼
                   Gemini (JSON schema)
                         ▼
                 Pydantic validation
                         ▼
                   SQLite (tickets + decisions)
```

The frontend never touches the database and never makes a policy judgement of its own; everything goes through the API.

## Project layout

```
.
├── src/
│   ├── api.py          FastAPI app: auth + ticket endpoints
│   ├── auth.py         bcrypt hashing, JWT issue/verify, current-user dependency
│   ├── config.py       settings from .env
│   ├── database.py     SQLAlchemy engine, session, Base
│   ├── models.py       users / tickets / decisions tables
│   ├── schemas.py      Pydantic request + response models (the trust boundary)
│   ├── actions.py      the closed set of allowed decisions
│   ├── retrieval.py    load → chunk → embed → cache → cosine search
│   └── decision.py     prompt assembly, Gemini call, validation, grounding
├── streamlit_app.py    frontend (Login/Register, New Decision, History)
├── evaluate.py         accuracy runner over labelled cases
├── tests/              61 automated tests, no API key required
├── knowledge_base/     the 6 supplied policy documents
├── data/tickets.csv    214 historical tickets (used for evaluation only)
└── sample_test_cases.json
```

## Setup

Requires Python 3.11+ (developed on 3.14) and a Gemini API key from
<https://aistudio.google.com/u/0/api-keys>.

```bash
# 1. create and activate a virtual environment
uv venv                      # or: python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 2. install dependencies
uv pip install -r requirements.txt
# or: python -m pip install -r requirements.txt

# 3. configure
cp .env.example .env         # Windows: copy .env.example .env
```

Then edit `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | **yes** | — | Gemini API key |
| `JWT_SECRET` | **yes** | — | HMAC key for signing JWTs; **must be ≥ 32 characters** |
| `JWT_EXPIRE_MINUTES` | no | `60` | Token lifetime |
| `DATABASE_URL` | no | `sqlite:///policypilot.db` | SQLite location |
| `GEMINI_MODEL` | no | `gemini-3.5-flash` | Decision model |
| `EMBEDDING_MODEL` | no | `gemini-embedding-001` | Embedding model |
| `API_BASE_URL` | no | `http://127.0.0.1:8000` | Where Streamlit finds the API |

Generate a secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

`.env`, `*.db` and the generated embedding index are all gitignored.

## Running it

Two terminals, both with the virtualenv active.

```bash
# terminal 1 — backend (tables are created on startup)
python -m uvicorn src.api:app --reload

# terminal 2 — frontend
python -m streamlit run streamlit_app.py
```

Backend: <http://127.0.0.1:8000> · interactive docs: <http://127.0.0.1:8000/docs> · frontend: <http://localhost:8501>

The first ticket you submit builds the embedding index (one call, ~29 vectors) and caches it to `src/index/policy_index.npz`. Later runs load it from disk, and it rebuilds automatically if a policy file changes.

## API reference

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `POST` | `/register` | — | Create an account → `201` |
| `POST` | `/login` | — | Verify credentials → `{access_token, token_type}` |
| `GET` | `/me` | ✅ | The authenticated user |
| `POST` | `/tickets` | ✅ | Submit a ticket, run the pipeline, persist → `201` |
| `GET` | `/tickets` | ✅ | The caller's own tickets |
| `GET` | `/tickets/{id}` | ✅ | One ticket and its decision |
| `GET` | `/health` | — | Liveness check |

Protected endpoints use `Authorization: Bearer <JWT>`.

<details>
<summary>Example</summary>

```bash
curl -X POST localhost:8000/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"agent@example.com","password":"a-good-password"}'

TOKEN=$(curl -s -X POST localhost:8000/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"agent@example.com","password":"a-good-password"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST localhost:8000/tickets \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"My order arrived damaged yesterday.","order_value_inr":3500,
       "days_since_delivery":1,"product_type":"non_food",
       "opened_status":"opened","order_status":"delivered"}'
```

```json
{
  "id": 1,
  "message": "My order arrived damaged yesterday.",
  "decision": {
    "action": "REQUEST_PHOTOS",
    "confidence": 0.91,
    "reason": "The order value of ₹3,500 is above the ₹2,000 threshold, so photographs of the product and packaging must be requested before approving a refund or replacement.",
    "sources": ["damaged_goods.md"]
  }
}
```
</details>

### Status codes

| Code | When |
|---|---|
| `401` | Missing, malformed, expired or wrongly-signed token; bad login |
| `404` | Ticket does not exist **or** belongs to another user (deliberately indistinguishable) |
| `409` | Email already registered |
| `422` | Payload failed validation |
| `503` | Gemini unavailable or its output failed validation twice — nothing is persisted |

## Database schema

```
users                      tickets                        decisions
─────                      ───────                        ─────────
id           PK            id              PK             id           PK
email        UNIQUE   ┌──< user_id         FK        ┌──< ticket_id    FK UNIQUE
password_hash          │   message                    │    action
created_at             │   order_value_inr            │    reason
                       │   days_since_delivery        │    confidence
                       │   days_since_dispatch        │    sources      JSON
                       │   product_type               │    created_at
                       │   opened_status              │
                       │   order_status               │
                       │   created_at ─────────────────┘
                       └── (one user → many tickets; one ticket → one decision)
```

The six structured columns on `tickets` are nullable on purpose: a missing value is exactly what should drive a `NEEDS_MORE_INFORMATION` decision, so it has to be representable.

## Retrieval design

1. **Load** every `.md` in `knowledge_base/`.
2. **Chunk** on numbered rules — one chunk per rule, prefixed with the document heading. The policies are already written as short self-contained clauses, so this is a natural boundary; it also means a chunk is always a *complete* rule and `sources` can point at the exact file.
3. **Embed** each chunk with Gemini `gemini-embedding-001` (`task_type=RETRIEVAL_DOCUMENT`), L2-normalised to 3072 dimensions.
4. **Cache** the vectors to `src/index/policy_index.npz` with a SHA-256 fingerprint of the chunk texts; a mismatch rebuilds the index automatically.
5. **Query** — the ticket message plus its structured facts, embedded with `task_type=RETRIEVAL_QUERY`.
6. **Search** — cosine similarity is a plain dot product (both sides are unit vectors); take top-k.
7. **Expand to whole documents.** Once a policy file has a rule in the top-k, *all* of that file's rules go into the prompt.

Step 7 is the one non-obvious part and it fixes a real failure mode: policy rules are interdependent. `damaged_goods.md` rule 3 sets the ₹2,000 photo threshold while rule 4 carves out the 7-day cutoff. Retrieving rule 3 alone produces a confidently wrong answer. Ranking still does real work — it picks 1–2 policies out of 6 — but nothing gets truncated mid-policy.

No Pinecone, FAISS, Chroma, LangChain or LlamaIndex; the assignment explicitly says they are not needed, and at 29 chunks NumPy is the whole vector store.

## Decision design

- **Closed action set.** `Action` (in `src/actions.py`) is a `StrEnum`. The assignment only names `REQUEST_PHOTOS` and `NEEDS_MORE_INFORMATION`, so the full 15-value vocabulary was taken from the distinct `resolved_action` values in `data/tickets.csv` — the only place the complete list appears.
- **Native structured output.** The Gemini call passes an explicit `response_schema` (including the action enum) with `response_mime_type="application/json"`, so the shape is enforced by the API rather than by asking nicely. The wire schema is declared by hand rather than derived from `LLMDecision`, because that model's `extra="forbid"` renders as `additionalProperties: false`, which the Gemini API rejects. Keeping them separate lets the wire format stay API-compatible while validation stays strict.
- **Validation before persistence.** The reply is re-validated with Pydantic (`extra="forbid"`, `0 ≤ confidence ≤ 1`, action must be in the enum). Invalid output is retried once with a stricter instruction; a second failure raises and the request returns `503` rather than storing a guess.
- **Source grounding.** Citations not present in the retrieved context are dropped, so `sources` can only ever name policy files the model was actually shown.
- **`temperature=0`** so evaluation runs are reproducible.
- **Policies are the only authority.** `data/tickets.csv` is never read at request time — `DATA_NOTES.md` requires decisions to come from the policies, not from looking up a similar past ticket.

## Tests

```bash
python -m pytest tests/ -v
```

**61 tests, fully automated, no API key needed** — every Gemini call is stubbed, and an autouse fixture fails the test if anything tries to construct a real client.

| File | Covers |
|---|---|
| `test_auth.py` | bcrypt hashing + salting, hash never stored as plaintext, duplicate emails, payload validation, the 72-byte bcrypt limit, login failures, forged/expired/deleted-user tokens |
| `test_tickets.py` | creation, validation, history, detail, **ownership**, rollback when the pipeline fails |
| `test_retrieval.py` | all 6 files load, 29 chunks, thresholds survive chunking, cosine ranking, index round-trip and fingerprinting |
| `test_decision.py` | schema rejection of invented actions / out-of-range confidence / extra keys, whole-document expansion, hallucinated-citation dropping, retry-once-then-fail |

The authorization requirement called out in the assignment is `test_alice_cannot_read_bobs_ticket_by_id`, plus `test_unknown_and_forbidden_ids_are_indistinguishable` which asserts a forbidden id and a nonexistent id return byte-identical responses.

Only `evaluate.py` needs a live key.

## Evaluation

```bash
python evaluate.py                    # the 5 supplied sample cases
python evaluate.py --csv --limit 20   # historical tickets, generalisation check
python evaluate.py --rpm 60           # faster, if you have a paid key
```

Measured result on the supplied cases (`gemini-3.5-flash`, `temperature=0`):

```
==============================================================================
CASE     EXPECTED                       PREDICTED
------------------------------------------------------------------------------
S01      REQUEST_PHOTOS                 REQUEST_PHOTOS                 OK
S02      APPROVE_RETURN                 APPROVE_RETURN                 OK
S03      OPEN_SHIPPING_INVESTIGATION    OPEN_SHIPPING_INVESTIGATION    OK
S04      REPLACE_CORRECT_ITEM           REPLACE_CORRECT_ITEM           OK
S05      NEEDS_MORE_INFORMATION         NEEDS_MORE_INFORMATION         OK
==============================================================================
5 test cases
Correct: 5
Incorrect: 0
Accuracy: 100%
==============================================================================
```

Five cases is a small sample — this shows the pipeline is correct end to end, not that it is 100% accurate in general.

The runner paces itself to `--rpm` (default 5) because the Gemini free tier is tightly rate limited; see [Limitations](#limitations). Failures print the ticket, its facts, the model's confidence, reasoning and sources, so a wrong answer can be diagnosed without a re-run.

`--csv` scores against `data/tickets.csv`. This guards against overfitting to the five visible cases — `DATA_NOTES.md` calls them "the visible sample test cases", implying held-back ones. The CSV is a *scoring* input only; the running system never reads it.

## Design decisions

| Decision | Why |
|---|---|
| Gemini embeddings, not Sentence Transformers | Avoids a ~2 GB PyTorch dependency for 29 chunks, and reuses a dependency the project already has. Vectors are still stored and searched locally. |
| PyJWT, not python-jose | One maintained library, one job. |
| `bcrypt` directly, not `passlib` | passlib adds a layer and has known friction with bcrypt ≥ 4.1. |
| RAG, not CAG | The whole knowledge base is ~1.5 KB and *would* fit in a prompt. Retrieval is implemented because chunking/embedding/grounding is explicitly part of what the assignment evaluates; whole-document expansion keeps the accuracy benefit of CAG without abandoning retrieval. |
| `404` for another user's ticket | A `403` would confirm the ticket exists. |
| Nothing persisted on pipeline failure | A ticket row with no decision is a dead record, and inventing a decision is what the spec forbids. |

## Limitations

- **The Gemini free tier is the binding constraint.** Quotas are per model: ~5 requests/minute and **20 requests/day** for `gemini-3.5-flash`. Consequences:
  - `evaluate.py --csv` over all 214 historical tickets is **not feasible** on a free key. Use `--limit` (≤ 20/day), spread it over days, switch `GEMINI_MODEL` to another model with its own daily quota, or enable billing.
  - If you exhaust a model's daily quota, every request returns `429 RESOURCE_EXHAUSTED` until it resets — no amount of retrying helps. Switch `GEMINI_MODEL` in `.env`.
  - The client retries transient `503`/`429` up to 4 times, honouring the `retryDelay` the API returns, but it cannot retry past a daily cap.
- **Model availability shifts.** `text-embedding-004` and `gemini-2.0-flash` are both gone; `gemini-2.5-flash` is closed to new keys. If you hit a `404`, list what your key can actually use:
  ```python
  from src.retrieval import get_client
  print([m.name for m in get_client().models.list()])
  ```
- **Every ticket costs two Gemini calls** (one embedding, one decision), with no caching of repeated tickets.
- **The index is process-local.** `get_index()` is `lru_cache`d, so multiple uvicorn workers each hold their own copy. Harmless at this size.
- **No refresh tokens or logout revocation.** A JWT stays valid until it expires; logout only clears it client-side.
- **No pagination** on `GET /tickets`.
- **Accuracy depends on the model.** Retrieval is deterministic and `temperature=0`, but Gemini can still misapply a rule — particularly where two policies overlap (a damaged *food* item engages both `damaged_goods.md` and `returns.md`).
- **SQLite + `check_same_thread=False`** suits a single-process demo; concurrent writes would need a real database.
- **No HTTPS, CORS policy, or rate limiting** — out of scope per the assignment's "not a production system".
