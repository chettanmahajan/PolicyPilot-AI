# DEVELOPMENT.md

An honest account of how this project was built, including where AI coding tools
were used and where their output had to be corrected.

---

## AI coding tool usage

The project was built with **Claude Code (Opus 5)** acting as a pair programmer,
which the assignment explicitly encourages. The workflow was:

1. **Inspection before code.** The agent read the assignment PDF, all six policy
   documents, `DATA_NOTES.md`, `sample_test_cases.json` and `tickets.csv` before
   writing anything, and produced a written plan that I reviewed and approved.
2. **Staged implementation.** Config/DB → auth → ticket endpoints → retrieval →
   decision engine → frontend → evaluation → docs. Tests were run at each stage.
3. **Verification, not assertion.** Every "this works" claim in this repo is
   backed by a command that was actually executed — the test suite, a live
   `uvicorn` smoke test against real HTTP endpoints, and the evaluation runner.

### What the AI was genuinely useful for

- Scaffolding the FastAPI/SQLAlchemy/Pydantic boilerplate quickly.
- Writing broad test coverage (61 tests) faster than by hand, including edge
  cases I would plausibly have skipped — forged tokens, expired tokens, the
  72-byte bcrypt truncation limit, tokens for deleted users.
- Catching that `sample_test_cases.json` case **S04** (food product, wrong
  flavour) is governed by `wrong_item.md`, not by the food-return ban in
  `returns.md` — an overlap that is easy to misread.

### Where the AI got it wrong, and how it was caught

These are the corrections that actually happened during development. They are
recorded because "the agent wrote it" is not the same as "it was right".

| # | Problem | How it surfaced | Fix |
|---|---|---|---|
| 1 | The PDF text extractor had an off-by-one in its regex capture groups, so it returned 3 bytes of output instead of the assignment text. | Empty extraction output. | Traced group numbering by instrumenting the parser; `t[7]` was the operator token, not the text argument. |
| 2 | A test helper used `payload or DEFAULT`, so the "empty payload must be rejected" case silently submitted the *valid* default instead. Test failed for the right reason but the wrong cause. | `pytest` reported `201 == 422` failure. | Replaced with an explicit sentinel object. A passing version of this test would have been worse than a failing one — it would have claimed coverage it did not have. |
| 3 | `select_context()` called both `retrieve()` and `get_index()`. Mocking only `retrieve` left `get_index` live, so the "offline" test suite made a **real network call** to Gemini and failed with `API key not valid`. | Test failure with a live 400 from googleapis.com. | Collapsed to a single `get_index()` seam, and added an autouse fixture that raises if any test constructs a real `genai.Client`. |
| 4 | The first draft of the README stated the evaluation scored 5/5 at 100% accuracy — a number nothing had produced. | Caught on review before commit. | Replaced with the output *format*. The real figure was only written in after the runner actually ran — it did turn out to be 5/5, but that was luck, not knowledge. |
| 5 | Both default model names were dead. `text-embedding-004` → `404 NOT_FOUND`; `gemini-2.0-flash` was not on the key's model list at all; `gemini-2.5-flash` returned "no longer available to new users". | First live API call. | Queried `client.models.list()` for what the key can actually use, and pinned `gemini-embedding-001` + `gemini-3.5-flash`. This is the clearest example of a model's training data being stale about its own ecosystem — the fix was to ask the API, not to guess again. |
| 6 | Passing `response_schema=LLMDecision` failed with `400 INVALID_ARGUMENT: Unknown name "additional_properties"`. Pydantic renders `extra="forbid"` as `additionalProperties: false`, which Gemini rejects. | First successful auth'd decision call. | Declared the wire schema by hand and kept `LLMDecision` strict for validation. Dropping `extra="forbid"` would have been the easy fix and would have quietly removed a real guarantee. |
| 7 | `evaluate.py` crashed with `UnicodeEncodeError: '₹'` — the failure *report* died printing ₹ on a cp1252 Windows console, hiding the actual errors behind it. | Every case showed `ERROR` with a traceback from the reporting code. | Reconfigured stdout/stderr to UTF-8. Worth noting: the visible traceback was from the error handler, not the bug. |
| 8 | Free-tier rate limits caused cases to fail as `ERROR` and be scored as wrong answers. Initial backoff was 1s/2s while the API was explicitly asking for 31s. | Evaluation scored 60%, then 80%, with different cases failing each run — the tell-tale sign of an environmental problem, not a logic one. | Parse the `retryDelay` out of the 429 body and honour it; added a `RateLimiter` to the runner; dropped default workers 4 → 1. **These were never wrong decisions** — conflating "the API refused" with "the model was wrong" would have understated accuracy. |

Two of these are worth dwelling on.

**Item 3** — the tests were *passing* for a while whilst quietly depending on
network access. It only became visible because a fake API key produced a loud
400. The guard fixture now makes that failure mode impossible rather than
merely unlikely.

**Item 9** (below) was the subtlest. Running the evaluation with 4 worker
threads produced `RuntimeError: Cannot send a request, as the client has been
closed`. The cause: `functools.lru_cache` prevents *storing* more than one
result, but it does not prevent several threads *entering* the factory at once.
Four threads each built a `genai.Client`; three were discarded, and garbage
collecting them closed the HTTP transport underneath the one that had been
cached. The fix was a shared lock-guarded client plus warming the index before
the pool starts. A symptom-level fix — retrying the closed client, or dropping
back to one worker — would have left the race in place for the API server,
which is also multi-threaded.

| # | Problem | How it surfaced | Fix |
|---|---|---|---|
| 9 | `lru_cache` is not an initialisation lock; concurrent threads each constructed a Gemini client and the discarded ones closed the shared transport. | `evaluate.py --workers 4`. | `get_client()` / `get_index()` behind a `threading.RLock`; `decision.py` reuses the shared client instead of constructing one per request. |

### Where I overrode the AI's default suggestions

- **Sentence Transformers → Gemini embeddings.** The initial plan followed the
  common default of a local `sentence-transformers` model, which drags in ~2 GB
  of PyTorch. For 29 chunks of ~25 tokens that is absurd. The assignment requires
  embeddings to be *stored* locally, not *computed* locally.
- **No LangChain.** The assignment says it is not required, and a 30-line NumPy
  cosine search is easier to explain in an interview than a framework abstraction.
- **Whole-document expansion in retrieval.** Plain top-k is the default answer.
  It is wrong here, because a retrieved threshold rule can be separated from the
  exception that overrides it. This was a deliberate departure.
- **`passlib` → `bcrypt` directly**, and **`python-jose` → `PyJWT`**: fewer
  layers, both are a handful of lines.

---

## Verification log

Commands actually run, and what they produced.

| What | Command | Result |
|---|---|---|
| Full test suite | `python -m pytest tests/ -q` | **61 passed**, no network |
| Policy chunking | `python -m src.retrieval` | 29 chunks across 6 files |
| Live embedding | `embed(...)` over the whole KB | 29 vectors, 3072 dims |
| Live decision | `generate_decision(...)` on the S01 ticket | `REQUEST_PHOTOS`, cites `damaged_goods.md` |
| Evaluation | `python evaluate.py` | **5/5, 100%** (see below) |
| Live API end-to-end | `uvicorn src.api:app --port 8002` + curl | register → `201`; login → JWT; `/me` → `200`; `/me` unauthenticated → `401`; `POST /tickets` → real decision |
| Live authorization | Alice's ticket requested with Bob's token | Bob → `404 {"detail":"Ticket not found"}`; Alice → `200`; Bob's history → `[]` |

### Evaluation results

Run with `gemini-3.5-flash` at `temperature=0`:

```
S01  REQUEST_PHOTOS                 -> REQUEST_PHOTOS                 OK
S02  APPROVE_RETURN                 -> APPROVE_RETURN                 OK
S03  OPEN_SHIPPING_INVESTIGATION    -> OPEN_SHIPPING_INVESTIGATION    OK
S04  REPLACE_CORRECT_ITEM           -> REPLACE_CORRECT_ITEM           OK
S05  NEEDS_MORE_INFORMATION         -> NEEDS_MORE_INFORMATION         OK

5 test cases
Correct: 5
Incorrect: 0
Accuracy: 100%
```

Five cases is a small sample and every one of them is a clean, single-policy
scenario. This demonstrates the pipeline is correct end to end; it is not
evidence of 100% accuracy in general. `DATA_NOTES.md` calls these "the visible
sample test cases", so held-back cases should be assumed. The genuinely harder
situations are the overlaps — a damaged *food* item engages both
`damaged_goods.md` and `returns.md` — and they are not represented here.

A broader run against `data/tickets.csv` was **not** completed: the free tier
allows 20 requests per day per model, and 214 tickets needs 214. That is a real
gap in the evidence, not something to paper over.

### Live decision, verbatim

```json
{
  "action": "REQUEST_PHOTOS",
  "confidence": 1.0,
  "reason": "The order value of ₹3,500 is above the ₹2,000 threshold, so photographs of the damaged product and packaging must be requested under Damaged Goods Policy (rule 3).",
  "sources": ["damaged_goods.md"]
}
```

The reasoning names the threshold, the actual value, and the rule number — which
is what grounding is supposed to buy. Note the confidence of `1.0`: the model is
poorly calibrated at the top of the range, which is why the system treats the
action and its cited source as the decision, and confidence as display metadata
only.

---

## Things I would do next

- **Finish the generalisation run.** Scoring all 214 historical tickets is the
  main missing evidence; it needs a paid key or several days of free quota.
- **Add overlap cases to the evaluation set.** The five supplied cases are all
  single-policy. Damaged food, an opened item reported as defective, and a wrong
  item reported after the 7-day window would all be more informative.
- Cache decisions for identical ticket payloads; every submission currently costs
  two API calls.
- Pagination on `GET /tickets`.
- A deterministic rules-engine baseline to run alongside the LLM, both as a
  regression check and to quantify how much the LLM is actually contributing.
