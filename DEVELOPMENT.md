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
| 4 | The first draft of the README stated the evaluation scored 5/5 at 100% accuracy — a number nothing had produced. | Caught on review before commit. | Replaced with the output *format*; real figures are only recorded after the runner is actually executed. |

Item 3 is the one worth dwelling on: the tests were *passing* in the sense of
not erroring for a while, while quietly depending on network access. It was only
visible because a fake API key produced a loud 400. The guard fixture now makes
that failure mode impossible rather than merely unlikely.

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
| Live API smoke test | `uvicorn src.api:app --port 8001` + curl | `/health` ok; register → `201`; login → JWT; `/me` ok; `/me` without token → `401`; `/tickets` → `[]` |
| Policy chunking | `python -m src.retrieval` | 29 chunks across 6 files |
| Evaluation | `python evaluate.py` | see below |

### Evaluation results

> Filled in after running `evaluate.py` against a live Gemini key.

---

## Things I would do next

- Cache decisions for identical ticket payloads; every submission currently costs
  two API calls.
- Add backoff/retry for Gemini rate limits — there is none today.
- Pagination on `GET /tickets`.
- A deterministic rules-engine baseline to run alongside the LLM, both as a
  regression check and to quantify how much the LLM is actually contributing.
