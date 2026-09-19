# Demo script (under 2 minutes)

A shot-by-shot walkthrough for the screen recording. Times are cumulative.

## Before you hit record

```bash
# 1. clean slate so History starts empty
rm policypilot.db          # Windows: del policypilot.db

# 2. terminal 1
.venv\Scripts\python.exe -m uvicorn src.api:app --reload

# 3. terminal 2
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Check before recording:
- `GEMINI_API_KEY` is set in `.env` and that model still has daily quota (20/day free tier — a failed demo looks like a broken app).
- Have one browser tab on `localhost:8501` and one on `localhost:8000/docs`.
- Pre-register `alice@demo.com` / `bob@demo.com` so you are not typing passwords on camera.

---

## 0:00 – 0:15 · What it is

> "PolicyPilot AI turns a customer support ticket into a policy-backed decision. A support agent submits a complaint, and the system retrieves the relevant company policy, asks Gemini to apply it, and returns a structured recommendation with its reasoning and sources."

Show the login screen.

## 0:15 – 0:30 · Auth

Log in as `alice@demo.com`.

> "Registration and login are JWT-based. Passwords are bcrypt-hashed — the frontend only ever holds a token, and it talks to FastAPI over HTTP. It has no database connection at all."

## 0:30 – 1:05 · The core flow

On **New Decision**, fill in:

| Field | Value |
|---|---|
| Message | `My order arrived damaged yesterday. The box was crushed.` |
| Order value | `3500` |
| Order status | `delivered` |
| Product type | `non_food` |
| Opened | `opened` |
| Days since delivery | `1` |

Submit, then read the result aloud:

> "It returns REQUEST_PHOTOS. And crucially, the reason cites the actual rule — the order is above the ₹2,000 threshold, so the damaged-goods policy requires photographs first. The source is `damaged_goods.md`. That's not the model guessing; that policy file was retrieved and put in front of it."

**This is the most important 30 seconds. Let the reason and sources stay on screen.**

## 1:05 – 1:20 · Missing information

Submit a second ticket with **only** a message:

> `I want to return this.`

> "With nothing else to go on, it returns NEEDS_MORE_INFORMATION rather than inventing an answer. That's a requirement of the brief — the system has to know when it can't decide."

## 1:20 – 1:35 · History

Open **History**, expand the first ticket.

> "Everything is persisted — ticket and decision — and each user sees only their own."

## 1:35 – 2:00 · Proof it works

Cut to a terminal:

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe evaluate.py
```

> "61 tests, all offline — Gemini is stubbed, so the suite runs without an API key. That includes the authorization test the brief asks for: Alice's token cannot fetch Bob's ticket. And the evaluation runner scores 5 out of 5 on the supplied cases."

---

## If asked follow-up questions

**"Why not just look up a similar past ticket?"**
`DATA_NOTES.md` forbids it. `tickets.csv` is used only to score the evaluation; the running system never reads it. Decisions come from the policy files.

**"Why RAG when the knowledge base is tiny?"**
It is ~1.5 KB and would fit in a prompt — the brief even hints at CAG. Retrieval is implemented because chunking, embedding and grounding are explicitly graded. The compromise: rank individual rules, then include every rule from the policies that matched. Policy rules are interdependent — one rule sets the ₹2,000 threshold, the next sets the 7-day cutoff — so retrieving one without the other produces a confidently wrong answer.

**"How do you know the AI output is safe to store?"**
Three layers. The Gemini call carries a response schema with the action enum, so the API constrains the shape. The reply is then re-validated with Pydantic — an invented action or an out-of-range confidence is a hard failure. And any cited source that was not in the retrieved context is dropped. If validation fails twice the request returns 503; nothing is persisted.

**"What's the accuracy?"**
5/5 on the supplied cases, but that is five clean single-policy scenarios. The harder cases are overlaps — damaged *food* engages two policies — and are not represented. The 214-ticket run needs 214 API calls and the free tier allows 20 a day, so it is not done. That's in the README as a known limitation.

**"What would you do next?"**
Finish the generalisation run, add overlap cases to the evaluation set, and cache decisions for repeated tickets.
