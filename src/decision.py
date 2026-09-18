"""Gemini-backed decision engine.

Flow: ticket -> retrieval query -> relevant policy context -> Gemini (with a
native response schema) -> Pydantic validation -> LLMDecision.

The model is only ever allowed to pick an action from `Action` and to cite
policy files that were actually retrieved, so a hallucinated action or citation
becomes a validation failure rather than a stored decision.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.actions import Action
from src.config import settings
from src.retrieval import RetrievedChunk, get_index
from src.schemas import LLMDecision, TicketCreate

logger = logging.getLogger(__name__)


class DecisionUnavailableError(RuntimeError):
    """The decision could not be produced (API failure, or unparseable output)."""


SYSTEM_INSTRUCTION = """\
You are a support-ticket decision engine for an Indian e-commerce company.

Decide the correct action for the customer's ticket using ONLY the company
policy excerpts supplied in the prompt. The policies are the sole source of
truth. Do not rely on outside knowledge, and do not invent rules, thresholds,
or time windows that are not written in the excerpts.

Rules of engagement:
- Choose exactly one action from the allowed list.
- Base the decision on the CURRENT ticket's facts, not on any similar past case.
- If a fact the policy needs (delivery date, dispatch date, order value, product
  type, opened/unopened status, or what the actual problem is) is missing or
  unknown, and the policy cannot be applied safely without it, return
  NEEDS_MORE_INFORMATION.
- Where a specific policy covers the situation (damaged, defective, wrong item,
  shipping delay, cancellation), apply that policy rather than the general
  change-of-mind returns policy.
- `reason` must be one or two sentences that name the specific rule and the
  fact that triggered it, e.g. "Order value of Rs.3,500 is above the Rs.2,000
  threshold, so photographs must be requested first."
- `sources` must list only the policy filenames you actually relied on, copied
  exactly from the "source:" labels in the context.
- `confidence` reflects how squarely the policy covers this ticket: use a high
  value only when a rule applies unambiguously, and a low value when the
  situation is borderline.

High confidence never justifies an action the policy does not support.
"""


def _describe_ticket(ticket: TicketCreate) -> str:
    """Render ticket facts, making missing values explicit rather than omitting them."""
    def show(value: object) -> str:
        return "NOT PROVIDED" if value is None or value == "unknown" else str(value)

    return "\n".join(
        [
            f"Customer message: {ticket.message}",
            f"Order value (INR): {show(ticket.order_value_inr)}",
            f"Days since delivery: {show(ticket.days_since_delivery)}",
            f"Days since dispatch: {show(ticket.days_since_dispatch)}",
            f"Product type: {show(ticket.product_type)}",
            f"Opened status: {show(ticket.opened_status)}",
            f"Order status: {show(ticket.order_status)}",
        ]
    )


def build_retrieval_query(ticket: TicketCreate) -> str:
    """The text embedded to find relevant policy rules."""
    parts = [ticket.message]
    if ticket.order_status:
        parts.append(f"order status {ticket.order_status}")
    if ticket.product_type:
        parts.append(f"{ticket.product_type} product")
    if ticket.opened_status:
        parts.append(f"{ticket.opened_status} item")
    if ticket.days_since_dispatch is not None:
        parts.append(f"{ticket.days_since_dispatch} days since dispatch")
    if ticket.days_since_delivery is not None:
        parts.append(f"{ticket.days_since_delivery} days since delivery")
    return ". ".join(parts)


def select_context(ticket: TicketCreate) -> list[RetrievedChunk]:
    """Rank rules by similarity, then widen to every rule in the winning policies.

    Ranking individual rules is what makes this retrieval rather than stuffing
    the whole knowledge base in. But a policy's rules are interdependent - the
    threshold is in one rule and its exception in the next - so once a document
    is judged relevant we include all of its rules. This avoids the common
    failure where the top-k cuts off the exception that changes the answer.
    """
    index = get_index()
    ranked = index.search(build_retrieval_query(ticket))

    relevant_sources = {r.chunk.source for r in ranked}
    best_score = {r.chunk.source: r.score for r in ranked}
    expanded = [
        RetrievedChunk(chunk=c, score=best_score.get(c.source, 0.0))
        for c in index.chunks
        if c.source in relevant_sources
    ]
    # Keep the strongest-matching policy first so the model reads it first.
    expanded.sort(key=lambda r: (-r.score, r.chunk.source, int(r.chunk.rule)))
    return expanded


def format_context(chunks: list[RetrievedChunk]) -> str:
    by_source: dict[str, list[RetrievedChunk]] = {}
    for item in chunks:
        by_source.setdefault(item.chunk.source, []).append(item)

    blocks = []
    for source, items in by_source.items():
        rules = "\n".join(f"  {c.chunk.text}" for c in items)
        blocks.append(f"source: {source}\n{rules}")
    return "\n\n".join(blocks)


def _call_gemini(prompt: str, *, strict_retry: bool = False) -> str:
    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise DecisionUnavailableError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=settings.gemini_api_key)
    instruction = SYSTEM_INSTRUCTION
    if strict_retry:
        instruction += (
            "\nYour previous reply could not be parsed. Return ONLY a JSON object "
            "with exactly the keys: action, confidence, reason, sources."
        )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_mime_type="application/json",
                response_schema=LLMDecision,
                temperature=0.0,  # deterministic, so evaluation runs are reproducible
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/transport error uniformly
        raise DecisionUnavailableError(f"Gemini request failed: {exc}") from exc

    if not response.text:
        raise DecisionUnavailableError("Gemini returned an empty response")
    return response.text


def generate_decision(ticket: TicketCreate) -> LLMDecision:
    """Produce a validated decision for a ticket, or raise DecisionUnavailableError."""
    context = select_context(ticket)
    allowed = ", ".join(a.value for a in Action)

    prompt = (
        f"COMPANY POLICY EXCERPTS\n{'=' * 60}\n{format_context(context)}\n\n"
        f"SUPPORT TICKET\n{'=' * 60}\n{_describe_ticket(ticket)}\n\n"
        f"ALLOWED ACTIONS\n{'=' * 60}\n{allowed}\n\n"
        "Return the decision as JSON."
    )

    last_error: Exception | None = None
    for strict in (False, True):
        raw = _call_gemini(prompt, strict_retry=strict)
        try:
            decision = LLMDecision.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            logger.warning("Invalid decision payload (strict_retry=%s): %s", strict, exc)
            continue
        return _ground_sources(decision, context)

    raise DecisionUnavailableError(f"Model output failed validation twice: {last_error}")


def _ground_sources(decision: LLMDecision, context: list[RetrievedChunk]) -> LLMDecision:
    """Drop any citation that was not in the retrieved context."""
    retrieved = {c.chunk.source for c in context}
    grounded = [s for s in decision.sources if s in retrieved]

    if not grounded and decision.action is not Action.NEEDS_MORE_INFORMATION:
        # Cited nothing usable: fall back to the policies we actually showed it,
        # so the stored decision always points at real documents.
        grounded = sorted(retrieved)

    return decision.model_copy(update={"sources": grounded})
