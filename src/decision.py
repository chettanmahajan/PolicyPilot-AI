"""Gemini-backed decision engine.

Flow: ticket -> retrieval query -> relevant policy context -> Gemini (with a
native response schema) -> Pydantic validation -> LLMDecision.

The model is only ever allowed to pick an action from `Action` and to cite
policy files that were actually retrieved, so a hallucinated action or citation
becomes a validation failure rather than a stored decision.
"""

from __future__ import annotations

import logging
import re
import time

from pydantic import ValidationError

from src.actions import Action
from src.config import settings
from src.retrieval import RetrievedChunk, get_client, get_index
from src.schemas import LLMDecision, TicketCreate

logger = logging.getLogger(__name__)


class DecisionUnavailableError(RuntimeError):
    """The decision could not be produced (API failure, or unparseable output)."""


# Declared explicitly rather than derived from LLMDecision: that model sets
# `extra="forbid"`, which Pydantic renders as `additionalProperties: false`, and
# the Gemini API rejects that key outright ("Unknown name additional_properties").
# Keeping the two separate lets the wire schema stay API-compatible while
# LLMDecision stays strict for validation. The enum is included so the API
# itself constrains `action` to the allowed vocabulary.
# Gemini flash models return 503 UNAVAILABLE under load and 429 when rate
# limited. Both clear on their own, so they are retried rather than surfaced as
# a failed decision.
TRANSIENT_RETRIES = 4
MAX_BACKOFF_SECONDS = 70
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

# The 429 body carries the wait the server actually wants, in both of these
# shapes. Honouring it matters on the free tier, where the limit is 5 requests
# per minute and a naive 1-2s backoff simply burns the remaining retries.
_RETRY_AFTER_RE = re.compile(r"'retryDelay':\s*'([\d.]+)s'|retry in ([\d.]+)s")


def _is_transient(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _TRANSIENT_MARKERS)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Server-requested delay if it gave one, else exponential backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        requested = float(match.group(1) or match.group(2))
        return min(requested + 1.0, MAX_BACKOFF_SECONDS)
    return float(2**attempt)


RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": [a.value for a in Action]},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
        "sources": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["action", "confidence", "reason", "sources"],
}


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
    from google.genai import types

    if not settings.gemini_api_key:
        raise DecisionUnavailableError("GEMINI_API_KEY is not set")

    # Shared client, not a fresh one per call: constructing them per request
    # leaks HTTP connections and lets a discarded client close the transport.
    client = get_client()
    instruction = SYSTEM_INSTRUCTION
    if strict_retry:
        instruction += (
            "\nYour previous reply could not be parsed. Return ONLY a JSON object "
            "with exactly the keys: action, confidence, reason, sources."
        )

    config = types.GenerateContentConfig(
        system_instruction=instruction,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.0,  # deterministic, so evaluation runs are reproducible
    )

    response = None
    for attempt in range(TRANSIENT_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model, contents=prompt, config=config
            )
            break
        except Exception as exc:  # noqa: BLE001 - uniform handling of SDK/transport errors
            is_last = attempt == TRANSIENT_RETRIES - 1
            if is_last or not _is_transient(exc):
                raise DecisionUnavailableError(f"Gemini request failed: {exc}") from exc
            delay = _retry_delay(exc, attempt)
            logger.warning(
                "Transient Gemini error (%s), retrying in %.0fs",
                str(exc).split(".")[0][:60],
                delay,
            )
            time.sleep(delay)

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
