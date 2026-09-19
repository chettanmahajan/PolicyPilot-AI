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
from dataclasses import dataclass

from pydantic import ValidationError

from src.actions import Action
from src.config import settings
from src.retrieval import RetrievedChunk, get_client, get_index
from src.schemas import LLMDecision, TicketCreate

logger = logging.getLogger(__name__)


class DecisionUnavailableError(RuntimeError):
    """The decision could not be produced (API failure, or unparseable output)."""


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


# Declared explicitly rather than derived from LLMDecision: that model sets
# `extra="forbid"`, which Pydantic renders as `additionalProperties: false`, and
# the Gemini API rejects that key outright ("Unknown name additional_properties").
# Keeping the two separate lets the wire schema stay API-compatible while
# LLMDecision stays strict for validation. The enum is included so the API
# itself constrains `action` to the allowed vocabulary.
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

Continuing tickets:
- The ticket may include earlier decisions, customer follow-up messages and
  photo evidence. Decide again from ALL of it, together with the policies.
- Follow-up messages may supply facts missing from the structured fields. If the
  customer's words clearly contradict a structured field (for example the
  message states a different order value), do not silently pick one - return
  NEEDS_MORE_INFORMATION and ask which is correct.
- Photo evidence is an automated description of what is visible in each photo.
  It is evidence of what can be seen, nothing more.
- When photographs (or defect evidence) were requested: if at least one photo is
  clear, relevant and visibly shows the reported problem, that evidence
  requirement is met - then apply the rest of the policy (reporting windows,
  thresholds) to reach the decision. If the photos are unclear, irrelevant, or
  do not show the problem, ask again (REQUEST_PHOTOS or REQUEST_DEFECT_EVIDENCE)
  and say in `reason` exactly what photo is needed.
- A photo never overrides any other policy condition, and never justifies an
  approval on its own.
- Whenever you return NEEDS_MORE_INFORMATION, REQUEST_PHOTOS or
  REQUEST_DEFECT_EVIDENCE, `reason` must tell the customer clearly and
  specifically what to provide next.
"""


@dataclass(frozen=True)
class PhotoEvidence:
    filename: str
    description: str
    is_clear: bool
    is_relevant: bool
    shows_issue: bool

    @property
    def usable(self) -> bool:
        """Good enough to satisfy an evidence request."""
        return self.is_clear and self.is_relevant and self.shows_issue


@dataclass(frozen=True)
class TicketHistory:
    """Everything that happened on a ticket after it was first submitted.

    Plain values rather than ORM rows, so the decision engine stays independent
    of the database layer.
    """

    prior_decisions: tuple[tuple[str, str], ...] = ()  # (action, reason), oldest first
    follow_ups: tuple[str, ...] = ()                    # customer messages, oldest first
    photos: tuple[PhotoEvidence, ...] = ()

    @property
    def requested_evidence(self) -> Action | None:
        """The most recent evidence request made on this ticket, if any."""
        for action, _ in reversed(self.prior_decisions):
            if action in _EVIDENCE_APPROVALS:
                return Action(action)
        return None


# Which approval each evidence request unlocks. Both pairings come straight from
# the policies: damaged_goods.md rule 3 (photos before refund/replacement) and
# defective_products.md rule 2 (evidence before replacement).
_EVIDENCE_APPROVALS: dict[str, Action] = {
    Action.REQUEST_PHOTOS.value: Action.APPROVE_REFUND_OR_REPLACEMENT,
    Action.REQUEST_DEFECT_EVIDENCE.value: Action.APPROVE_REPLACEMENT,
}


def _describe_history(history: TicketHistory) -> str:
    sections = []
    if history.prior_decisions:
        lines = [f"- {action}: {reason}" for action, reason in history.prior_decisions]
        sections.append("EARLIER DECISIONS ON THIS TICKET (oldest first)\n" + "\n".join(lines))
    if history.follow_ups:
        lines = [f"- {text}" for text in history.follow_ups]
        sections.append("CUSTOMER FOLLOW-UP MESSAGES (oldest first)\n" + "\n".join(lines))
    if history.photos:
        yes_no = lambda flag: "yes" if flag else "no"  # noqa: E731
        lines = [
            f"- Photo {i} ({p.filename}): clear={yes_no(p.is_clear)}, "
            f"relevant={yes_no(p.is_relevant)}, shows reported problem={yes_no(p.shows_issue)}. "
            f"Visible: {p.description}"
            for i, p in enumerate(history.photos, start=1)
        ]
        sections.append(
            "PHOTO EVIDENCE (automated description of what is visible; not proof on its own)\n"
            + "\n".join(lines)
        )
    return "\n\n".join(sections)


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


def select_context(ticket: TicketCreate, extra_query: str = "") -> list[RetrievedChunk]:
    """Rank rules by similarity, then widen to every rule in the winning policies.

    Ranking individual rules is what makes this retrieval rather than stuffing
    the whole knowledge base in. But a policy's rules are interdependent - the
    threshold is in one rule and its exception in the next - so once a document
    is judged relevant we include all of its rules. This avoids the common
    failure where the top-k cuts off the exception that changes the answer.
    """
    index = get_index()
    query = build_retrieval_query(ticket)
    if extra_query:
        # Follow-ups can move the conversation onto a different policy (e.g.
        # "actually it was the wrong flavour"), so they steer retrieval too.
        query = f"{query}. {extra_query}"
    ranked = index.search(query)

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


def generate_json(contents: object, *, instruction: str, schema: dict[str, object]) -> str:
    """One structured Gemini call with transient-error retries; returns raw JSON text.

    Shared by the decision engine and photo analysis so both get the same
    backoff behaviour. `contents` may be a prompt string or a list mixing text
    and image parts.
    """
    from google.genai import types

    if not settings.gemini_api_key:
        raise DecisionUnavailableError("GEMINI_API_KEY is not set")

    # Shared client, not a fresh one per call: constructing them per request
    # leaks HTTP connections and lets a discarded client close the transport.
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=instruction,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.0,  # deterministic, so evaluation runs are reproducible
    )

    response = None
    for attempt in range(TRANSIENT_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model, contents=contents, config=config
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


def _call_gemini(prompt: str, *, strict_retry: bool = False) -> str:
    instruction = SYSTEM_INSTRUCTION
    if strict_retry:
        instruction += (
            "\nYour previous reply could not be parsed. Return ONLY a JSON object "
            "with exactly the keys: action, confidence, reason, sources."
        )
    return generate_json(prompt, instruction=instruction, schema=RESPONSE_SCHEMA)


def build_prompt(
    ticket: TicketCreate, context: list[RetrievedChunk], history: TicketHistory | None = None
) -> str:
    allowed = ", ".join(a.value for a in Action)
    prompt = (
        f"COMPANY POLICY EXCERPTS\n{'=' * 60}\n{format_context(context)}\n\n"
        f"SUPPORT TICKET\n{'=' * 60}\n{_describe_ticket(ticket)}\n\n"
    )
    if history is not None and (history.prior_decisions or history.follow_ups or history.photos):
        prompt += f"TICKET HISTORY\n{'=' * 60}\n{_describe_history(history)}\n\n"
    return prompt + f"ALLOWED ACTIONS\n{'=' * 60}\n{allowed}\n\nReturn the decision as JSON."


def generate_decision(ticket: TicketCreate, history: TicketHistory | None = None) -> LLMDecision:
    """Produce a validated decision for a ticket, or raise DecisionUnavailableError.

    `history` is given when an existing ticket is reassessed after a follow-up.
    """
    extra_query = " ".join(history.follow_ups) if history else ""
    context = select_context(ticket, extra_query)
    prompt = build_prompt(ticket, context, history)

    last_error: Exception | None = None
    for strict in (False, True):
        raw = _call_gemini(prompt, strict_retry=strict)
        try:
            decision = LLMDecision.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            logger.warning("Invalid decision payload (strict_retry=%s): %s", strict, exc)
            continue
        decision = _ground_sources(decision, context)
        return enforce_evidence_requirement(decision, history)

    raise DecisionUnavailableError(f"Model output failed validation twice: {last_error}")


def enforce_evidence_requirement(
    decision: LLMDecision, history: TicketHistory | None
) -> LLMDecision:
    """Code-level backstop: no approval after an evidence request without usable evidence.

    The prompt already says this, but "an image was uploaded, so approve" is
    exactly the failure that must not depend on the model behaving. This is not
    a new rule - it is damaged_goods.md rule 3 / defective_products.md rule 2
    applied deterministically: the evidence has to exist before the approval.
    """
    if history is None:
        return decision

    requested = history.requested_evidence
    if requested is None or decision.action is not _EVIDENCE_APPROVALS[requested.value]:
        return decision
    if any(photo.usable for photo in history.photos):
        return decision

    logger.warning(
        "Blocked %s: %s was requested but no photo is clear, relevant and shows the issue",
        decision.action,
        requested,
    )
    what = (
        "the damaged product and its packaging"
        if requested is Action.REQUEST_PHOTOS
        else "the defect"
    )
    return decision.model_copy(
        update={
            "action": requested,
            "confidence": 0.9,  # rule-based: the missing evidence is certain
            "reason": (
                f"The photos received do not clearly show {what}, so the evidence the "
                f"policy requires is still missing. Please upload a clear, well-lit photo "
                f"of {what} before this can be approved."
            ),
        }
    )


def _ground_sources(decision: LLMDecision, context: list[RetrievedChunk]) -> LLMDecision:
    """Drop any citation that was not in the retrieved context."""
    retrieved = {c.chunk.source for c in context}
    grounded = [s for s in decision.sources if s in retrieved]

    if not grounded and decision.action is not Action.NEEDS_MORE_INFORMATION:
        # Cited nothing usable: fall back to the policies we actually showed it,
        # so the stored decision always points at real documents.
        grounded = sorted(retrieved)

    return decision.model_copy(update={"sources": grounded})
