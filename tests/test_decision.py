"""Decision engine: schema validation, source grounding, and failure handling.

Gemini is never called; `_call_gemini` is stubbed so these run offline.
"""

import json

import pytest
from pydantic import ValidationError

from src import decision as decision_module
from src.actions import Action
from src.decision import (
    DecisionUnavailableError,
    _ground_sources,
    build_retrieval_query,
    format_context,
    generate_decision,
    select_context,
)
from src.retrieval import Chunk, RetrievedChunk, load_chunks
from src.schemas import LLMDecision, TicketCreate

DAMAGED = TicketCreate(
    message="My order arrived damaged yesterday.",
    order_value_inr=3500,
    days_since_delivery=1,
    product_type="non_food",
    opened_status="opened",
    order_status="delivered",
)

VAGUE = TicketCreate(message="I want to return this.", order_value_inr=900)


def chunk(source: str, rule: str, text: str = "some rule", score: float = 0.9):
    return RetrievedChunk(Chunk(text=text, source=source, rule=rule), score)


class _FakeIndex:
    """Stands in for PolicyIndex: real chunks, fixed ranking, no embedding calls."""

    def __init__(self, ranked: list[RetrievedChunk]):
        self.chunks = load_chunks()
        self._ranked = ranked

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._ranked


# --------------------------------------------------------------------------
# Output schema validation
# --------------------------------------------------------------------------


def test_valid_payload_parses():
    parsed = LLMDecision.model_validate_json(
        json.dumps(
            {
                "action": "REQUEST_PHOTOS",
                "confidence": 0.91,
                "reason": "Above the 2,000 threshold.",
                "sources": ["damaged_goods.md"],
            }
        )
    )
    assert parsed.action is Action.REQUEST_PHOTOS


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "GIVE_FREE_STUFF", "confidence": 0.9, "reason": "r", "sources": []},
        {"action": "REQUEST_PHOTOS", "confidence": 1.5, "reason": "r", "sources": []},
        {"action": "REQUEST_PHOTOS", "confidence": -0.1, "reason": "r", "sources": []},
        {"action": "REQUEST_PHOTOS", "confidence": 0.9, "reason": "", "sources": []},
        {"action": "REQUEST_PHOTOS", "confidence": 0.9, "sources": []},
        {"confidence": 0.9, "reason": "r", "sources": []},
        {"action": "REQUEST_PHOTOS", "confidence": 0.9, "reason": "r", "extra": "x"},
    ],
    ids=[
        "invented-action",
        "confidence-too-high",
        "confidence-negative",
        "empty-reason",
        "missing-reason",
        "missing-action",
        "unexpected-key",
    ],
)
def test_malformed_payloads_are_rejected(payload):
    with pytest.raises(ValidationError):
        LLMDecision.model_validate(payload)


# --------------------------------------------------------------------------
# Retrieval query + context assembly
# --------------------------------------------------------------------------


def test_retrieval_query_includes_message_and_structured_facts():
    query = build_retrieval_query(DAMAGED)

    assert "arrived damaged" in query
    assert "delivered" in query
    assert "1 days since delivery" in query


def test_context_expands_to_all_rules_of_a_relevant_policy(monkeypatch):
    """Top-k ranks rules, but every rule of a matched policy must be included.

    Otherwise the threshold can be retrieved while its exception is cut off.
    """
    monkeypatch.setattr(
        decision_module,
        "get_index",
        lambda: _FakeIndex(ranked=[chunk("damaged_goods.md", "3", score=0.9)]),
    )

    context = select_context(DAMAGED)
    sources = {c.chunk.source for c in context}
    rules = {c.chunk.rule for c in context if c.chunk.source == "damaged_goods.md"}

    assert sources == {"damaged_goods.md"}
    assert rules == {"1", "2", "3", "4", "5"}, "all 5 damaged-goods rules expected"


def test_format_context_labels_each_source(monkeypatch):
    text = format_context([chunk("returns.md", "1", "Returns Policy (rule 1): ...")])
    assert "source: returns.md" in text


# --------------------------------------------------------------------------
# Source grounding
# --------------------------------------------------------------------------


def test_hallucinated_citations_are_dropped():
    context = [chunk("damaged_goods.md", "3")]
    raw = LLMDecision(
        action=Action.REQUEST_PHOTOS,
        confidence=0.9,
        reason="r",
        sources=["damaged_goods.md", "refunds.md", "made_up.md"],
    )

    assert _ground_sources(raw, context).sources == ["damaged_goods.md"]


def test_decision_without_usable_citations_falls_back_to_retrieved_files():
    context = [chunk("damaged_goods.md", "3"), chunk("returns.md", "1")]
    raw = LLMDecision(
        action=Action.REQUEST_PHOTOS, confidence=0.9, reason="r", sources=["nonsense.md"]
    )

    assert _ground_sources(raw, context).sources == ["damaged_goods.md", "returns.md"]


def test_needs_more_information_may_cite_nothing():
    context = [chunk("returns.md", "5")]
    raw = LLMDecision(
        action=Action.NEEDS_MORE_INFORMATION, confidence=0.4, reason="r", sources=[]
    )

    assert _ground_sources(raw, context).sources == []


# --------------------------------------------------------------------------
# End-to-end with a stubbed model
# --------------------------------------------------------------------------


@pytest.fixture
def stub_context(monkeypatch):
    monkeypatch.setattr(
        decision_module,
        "select_context",
        lambda ticket: [chunk("damaged_goods.md", "3")],
    )


def test_generate_decision_returns_validated_result(monkeypatch, stub_context):
    monkeypatch.setattr(
        decision_module,
        "_call_gemini",
        lambda prompt, strict_retry=False: json.dumps(
            {
                "action": "REQUEST_PHOTOS",
                "confidence": 0.91,
                "reason": "Order is above the 2,000 threshold.",
                "sources": ["damaged_goods.md"],
            }
        ),
    )

    result = generate_decision(DAMAGED)
    assert result.action is Action.REQUEST_PHOTOS
    assert result.sources == ["damaged_goods.md"]


def test_malformed_output_is_retried_once_then_succeeds(monkeypatch, stub_context):
    calls = []

    def flaky(prompt, strict_retry=False):
        calls.append(strict_retry)
        if not strict_retry:
            return "Sure! Here is the decision: {not valid json"
        return json.dumps(
            {
                "action": "NEEDS_MORE_INFORMATION",
                "confidence": 0.3,
                "reason": "Missing delivery date.",
                "sources": [],
            }
        )

    monkeypatch.setattr(decision_module, "_call_gemini", flaky)

    result = generate_decision(VAGUE)
    assert calls == [False, True], "should retry exactly once, with the strict nudge"
    assert result.action is Action.NEEDS_MORE_INFORMATION


def test_persistently_malformed_output_raises(monkeypatch, stub_context):
    monkeypatch.setattr(
        decision_module, "_call_gemini", lambda prompt, strict_retry=False: "still not json"
    )

    with pytest.raises(DecisionUnavailableError, match="failed validation twice"):
        generate_decision(VAGUE)


def test_api_failure_propagates_as_decision_unavailable(monkeypatch, stub_context):
    def boom(prompt, strict_retry=False):
        raise DecisionUnavailableError("Gemini request failed: 503")

    monkeypatch.setattr(decision_module, "_call_gemini", boom)

    with pytest.raises(DecisionUnavailableError):
        generate_decision(DAMAGED)


def test_missing_api_key_is_reported_clearly(monkeypatch, stub_context):
    monkeypatch.setattr(decision_module.settings, "gemini_api_key", "")

    with pytest.raises(DecisionUnavailableError, match="GEMINI_API_KEY"):
        generate_decision(DAMAGED)
