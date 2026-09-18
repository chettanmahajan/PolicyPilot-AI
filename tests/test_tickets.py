"""Ticket creation, history, detail - and the ownership boundary between users."""

import pytest

from src.decision import DecisionUnavailableError
from tests.conftest import STUB_DECISION, auth, register_and_login

DAMAGED_TICKET = {
    "message": "My order arrived damaged yesterday.",
    "order_value_inr": 3500,
    "days_since_delivery": 1,
    "product_type": "non_food",
    "opened_status": "opened",
    "order_status": "delivered",
}


_DEFAULT = object()


def create_ticket(client, token, payload=_DEFAULT):
    # Sentinel rather than `payload or DAMAGED_TICKET`: an empty dict is a
    # deliberate test case and must not be swapped for the valid default.
    body = DAMAGED_TICKET if payload is _DEFAULT else payload
    return client.post("/tickets", json=body, headers=auth(token))


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------


def test_create_ticket_returns_the_stored_decision(client):
    token = register_and_login(client, "owner@example.com")
    response = create_ticket(client, token)

    assert response.status_code == 201
    body = response.json()
    assert body["message"] == DAMAGED_TICKET["message"]

    decision = body["decision"]
    assert decision["action"] == STUB_DECISION.action.value
    assert decision["confidence"] == STUB_DECISION.confidence
    assert decision["sources"] == STUB_DECISION.sources
    assert decision["reason"]


def test_create_ticket_requires_authentication(client):
    assert client.post("/tickets", json=DAMAGED_TICKET).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "ok", "order_value_inr": -5},
        {"message": "ok", "days_since_delivery": -1},
        {"message": "ok", "product_type": "liquid"},
        {},
    ],
)
def test_invalid_ticket_payloads_are_rejected(client, payload):
    token = register_and_login(client, "validate@example.com")
    assert create_ticket(client, token, payload).status_code == 422


def test_optional_fields_may_be_omitted(client):
    token = register_and_login(client, "sparse@example.com")
    response = create_ticket(client, token, {"message": "I want to return this."})

    assert response.status_code == 201
    assert response.json()["order_value_inr"] is None


def test_nothing_is_persisted_when_the_decision_pipeline_fails(client, monkeypatch):
    def boom(ticket):
        raise DecisionUnavailableError("gemini is down")

    monkeypatch.setattr("src.api.generate_decision", boom)
    token = register_and_login(client, "outage@example.com")

    response = create_ticket(client, token)
    assert response.status_code == 503

    # The ticket must not be left behind without a decision.
    history = client.get("/tickets", headers=auth(token))
    assert history.json() == []


# --------------------------------------------------------------------------
# History and detail
# --------------------------------------------------------------------------


def test_history_lists_only_the_callers_own_tickets(client):
    alice = register_and_login(client, "alice@example.com")
    bob = register_and_login(client, "bob@example.com")

    create_ticket(client, alice)
    create_ticket(client, alice)
    create_ticket(client, bob)

    assert len(client.get("/tickets", headers=auth(alice)).json()) == 2
    assert len(client.get("/tickets", headers=auth(bob)).json()) == 1


def test_history_includes_the_decision_summary(client):
    token = register_and_login(client, "summary@example.com")
    create_ticket(client, token)

    row = client.get("/tickets", headers=auth(token)).json()[0]
    assert row["action"] == STUB_DECISION.action.value
    assert row["confidence"] == STUB_DECISION.confidence


def test_ticket_detail_returns_decision_and_sources(client):
    token = register_and_login(client, "detail@example.com")
    ticket_id = create_ticket(client, token).json()["id"]

    response = client.get(f"/tickets/{ticket_id}", headers=auth(token))
    assert response.status_code == 200
    assert response.json()["decision"]["sources"] == STUB_DECISION.sources


def test_history_and_detail_require_authentication(client):
    assert client.get("/tickets").status_code == 401
    assert client.get("/tickets/1").status_code == 401


# --------------------------------------------------------------------------
# Authorization: the requirement called out explicitly in the assignment
# --------------------------------------------------------------------------


def test_alice_cannot_read_bobs_ticket_by_id(client):
    alice = register_and_login(client, "alice2@example.com")
    bob = register_and_login(client, "bob2@example.com")

    bobs_ticket_id = create_ticket(client, bob).json()["id"]

    # Bob can read his own ticket...
    assert client.get(f"/tickets/{bobs_ticket_id}", headers=auth(bob)).status_code == 200

    # ...but Alice's token must not, even with the correct id.
    response = client.get(f"/tickets/{bobs_ticket_id}", headers=auth(alice))
    assert response.status_code == 404, "must not expose another user's ticket"
    assert "message" not in response.json()


def test_unknown_and_forbidden_ids_are_indistinguishable(client):
    alice = register_and_login(client, "alice3@example.com")
    bob = register_and_login(client, "bob3@example.com")
    bobs_ticket_id = create_ticket(client, bob).json()["id"]

    forbidden = client.get(f"/tickets/{bobs_ticket_id}", headers=auth(alice))
    missing = client.get("/tickets/999999", headers=auth(alice))

    # Identical responses, so ticket ids cannot be probed for existence.
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()
