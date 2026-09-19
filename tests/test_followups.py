"""Ticket follow-ups, photo evidence, and reassessment.

Both Gemini calls (photo analysis and the decision) are stubbed, and uploads
land in a per-test temporary folder (see conftest.client).
"""

import hashlib
import io
import struct
import zlib

import pytest

from src.actions import Action
from src.config import settings
from src.decision import (
    DecisionUnavailableError,
    PhotoEvidence,
    TicketHistory,
    build_prompt,
    enforce_evidence_requirement,
)
from src.evidence import clean_filename, detect_image_type, validate_photo, PhotoRejectedError
from src.retrieval import Chunk, RetrievedChunk
from src.schemas import LLMDecision, PhotoAnalysis, TicketCreate
from tests.conftest import STUB_ANALYSIS, STUB_DECISION, auth, register_and_login

DAMAGED = {
    "message": "My order worth 3500 arrived damaged yesterday.",
    "order_value_inr": 3500,
    "days_since_delivery": 1,
    "product_type": "non_food",
    "opened_status": "opened",
    "order_status": "delivered",
}


def make_png(width: int = 8, height: int = 8) -> bytes:
    """A real, decodable PNG built with the standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xcc\x22\x22" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def new_ticket(client, token) -> int:
    response = client.post("/tickets", json=DAMAGED, headers=auth(token))
    assert response.status_code == 201, response.text
    return response.json()["id"]


def follow_up(client, token, ticket_id, message=None, files=None):
    data = {"message": message} if message is not None else {}
    return client.post(
        f"/tickets/{ticket_id}/follow-ups", data=data, files=files or [], headers=auth(token)
    )


def uploaded_files():
    return sorted(settings.uploads_dir.glob("*")) if settings.uploads_dir.exists() else []


# --------------------------------------------------------------------------
# Feature 1: text follow-ups on the same ticket
# --------------------------------------------------------------------------


def test_follow_up_continues_the_same_ticket(client):
    token = register_and_login(client, "fu1@example.com")
    ticket_id = new_ticket(client, token)

    response = follow_up(client, token, ticket_id, "The delivery was yesterday, box was crushed.")
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["id"] == ticket_id
    assert [m["body"] for m in body["messages"]] == ["The delivery was yesterday, box was crushed."]
    assert len(body["decisions"]) == 2, "the first decision must be preserved, not overwritten"
    assert body["decision"] == body["decisions"][-1], "current decision is the latest one"

    # Still exactly one ticket - a follow-up never creates another.
    assert len(client.get("/tickets", headers=auth(token)).json()) == 1


def test_conversation_persists_on_reload(client):
    token = register_and_login(client, "fu2@example.com")
    ticket_id = new_ticket(client, token)
    follow_up(client, token, ticket_id, "first")
    follow_up(client, token, ticket_id, "second")

    detail = client.get(f"/tickets/{ticket_id}", headers=auth(token)).json()
    assert [m["body"] for m in detail["messages"]] == ["first", "second"]
    assert len(detail["decisions"]) == 3


def test_history_list_shows_the_latest_action(client, monkeypatch):
    token = register_and_login(client, "fu3@example.com")
    ticket_id = new_ticket(client, token)

    updated = STUB_DECISION.model_copy(update={"action": Action.APPROVE_REFUND_OR_REPLACEMENT})
    monkeypatch.setattr("src.api.generate_decision", lambda ticket, history=None: updated)
    follow_up(client, token, ticket_id, "more detail")

    row = client.get("/tickets", headers=auth(token)).json()[0]
    assert row["action"] == "APPROVE_REFUND_OR_REPLACEMENT"


def test_reassessment_receives_the_whole_conversation(client, monkeypatch):
    token = register_and_login(client, "fu4@example.com")
    ticket_id = new_ticket(client, token)
    follow_up(client, token, ticket_id, "It is a ceramic mug.")

    seen = {}

    def capture(ticket, history=None):
        seen["ticket"], seen["history"] = ticket, history
        return STUB_DECISION

    monkeypatch.setattr("src.api.generate_decision", capture)
    follow_up(client, token, ticket_id, "Photos attached.", files=[("photos", ("a.png", make_png(), "image/png"))])

    history = seen["history"]
    assert seen["ticket"].message == DAMAGED["message"], "original complaint is reused"
    assert seen["ticket"].order_value_inr == 3500, "original structured fields are reused"
    assert history.follow_ups == ("It is a ceramic mug.", "Photos attached.")
    assert [a for a, _ in history.prior_decisions] == ["REQUEST_PHOTOS", "REQUEST_PHOTOS"]
    assert len(history.photos) == 1 and history.photos[0].description == STUB_ANALYSIS.description


def test_empty_follow_up_is_rejected(client):
    token = register_and_login(client, "fu5@example.com")
    ticket_id = new_ticket(client, token)

    assert follow_up(client, token, ticket_id, "   ").status_code == 422
    assert follow_up(client, token, ticket_id).status_code == 422


# --------------------------------------------------------------------------
# Features 2 + 3: photo upload, storage, retrieval
# --------------------------------------------------------------------------


def test_photos_are_saved_privately_and_linked_to_the_ticket(client):
    token = register_and_login(client, "ph1@example.com")
    ticket_id = new_ticket(client, token)
    png = make_png()

    response = follow_up(
        client, token, ticket_id, "Here is the damage.",
        files=[("photos", ("../../etc/mug.png", png, "image/png")), ("photos", ("box.jpg", JPEG, "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    photos = response.json()["photos"]

    assert [p["original_filename"] for p in photos] == ["mug.png", "box.jpg"], "path parts stripped"
    assert photos[0]["content_type"] == "image/png" and photos[1]["content_type"] == "image/jpeg"
    assert photos[0]["analysis"] == STUB_ANALYSIS.description
    assert photos[0]["is_clear"] and photos[0]["is_relevant"] and photos[0]["shows_issue"]

    # Nothing in the API response reveals where or under what name it is stored.
    text = response.text
    assert "stored_name" not in text and str(settings.uploads_dir) not in text

    stored = uploaded_files()
    assert len(stored) == 2
    assert all(len(f.stem) == 32 for f in stored), "stored under random server names"
    assert hashlib.sha256(png).hexdigest() in {hashlib.sha256(f.read_bytes()).hexdigest() for f in stored}


def test_owner_can_download_their_photo(client):
    token = register_and_login(client, "ph2@example.com")
    ticket_id = new_ticket(client, token)
    png = make_png()
    photo_id = follow_up(
        client, token, ticket_id, files=[("photos", ("mug.png", png, "image/png"))]
    ).json()["photos"][0]["id"]

    response = client.get(f"/tickets/{ticket_id}/photos/{photo_id}", headers=auth(token))
    assert response.status_code == 200
    assert response.content == png
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "no-store" in response.headers["cache-control"]


def test_photo_download_requires_authentication(client):
    token = register_and_login(client, "ph3@example.com")
    ticket_id = new_ticket(client, token)
    photo_id = follow_up(
        client, token, ticket_id, files=[("photos", ("m.png", make_png(), "image/png"))]
    ).json()["photos"][0]["id"]

    assert client.get(f"/tickets/{ticket_id}/photos/{photo_id}").status_code == 401


@pytest.mark.parametrize(
    ("name", "payload", "mime", "expected"),
    [
        ("notes.png", b"this is plain text pretending to be a png", "image/png", 415),
        ("anim.gif", b"GIF89a" + b"\x00" * 20, "image/gif", 415),
        ("script.jpg", b"<script>alert(1)</script>", "image/jpeg", 415),
        ("empty.png", b"", "image/png", 422),
    ],
    ids=["text-renamed-png", "gif", "html-as-jpg", "empty-file"],
)
def test_invalid_files_are_rejected_and_nothing_is_saved(client, name, payload, mime, expected):
    token = register_and_login(client, "bad@example.com")
    ticket_id = new_ticket(client, token)

    response = follow_up(client, token, ticket_id, "see photo", files=[("photos", (name, payload, mime))])
    assert response.status_code == expected, response.text

    detail = client.get(f"/tickets/{ticket_id}", headers=auth(token)).json()
    assert detail["photos"] == [] and detail["messages"] == [] and len(detail["decisions"]) == 1
    assert uploaded_files() == []


def test_oversized_photo_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_photo_bytes", 1024)
    token = register_and_login(client, "big@example.com")
    ticket_id = new_ticket(client, token)

    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000
    response = follow_up(client, token, ticket_id, files=[("photos", ("big.png", big, "image/png"))])
    assert response.status_code == 413
    assert uploaded_files() == []


def test_too_many_photos_are_rejected(client):
    token = register_and_login(client, "many@example.com")
    ticket_id = new_ticket(client, token)

    files = [("photos", (f"p{i}.png", make_png(), "image/png")) for i in range(settings.max_photos_per_upload + 1)]
    assert follow_up(client, token, ticket_id, files=files).status_code == 422
    assert uploaded_files() == []


def test_ai_failure_leaves_no_rows_and_no_files(client, monkeypatch):
    token = register_and_login(client, "fail@example.com")
    ticket_id = new_ticket(client, token)

    def down(ticket, history=None):
        raise DecisionUnavailableError("gemini is down")

    monkeypatch.setattr("src.api.generate_decision", down)
    response = follow_up(client, token, ticket_id, "photo", files=[("photos", ("m.png", make_png(), "image/png"))])
    assert response.status_code == 503

    detail = client.get(f"/tickets/{ticket_id}", headers=auth(token)).json()
    assert detail["photos"] == [] and detail["messages"] == [] and len(detail["decisions"]) == 1
    assert uploaded_files() == []


def test_analysis_failure_leaves_no_rows_and_no_files(client, monkeypatch):
    token = register_and_login(client, "fail2@example.com")
    ticket_id = new_ticket(client, token)

    def down(photos, complaint):
        raise DecisionUnavailableError("vision is down")

    monkeypatch.setattr("src.api.analyze_photos", down)
    assert follow_up(
        client, token, ticket_id, files=[("photos", ("m.png", make_png(), "image/png"))]
    ).status_code == 503
    assert uploaded_files() == []


# --------------------------------------------------------------------------
# Ownership: another user can't touch the ticket or its photos
# --------------------------------------------------------------------------


def test_bob_cannot_follow_up_upload_to_or_download_from_alices_ticket(client):
    alice = register_and_login(client, "alice-fu@example.com")
    bob = register_and_login(client, "bob-fu@example.com")
    alice_ticket = new_ticket(client, alice)
    photo_id = follow_up(
        client, alice, alice_ticket, files=[("photos", ("m.png", make_png(), "image/png"))]
    ).json()["photos"][0]["id"]
    files_before = uploaded_files()

    assert follow_up(client, bob, alice_ticket, "let me in").status_code == 404
    assert follow_up(
        client, bob, alice_ticket, files=[("photos", ("x.png", make_png(), "image/png"))]
    ).status_code == 404
    assert client.get(f"/tickets/{alice_ticket}/photos/{photo_id}", headers=auth(bob)).status_code == 404

    # Using his OWN ticket id with Alice's photo id doesn't work either.
    bob_ticket = new_ticket(client, bob)
    assert client.get(f"/tickets/{bob_ticket}/photos/{photo_id}", headers=auth(bob)).status_code == 404

    assert uploaded_files() == files_before, "Bob's rejected upload wrote nothing"
    detail = client.get(f"/tickets/{alice_ticket}", headers=auth(alice)).json()
    assert detail["messages"] == [] and len(detail["photos"]) == 1


# --------------------------------------------------------------------------
# Evidence guardrail: no approval without usable evidence
# --------------------------------------------------------------------------


def evidence(clear=True, relevant=True, shows=True) -> PhotoEvidence:
    return PhotoEvidence("p.jpg", "a photo", clear, relevant, shows)


APPROVE = LLMDecision(
    action=Action.APPROVE_REFUND_OR_REPLACEMENT, confidence=0.95, reason="ok", sources=["damaged_goods.md"]
)
ASKED_FOR_PHOTOS = (("REQUEST_PHOTOS", "Please send photos."),)


@pytest.mark.parametrize(
    "photo",
    [evidence(clear=False), evidence(relevant=False), evidence(shows=False)],
    ids=["blurry", "irrelevant", "no-visible-damage"],
)
def test_unusable_photo_cannot_unlock_an_approval(photo):
    history = TicketHistory(prior_decisions=ASKED_FOR_PHOTOS, photos=(photo,))
    result = enforce_evidence_requirement(APPROVE, history)

    assert result.action is Action.REQUEST_PHOTOS
    assert "photo" in result.reason.lower()


def test_no_photo_at_all_cannot_unlock_an_approval():
    history = TicketHistory(prior_decisions=ASKED_FOR_PHOTOS, follow_ups=("please just refund me",))
    assert enforce_evidence_requirement(APPROVE, history).action is Action.REQUEST_PHOTOS


def test_one_usable_photo_among_bad_ones_allows_the_policy_decision():
    history = TicketHistory(
        prior_decisions=ASKED_FOR_PHOTOS, photos=(evidence(clear=False), evidence())
    )
    assert enforce_evidence_requirement(APPROVE, history).action is Action.APPROVE_REFUND_OR_REPLACEMENT


def test_defect_evidence_request_guards_replacement():
    history = TicketHistory(
        prior_decisions=(("REQUEST_DEFECT_EVIDENCE", "Show the defect."),),
        photos=(evidence(shows=False),),
    )
    approve = APPROVE.model_copy(update={"action": Action.APPROVE_REPLACEMENT})
    assert enforce_evidence_requirement(approve, history).action is Action.REQUEST_DEFECT_EVIDENCE


@pytest.mark.parametrize(
    "granting",
    [
        Action.OFFER_REPLACEMENT_OR_REFUND,  # the look-alike a live run actually produced
        Action.APPROVE_RETURN,
        Action.CANCEL_AND_REFUND,
        Action.REPLACE_CORRECT_ITEM,
    ],
)
def test_no_granting_action_of_any_kind_slips_past_an_evidence_request(granting):
    """Regression: the first guard only watched the one 'paired' approval, so a
    different remedy with an irrelevant photo would have gone straight through."""
    history = TicketHistory(prior_decisions=ASKED_FOR_PHOTOS, photos=(evidence(relevant=False),))
    decision = APPROVE.model_copy(update={"action": granting})

    assert enforce_evidence_requirement(decision, history).action is Action.REQUEST_PHOTOS


def test_every_action_is_described_to_the_model():
    from src.actions import ACTION_GUIDE

    assert set(ACTION_GUIDE) == set(Action), "an undescribed action would be guessed at"
    prompt = build_prompt(TicketCreate(**DAMAGED), [])
    assert "- APPROVE_REFUND_OR_REPLACEMENT: damaged-goods claim that is approved" in prompt
    assert "OFFER_REPLACEMENT_OR_REFUND: undelivered order" in prompt


def test_guardrail_leaves_non_approvals_and_unrequested_cases_alone():
    history = TicketHistory(prior_decisions=ASKED_FOR_PHOTOS, photos=(evidence(clear=False),))
    reject = APPROVE.model_copy(update={"action": Action.REJECT_OUTSIDE_WINDOW})
    assert enforce_evidence_requirement(reject, history).action is Action.REJECT_OUTSIDE_WINDOW

    never_asked = TicketHistory(prior_decisions=(("APPROVE_RETURN", "ok"),))
    assert enforce_evidence_requirement(APPROVE, never_asked) is APPROVE
    assert enforce_evidence_requirement(APPROVE, None) is APPROVE


def test_prompt_contains_conversation_and_photo_findings():
    ticket = TicketCreate(**DAMAGED)
    context = [RetrievedChunk(Chunk(text="Damaged Goods Policy (rule 3): ...", source="damaged_goods.md", rule="3"), 0.9)]
    history = TicketHistory(
        prior_decisions=ASKED_FOR_PHOTOS,
        follow_ups=("Photos attached.",),
        photos=(PhotoEvidence("mug.jpg", "A cracked mug.", True, True, True),),
    )

    prompt = build_prompt(ticket, context, history)
    assert "EARLIER DECISIONS" in prompt and "REQUEST_PHOTOS: Please send photos." in prompt
    assert "Photos attached." in prompt
    assert "mug.jpg" in prompt and "A cracked mug." in prompt and "shows reported problem=yes" in prompt
    assert "TICKET HISTORY" not in build_prompt(ticket, context), "first decisions are unchanged"


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def test_image_type_comes_from_bytes_not_the_name():
    assert detect_image_type(make_png()) == ("image/png", "png")
    assert detect_image_type(JPEG) == ("image/jpeg", "jpg")
    assert detect_image_type(WEBP) == ("image/webp", "webp")
    assert detect_image_type(b"GIF89a....") is None


def test_clean_filename_strips_paths_and_odd_characters():
    assert clean_filename("..\\..\\windows\\evil.png") == "evil.png"
    assert clean_filename("/etc/passwd") == "passwd"
    assert clean_filename("my photo<>|.jpg") == "my photo___.jpg"
    assert clean_filename("") == "photo"


def test_validate_photo_reads_at_most_one_byte_past_the_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_photo_bytes", 100)

    class CountingStream(io.BytesIO):
        requested: list[int] = []

        def read(self, size=-1):
            self.requested.append(size)
            return super().read(size)

    stream = CountingStream(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10_000)
    with pytest.raises(PhotoRejectedError) as excinfo:
        validate_photo(stream, "big.png")
    assert excinfo.value.status_code == 413
    assert stream.requested == [101], "never buffers the whole oversized upload"


def test_photo_analysis_schema_rejects_extra_or_missing_fields():
    with pytest.raises(ValueError):
        PhotoAnalysis.model_validate({"description": "x", "is_clear": True, "is_relevant": True})
    with pytest.raises(ValueError):
        PhotoAnalysis.model_validate(
            {"description": "x", "is_clear": True, "is_relevant": True, "shows_issue": True, "approve": True}
        )
