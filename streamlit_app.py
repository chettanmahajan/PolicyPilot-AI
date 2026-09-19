"""PolicyPilot AI - Streamlit frontend.

Talks to the FastAPI backend over HTTP only. It holds no database connection
and repeats none of the backend's decision or authorization logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
import streamlit as st

from src.config import settings

TIMEOUT = 60

st.set_page_config(page_title="PolicyPilot AI", page_icon="🧭", layout="centered")


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api(method: str, path: str, **kwargs: Any) -> tuple[bool, Any]:
    """Call the backend. Returns (ok, payload-or-error-message)."""
    timeout = kwargs.pop("timeout", TIMEOUT)
    try:
        response = requests.request(
            method,
            f"{settings.api_base_url}{path}",
            headers=_headers(),
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the API at {settings.api_base_url} ({exc})."

    if response.status_code == 401:
        st.session_state.pop("token", None)
        st.session_state.pop("active_ticket", None)
        return False, "Your session expired. Please log in again."

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(d.get("msg", str(d)) for d in detail)
        return False, str(detail)

    return True, response.json() if response.content else None


def logged_in() -> bool:
    return bool(st.session_state.get("token"))


# --------------------------------------------------------------------------
# Login / Register
# --------------------------------------------------------------------------


def render_auth() -> None:
    st.title("🧭 PolicyPilot AI")
    st.caption("Turning customer complaints into policy-backed decisions.")

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pw")
            if st.form_submit_button("Log in", width="stretch"):
                ok, data = api("POST", "/login", json={"email": email, "password": password})
                if ok:
                    st.session_state["token"] = data["access_token"]
                    st.session_state["email"] = email
                    st.rerun()
                else:
                    st.error(data)

    with register_tab:
        with st.form("register"):
            email = st.text_input("Email", key="reg_email")
            password = st.text_input(
                "Password", type="password", key="reg_pw", help="At least 8 characters."
            )
            if st.form_submit_button("Create account", width="stretch"):
                ok, data = api("POST", "/register", json={"email": email, "password": password})
                if ok:
                    st.success("Account created. Switch to the Log in tab.")
                else:
                    st.error(data)


# --------------------------------------------------------------------------
# Decision rendering
# --------------------------------------------------------------------------


def render_decision(decision: dict[str, Any] | None) -> None:
    if not decision:
        st.warning("No decision is attached to this ticket.")
        return

    st.subheader(decision["action"].replace("_", " ").title())

    # "Decision basis" replaces a confidence percentage: the model's number is
    # an uncalibrated self-rating (it is ~1.0 almost every time), so showing it
    # as "100%" presented a guess as a probability. The basis comes from
    # explicit rules in decision.decision_basis(), and is shown beside the
    # decision - it says how much weight the decision bears, not whether the
    # customer is eligible.
    basis = decision.get("basis", "clear")
    reasons = decision.get("basis_reasons") or []
    if basis == "awaiting_customer":
        st.info("⏳ **Awaiting customer** — not a final decision yet.")
    elif basis == "review":
        st.warning("🔎 **Review recommended** — " + " ".join(reasons))
    else:
        st.success("✅ **Clear policy match** — a cited policy rule applies and no review flags were raised.")
    st.caption(
        f"Model self-rating: {float(decision['confidence']):.2f} "
        "(uncalibrated — not a probability, and not part of the eligibility decision)"
    )

    st.markdown("**Reason**")
    st.write(decision["reason"])

    st.markdown("**Policy sources**")
    if decision["sources"]:
        for source in decision["sources"]:
            st.markdown(f"- `{source}`")
    else:
        st.caption("No policy source cited.")


# --------------------------------------------------------------------------
# Ticket view: conversation, evidence, follow-ups (used by both tabs)
# --------------------------------------------------------------------------

EITHER_OR = {"APPROVE_REFUND_OR_REPLACEMENT", "OFFER_REPLACEMENT_OR_REFUND"}

EVIDENCE_REQUESTS = {
    "REQUEST_PHOTOS": "clear photos of the damaged product **and** its packaging",
    "REQUEST_DEFECT_EVIDENCE": "a clear photo showing the defect",
}


@st.cache_data(ttl=600, show_spinner=False, max_entries=200)
def _photo_bytes(token: str, ticket_id: int, photo_id: int) -> bytes | None:
    """Fetch an evidence photo through the authenticated endpoint.

    The token is part of the cache key, so one user's cached photo can never
    be served to another. Photos are immutable once uploaded, so caching is safe
    and avoids re-downloading every image on every rerun.
    """
    try:
        response = requests.get(
            f"{settings.api_base_url}/tickets/{ticket_id}/photos/{photo_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        return None
    return response.content if response.ok else None


def _when(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")


def _flag(ok: bool, label: str) -> str:
    return f"{'✅' if ok else '❌'} {label}"


def render_timeline(detail: dict[str, Any]) -> None:
    """Every message, photo and decision change on the ticket, oldest first."""
    # Order within one follow-up: customer message, photos, decision change, AI reply.
    rank = {"customer": 0, "photo": 1, "decision": 2, "assistant": 3}
    events = (
        [(m["role"], m) for m in detail["messages"]]
        + [("photo", p) for p in detail["photos"]]
        + [("decision", d) for d in detail["decisions"]]
    )
    events.sort(key=lambda e: (datetime.fromisoformat(e[1]["created_at"]), rank[e[0]]))

    st.markdown("**Conversation**")
    first_decision = detail["decisions"][0]["created_at"] if detail["decisions"] else None
    for kind, item in events:
        if kind == "customer":
            st.markdown(f"🗨️ **Customer** · {_when(item['created_at'])}")
            st.write(item["body"])
        elif kind == "assistant":
            with st.container(border=True):
                st.markdown(f"🤖 **PolicyPilot** · {_when(item['created_at'])}")
                st.write(item["body"])
        elif kind == "photo":
            data = _photo_bytes(st.session_state["token"], detail["id"], item["id"])
            left, right = st.columns([1, 2])
            if data:
                left.image(data, caption=item["original_filename"], width="stretch")
            else:
                left.caption(f"({item['original_filename']} could not be loaded)")
            right.markdown(f"📷 **Photo evidence** · {_when(item['created_at'])}")
            right.caption(
                " · ".join(
                    [
                        _flag(item["is_clear"], "Clear"),
                        _flag(item["is_relevant"], "Relevant"),
                        _flag(item["shows_issue"], "Shows the issue"),
                    ]
                )
            )
            right.write(item["analysis"])
        else:
            label = "Initial decision" if item["created_at"] == first_decision else "Decision updated"
            st.markdown(
                f"📋 **{label}** · {_when(item['created_at'])} · "
                f"`{item['action']}` — {item['reason']}"
            )


def _submit_follow_up(ticket_id: int, message: str, uploads: list[Any]) -> None:
    files = [("photos", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
    with st.spinner("Reassessing the ticket against the policies..."):
        ok, data = api(
            "POST",
            f"/tickets/{ticket_id}/follow-ups",
            data={"message": message} if message else {},
            files=files or None,
            timeout=180,  # photo analysis + reassessment, with rate-limit backoff
        )
    if ok:
        st.session_state["flash"] = (ticket_id, "Ticket updated with your new information.")
        st.rerun()
    else:
        st.error(data)


def render_ticket(detail: dict[str, Any], key_prefix: str) -> None:
    ticket_id = detail["id"]
    key = f"{key_prefix}-{ticket_id}"

    flash = st.session_state.get("flash")
    if flash and flash[0] == ticket_id:
        st.success(flash[1])  # cleared by main() once both tabs have rendered

    st.markdown(f"**Ticket #{ticket_id}**")
    st.write(detail["message"])

    facts = {
        "Order value (₹)": detail["order_value_inr"],
        "Days since delivery": detail["days_since_delivery"],
        "Days since dispatch": detail["days_since_dispatch"],
        "Product type": detail["product_type"],
        "Opened": detail["opened_status"],
        "Order status": detail["order_status"],
    }
    # Rendered as markdown rather than st.table/st.dataframe on purpose:
    # those pull in pandas, which is a heavy import for six key-value
    # pairs and fails outright on machines where an application-control
    # policy blocks its compiled DLLs.
    st.markdown("**Ticket details**")
    st.markdown(
        "\n".join(
            f"- {label}: {'—' if value is None else value}" for label, value in facts.items()
        )
    )

    if detail["messages"] or detail["photos"] or len(detail["decisions"]) > 1:
        st.divider()
        render_timeline(detail)

    st.divider()
    st.markdown("**Current decision**")
    decision = detail.get("decision")
    render_decision(decision)

    if decision and decision["action"] in EITHER_OR:
        preference = detail.get("preferred_resolution")
        if preference:
            st.markdown(
                f"**Customer preference:** {preference.title()} — recorded, not yet processed. "
                "The policy doesn't describe processing times or next steps."
            )
        else:
            st.caption("Eligible for a refund or a replacement. Tell us below which you'd prefer.")

    # Photo upload appears only when the policy is actually asking for evidence.
    needed = EVIDENCE_REQUESTS.get(decision["action"]) if decision else None
    if needed:
        st.info(f"📷 **Photos needed.** Please upload {needed}. JPEG, PNG or WEBP, up to 5 MB each.")
        with st.form(f"{key}-photos", clear_on_submit=True):
            uploads = st.file_uploader(
                "Photos",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key=f"{key}-uploader",
            )
            note = st.text_input("Note about the photos (optional)", key=f"{key}-note")
            if st.form_submit_button("Submit Photos", width="stretch"):
                if uploads:
                    _submit_follow_up(ticket_id, note.strip(), uploads)
                else:
                    st.error("Choose at least one photo first.")

    with st.form(f"{key}-follow-up", clear_on_submit=True):
        text = st.text_area(
            "Ask a question, answer the one above, or add information",
            placeholder="e.g. When will I get my refund?  ·  I'd prefer a replacement.  ·  It was delivered 2 days ago.",
            height=90,
            key=f"{key}-text",
        )
        if st.form_submit_button("Submit follow-up", width="stretch"):
            if text.strip():
                _submit_follow_up(ticket_id, text.strip(), [])
            else:
                st.error("Write a message first.")


# --------------------------------------------------------------------------
# New Decision
# --------------------------------------------------------------------------

UNKNOWN = "unknown / not sure"


def _optional(value: str) -> str | None:
    return None if value == UNKNOWN else value


def render_new_decision() -> None:
    st.header("New Decision")
    st.caption("Describe the issue. Leave anything you do not know as 'unknown'.")

    with st.form("ticket"):
        message = st.text_area(
            "What is the customer's problem?",
            placeholder="My ₹3,500 order arrived damaged yesterday.",
            height=110,
        )

        # Numbers start empty and empty means "unknown". An earlier version used
        # "known?" checkboxes, but inside st.form a checkbox does not rerun the
        # page, so the number inputs only appeared after a first submit - and
        # the order value silently defaulted to 1000, overriding the message.
        col1, col2 = st.columns(2)
        order_value = col1.number_input(
            "Order value (₹)", min_value=0.0, step=100.0, value=None, placeholder="unknown"
        )
        order_status = col2.selectbox(
            "Order status", [UNKNOWN, "processing", "dispatched", "delivered"]
        )

        col3, col4 = st.columns(2)
        product_type = col3.selectbox("Product type", [UNKNOWN, "non_food", "food", "mixed"])
        opened_status = col4.selectbox("Opened?", [UNKNOWN, "unopened", "opened"])

        col5, col6 = st.columns(2)
        days_since_delivery = col5.number_input(
            "Days since delivery", min_value=0, step=1, value=None, placeholder="unknown"
        )
        days_since_dispatch = col6.number_input(
            "Days since dispatch", min_value=0, step=1, value=None, placeholder="unknown"
        )

        submitted = st.form_submit_button("Get recommendation", width="stretch")

    if submitted:
        if not message.strip():
            st.error("Please describe the problem first.")
        else:
            payload = {
                "message": message.strip(),
                "order_value_inr": order_value,
                "days_since_delivery": days_since_delivery,
                "days_since_dispatch": days_since_dispatch,
                "product_type": _optional(product_type),
                "opened_status": _optional(opened_status),
                "order_status": _optional(order_status),
            }
            with st.spinner("Checking the policies..."):
                ok, data = api("POST", "/tickets", json=payload)
            if ok:
                # Keep the ticket on screen across reruns so the customer can
                # continue the conversation on it.
                st.session_state["active_ticket"] = data["id"]
                st.session_state["flash"] = (data["id"], f"Ticket #{data['id']} created.")
            else:
                st.error(data)

    active = st.session_state.get("active_ticket")
    if active:
        st.divider()
        ok, detail = api("GET", f"/tickets/{active}")
        if ok:
            render_ticket(detail, key_prefix="new")
        else:
            st.session_state.pop("active_ticket", None)
            st.error(detail)


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def render_history() -> None:
    st.header("History")

    ok, tickets = api("GET", "/tickets")
    if not ok:
        st.error(tickets)
        return

    if not tickets:
        st.info("No tickets yet. Submit one from the New Decision tab.")
        return

    st.caption(f"{len(tickets)} ticket(s).")

    for ticket in tickets:
        action = (ticket.get("action") or "NO DECISION").replace("_", " ").title()
        preview = ticket["message"][:60] + ("..." if len(ticket["message"]) > 60 else "")

        # Re-open the ticket the customer just updated; its label changes with
        # the new decision, so Streamlit would otherwise render it collapsed.
        just_updated = (st.session_state.get("flash") or (None,))[0] == ticket["id"]
        with st.expander(f"#{ticket['id']} — {action} — {preview}", expanded=just_updated):
            ok, detail = api("GET", f"/tickets/{ticket['id']}")
            if not ok:
                st.error(detail)
                continue
            render_ticket(detail, key_prefix="hist")


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------


def main() -> None:
    if not logged_in():
        render_auth()
        return

    with st.sidebar:
        st.markdown("### 🧭 PolicyPilot AI")
        ok, user = api("GET", "/me")
        st.caption(f"Signed in as **{user['email']}**" if ok else "Signed in")
        if st.button("Log out", width="stretch"):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.caption(f"API: `{settings.api_base_url}`")

    new_tab, history_tab = st.tabs(["New Decision", "History"])
    with new_tab:
        render_new_decision()
    with history_tab:
        render_history()
    st.session_state.pop("flash", None)


main()
