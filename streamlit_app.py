"""PolicyPilot AI - Streamlit frontend.

Talks to the FastAPI backend over HTTP only. It holds no database connection
and repeats none of the backend's decision or authorization logic.
"""

from __future__ import annotations

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
    try:
        response = requests.request(
            method,
            f"{settings.api_base_url}{path}",
            headers=_headers(),
            timeout=TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the API at {settings.api_base_url} ({exc})."

    if response.status_code == 401:
        st.session_state.pop("token", None)
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
            if st.form_submit_button("Log in", use_container_width=True):
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
            if st.form_submit_button("Create account", use_container_width=True):
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

    confidence = float(decision["confidence"])
    left, right = st.columns([3, 1])
    left.progress(confidence, text=f"Confidence {confidence:.0%}")
    if decision["action"] == "NEEDS_MORE_INFORMATION":
        right.info("More info")
    elif confidence >= 0.75:
        right.success("High")
    else:
        right.warning("Low")

    st.markdown("**Reason**")
    st.write(decision["reason"])

    st.markdown("**Policy sources**")
    if decision["sources"]:
        for source in decision["sources"]:
            st.markdown(f"- `{source}`")
    else:
        st.caption("No policy source cited.")


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

        col1, col2 = st.columns(2)
        known_value = col1.checkbox("Order value known", value=True)
        order_value = (
            col1.number_input("Order value (₹)", min_value=0.0, step=100.0, value=1000.0)
            if known_value
            else None
        )
        order_status = col2.selectbox(
            "Order status", [UNKNOWN, "processing", "dispatched", "delivered"]
        )

        col3, col4 = st.columns(2)
        product_type = col3.selectbox("Product type", [UNKNOWN, "non_food", "food", "mixed"])
        opened_status = col4.selectbox("Opened?", [UNKNOWN, "unopened", "opened"])

        col5, col6 = st.columns(2)
        known_delivery = col5.checkbox("Days since delivery known")
        days_since_delivery = (
            col5.number_input("Days since delivery", min_value=0, step=1, value=1)
            if known_delivery
            else None
        )
        known_dispatch = col6.checkbox("Days since dispatch known")
        days_since_dispatch = (
            col6.number_input("Days since dispatch", min_value=0, step=1, value=1)
            if known_dispatch
            else None
        )

        submitted = st.form_submit_button("Get recommendation", use_container_width=True)

    if not submitted:
        return

    if not message.strip():
        st.error("Please describe the problem first.")
        return

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

    if not ok:
        st.error(data)
        return

    st.success(f"Ticket #{data['id']} created.")
    render_decision(data.get("decision"))


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

        with st.expander(f"#{ticket['id']} — {action} — {preview}"):
            ok, detail = api("GET", f"/tickets/{ticket['id']}")
            if not ok:
                st.error(detail)
                continue

            st.markdown("**Ticket**")
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
                    f"- {label}: {'—' if value is None else value}"
                    for label, value in facts.items()
                )
            )

            st.divider()
            render_decision(detail.get("decision"))


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
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.caption(f"API: `{settings.api_base_url}`")

    new_tab, history_tab = st.tabs(["New Decision", "History"])
    with new_tab:
        render_new_decision()
    with history_tab:
        render_history()


main()
