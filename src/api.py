"""FastAPI application: auth + ticket decision endpoints."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.auth import (
    CurrentUser,
    DbSession,
    create_access_token,
    hash_password,
    verify_password,
)
from src.database import init_db
from src.decision import DecisionUnavailableError, generate_decision
from src.models import Decision, Ticket, User
from src.schemas import (
    TicketCreate,
    TicketOut,
    TicketSummary,
    Token,
    UserCreate,
    UserLogin,
    UserOut,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="PolicyPilot AI",
    description="AI-powered support ticket decision assistant.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@app.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED, tags=["auth"])
def register(payload: UserCreate, db: DbSession) -> User:
    user = User(email=payload.email.lower(), password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        ) from None
    db.refresh(user)
    return user


@app.post("/login", response_model=Token, tags=["auth"])
def login(payload: UserLogin, db: DbSession) -> Token:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))

    # Same response for "no such user" and "wrong password" so the endpoint
    # cannot be used to enumerate registered emails.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    return Token(access_token=create_access_token(user.id))


@app.get("/me", response_model=UserOut, tags=["auth"])
def me(current_user: CurrentUser) -> User:
    return current_user


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------


@app.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED, tags=["tickets"])
def create_ticket(payload: TicketCreate, current_user: CurrentUser, db: DbSession) -> Ticket:
    """Submit a ticket, run it through retrieval + the LLM, and persist both."""
    ticket = Ticket(user_id=current_user.id, **payload.model_dump())

    try:
        result = generate_decision(payload)
    except DecisionUnavailableError as exc:
        # Nothing is persisted: a ticket with no decision would be a dead row,
        # and inventing a decision is exactly what the spec forbids.
        logger.exception("decision pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Decision service unavailable: {exc}",
        ) from exc

    ticket.decision = Decision(
        action=result.action.value,
        confidence=result.confidence,
        reason=result.reason,
        sources=result.sources,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/tickets", response_model=list[TicketSummary], tags=["tickets"])
def list_tickets(current_user: CurrentUser, db: DbSession) -> list[TicketSummary]:
    tickets = db.scalars(
        select(Ticket)
        .where(Ticket.user_id == current_user.id)
        .order_by(Ticket.created_at.desc(), Ticket.id.desc())
    ).all()

    return [
        TicketSummary(
            id=t.id,
            message=t.message,
            created_at=t.created_at,
            action=t.decision.action if t.decision else None,
            confidence=t.decision.confidence if t.decision else None,
        )
        for t in tickets
    ]


@app.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"])
def get_ticket(ticket_id: int, current_user: CurrentUser, db: DbSession) -> Ticket:
    # Ownership is part of the WHERE clause, not a check after the fetch, so
    # there is no code path that loads another user's ticket at all.
    ticket = db.scalar(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.user_id == current_user.id)
    )
    if ticket is None:
        # 404 rather than 403: a 403 would confirm the ticket exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket
