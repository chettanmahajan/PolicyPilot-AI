"""FastAPI application: auth + ticket decision endpoints."""

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth import (
    CurrentUser,
    DbSession,
    create_access_token,
    hash_password,
    verify_password,
)
from src.config import settings
from src.database import init_db
from src.decision import (
    DecisionUnavailableError,
    PhotoEvidence,
    EITHER_OR_ACTIONS,
    TicketHistory,
    decision_basis,
    generate_decision,
    generate_follow_up,
)
from src.evidence import (
    PhotoRejectedError,
    analyze_photos,
    delete_photos,
    read_photo,
    save_photo,
    validate_photo,
)
from src.models import Decision, Ticket, TicketMessage, TicketPhoto, User
from src.schemas import (
    LLMDecision,
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

    ticket.decisions.append(_decision_row(result))
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


def _owned_ticket(db: Session, ticket_id: int, user: User) -> Ticket:
    """Load a ticket only if it belongs to `user`; every ticket route goes through here.

    Ownership is part of the WHERE clause, not a check after the fetch, so
    there is no code path that loads another user's ticket at all.
    """
    ticket = db.scalar(select(Ticket).where(Ticket.id == ticket_id, Ticket.user_id == user.id))
    if ticket is None:
        # 404 rather than 403: a 403 would confirm the ticket exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket


def _decision_row(
    result: LLMDecision, created_at: datetime | None = None, *, changed_by_statement: bool = False
) -> Decision:
    basis, basis_reasons = decision_basis(result, changed_by_statement=changed_by_statement)
    row = Decision(
        action=result.action.value,
        confidence=result.confidence,
        reason=result.reason,
        sources=result.sources,
        basis=basis,
        basis_reasons=basis_reasons,
    )
    if created_at is not None:
        row.created_at = created_at
    return row


def _ticket_facts(ticket: Ticket) -> TicketCreate:
    """The ticket's original structured facts, as the decision engine expects them."""
    return TicketCreate(
        message=ticket.message,
        order_value_inr=ticket.order_value_inr,
        days_since_delivery=ticket.days_since_delivery,
        days_since_dispatch=ticket.days_since_dispatch,
        product_type=ticket.product_type,
        opened_status=ticket.opened_status,
        order_status=ticket.order_status,
    )


@app.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"])
def get_ticket(ticket_id: int, current_user: CurrentUser, db: DbSession) -> Ticket:
    return _owned_ticket(db, ticket_id, current_user)


@app.post(
    "/tickets/{ticket_id}/follow-ups",
    response_model=TicketOut,
    status_code=status.HTTP_201_CREATED,
    tags=["tickets"],
)
def add_follow_up(
    ticket_id: int,
    current_user: CurrentUser,
    db: DbSession,
    message: Annotated[str | None, Form(max_length=4000)] = None,
    photos: Annotated[list[UploadFile] | None, File()] = None,
) -> Ticket:
    """Continue a ticket: a message, photos, or both. The ticket is then reassessed.

    Order of work is deliberate: validate everything, run both AI calls, and
    only then write files and rows. A failure at any point leaves no rows and
    no files behind - the same all-or-nothing rule as ticket creation.
    """
    ticket = _owned_ticket(db, ticket_id, current_user)
    text = (message or "").strip()
    uploads = [p for p in (photos or []) if p.filename or p.size]

    if not text and not uploads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Send a message, at least one photo, or both.",
        )
    if len(uploads) > settings.max_photos_per_upload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Upload at most {settings.max_photos_per_upload} photos at a time.",
        )

    try:
        validated = [validate_photo(upload.file, upload.filename) for upload in uploads]
    except PhotoRejectedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None

    conversation = [(m.role, m.body) for m in ticket.messages] + ([("customer", text)] if text else [])
    customer_said = [body for role, body in conversation if role == "customer"]
    try:
        analyses = (
            analyze_photos(validated, complaint="\n".join([ticket.message, *customer_said]))
            if validated
            else []
        )
        history = TicketHistory(
            prior_decisions=tuple((d.action, d.reason) for d in ticket.decisions),
            conversation=tuple(conversation),
            latest_message=text,
            new_photos=len(validated),
            preferred_resolution=ticket.preferred_resolution,
            photos=tuple(
                [
                    PhotoEvidence(p.original_filename, p.analysis, p.is_clear, p.is_relevant, p.shows_issue)
                    for p in ticket.photos
                ]
                + [
                    PhotoEvidence(v.original_filename, a.description, a.is_clear, a.is_relevant, a.shows_issue)
                    for v, a in zip(validated, analyses, strict=True)
                ]
            ),
        )
        result = generate_follow_up(_ticket_facts(ticket), history)
    except DecisionUnavailableError as exc:
        logger.exception("reassessment failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Decision service unavailable: {exc}",
        ) from exc

    # One timestamp for everything in this follow-up, so the conversation reads
    # customer message -> photos -> decision change -> AI reply regardless of
    # insert order (the UI orders same-timestamp items by kind).
    now = datetime.now(UTC)
    current = ticket.decision
    action_changed = current is None or result.action.value != current.action
    stored: list[str] = []
    try:
        if text:
            ticket.messages.append(TicketMessage(role="customer", body=text, created_at=now))
        for photo, analysis in zip(validated, analyses, strict=True):
            stored_name = save_photo(photo)
            stored.append(stored_name)
            ticket.photos.append(
                TicketPhoto(
                    stored_name=stored_name,
                    original_filename=photo.original_filename,
                    content_type=photo.content_type,
                    size_bytes=len(photo.data),
                    sha256=photo.sha256,
                    analysis=analysis.description,
                    is_clear=analysis.is_clear,
                    is_relevant=analysis.is_relevant,
                    shows_issue=analysis.shows_issue,
                    created_at=now,
                )
            )
        # A decision row is added only when the action actually changes. A
        # question ("when is my refund?") gets a reply, not a duplicate decision.
        if action_changed:
            ticket.decisions.append(
                # A change on a text-only turn rests on the customer's own
                # statements rather than ticket data or photo evidence.
                _decision_row(result, created_at=now, changed_by_statement=not validated)
            )
        ticket.messages.append(TicketMessage(role="assistant", body=result.reply, created_at=now))

        # Record a stated preference only where the decision actually offers
        # a choice, and only from something the customer wrote this turn.
        # "none" never erases an earlier preference.
        if (
            text
            and result.action in EITHER_OR_ACTIONS
            and result.customer_preference in ("refund", "replacement")
        ):
            ticket.preferred_resolution = result.customer_preference
        db.commit()
    except Exception:
        db.rollback()
        delete_photos(stored)
        raise

    db.refresh(ticket)
    return ticket


@app.get("/tickets/{ticket_id}/photos/{photo_id}", tags=["tickets"], response_class=Response)
def get_photo(ticket_id: int, photo_id: int, current_user: CurrentUser, db: DbSession) -> Response:
    """Stream one evidence photo to its owner. There is no other way to reach the file."""
    ticket = _owned_ticket(db, ticket_id, current_user)
    photo = db.scalar(
        select(TicketPhoto).where(TicketPhoto.id == photo_id, TicketPhoto.ticket_id == ticket.id)
    )
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found")
    try:
        data = read_photo(photo.stored_name)
    except OSError:
        logger.exception("photo %s is missing from storage", photo.id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found") from None

    return Response(
        content=data,
        media_type=photo.content_type,
        headers={
            "X-Content-Type-Options": "nosniff",  # browsers must not reinterpret it
            "Cache-Control": "private, no-store",
            "Content-Disposition": "inline",
        },
    )
