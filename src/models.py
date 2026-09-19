"""ORM models: users, tickets, decisions (schema per assignment section 6)."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Structured ticket facts. All optional: a customer may not supply them,
    # and a missing value is exactly what should drive NEEDS_MORE_INFORMATION.
    order_value_inr: Mapped[float | None] = mapped_column(Float)
    days_since_delivery: Mapped[int | None] = mapped_column(Integer)
    days_since_dispatch: Mapped[int | None] = mapped_column(Integer)
    product_type: Mapped[str | None] = mapped_column(String(32))
    opened_status: Mapped[str | None] = mapped_column(String(32))
    order_status: Mapped[str | None] = mapped_column(String(32))

    # "refund" / "replacement" - what the customer said they want when the
    # decision offers either. A recorded preference, never a completed action:
    # nothing in this system issues refunds or ships replacements.
    preferred_resolution: Mapped[str | None] = mapped_column(String(16))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="tickets")

    # A ticket accumulates decisions as the conversation continues; earlier
    # ones are kept, never overwritten, so the history shows how it evolved.
    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="Decision.id"
    )
    messages: Mapped[list["TicketMessage"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="TicketMessage.id"
    )
    photos: Mapped[list["TicketPhoto"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="TicketPhoto.id"
    )

    @property
    def decision(self) -> "Decision | None":
        """The current decision: the most recent one."""
        return self.decisions[-1] if self.decisions else None


class TicketMessage(Base):
    """One turn of the ticket conversation: a customer follow-up or the AI's reply."""

    __tablename__ = "ticket_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="customer")  # customer | assistant
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship(back_populates="messages")


class TicketPhoto(Base):
    """Metadata for an uploaded evidence photo.

    The image bytes live on disk under settings.uploads_dir, not in SQLite.
    `stored_name` is a random server-generated filename and is never sent to
    clients; `original_filename` is kept only for display.
    """

    __tablename__ = "ticket_photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    stored_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    # What the vision model could actually see - never a verdict on the claim.
    analysis: Mapped[str] = mapped_column(Text, nullable=False)
    is_clear: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_relevant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    shows_issue: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship(back_populates="photos")


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # model self-rating, uncalibrated
    sources: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # How much weight the decision can bear, from explicit rules rather than the
    # model's number - see decision.decision_basis().
    basis: Mapped[str] = mapped_column(String(24), nullable=False, default="clear")
    basis_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship(back_populates="decisions")
