"""ORM models: users, tickets, decisions (schema per assignment section 6)."""

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
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

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="tickets")
    decision: Mapped["Decision | None"] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", uselist=False
    )


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sources: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship(back_populates="decision")
