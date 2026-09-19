"""Pydantic request/response models. These are the API's trust boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, PrivateAttr, field_validator

from src.actions import Action

ProductType = Literal["food", "non_food", "mixed", "unknown"]
OpenedStatus = Literal["opened", "unopened", "unknown"]
OrderStatus = Literal["processing", "dispatched", "delivered", "unknown"]

# bcrypt silently truncates anything past 72 bytes, so reject it up front
# rather than hashing only a prefix.
BCRYPT_MAX_BYTES = 72


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=BCRYPT_MAX_BYTES)

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, v: str) -> str:
        if len(v.encode("utf-8")) > BCRYPT_MAX_BYTES:
            raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Note the absence of password_hash - it must never leave the server."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class TicketCreate(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    order_value_inr: float | None = Field(default=None, ge=0)
    days_since_delivery: int | None = Field(default=None, ge=0, le=3650)
    days_since_dispatch: int | None = Field(default=None, ge=0, le=3650)
    product_type: ProductType | None = None
    opened_status: OpenedStatus | None = None
    order_status: OrderStatus | None = None


class DecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)  # the model's self-rating; not a probability
    reason: str
    sources: list[str]
    basis: str = "clear"  # awaiting_customer | clear | review
    basis_reasons: list[str] = Field(default_factory=list)
    created_at: datetime


class TicketSummary(BaseModel):
    """Row shape for the History list."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    message: str
    created_at: datetime
    action: Action | None = None
    confidence: float | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["customer", "assistant"]
    body: str
    created_at: datetime


class PhotoOut(BaseModel):
    """Photo metadata for clients. Deliberately omits the on-disk filename:
    the image is only reachable through the owner-checked download endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    content_type: str
    size_bytes: int
    analysis: str
    is_clear: bool
    is_relevant: bool
    shows_issue: bool
    created_at: datetime


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message: str
    order_value_inr: float | None
    days_since_delivery: int | None
    days_since_dispatch: int | None
    product_type: str | None
    opened_status: str | None
    order_status: str | None
    preferred_resolution: str | None = None  # recorded preference, not a completed action
    created_at: datetime
    decision: DecisionOut | None = None  # the current (latest) decision
    decisions: list[DecisionOut] = Field(default_factory=list)  # full history, oldest first
    messages: list[MessageOut] = Field(default_factory=list)
    photos: list[PhotoOut] = Field(default_factory=list)


class PhotoAnalysis(BaseModel):
    """What the vision model reports about one photo - observation only."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=1000)
    is_clear: bool       # in focus, lit, subject visible
    is_relevant: bool    # shows the product/packaging the ticket is about
    shows_issue: bool    # the reported problem (e.g. damage) is visibly present


class LLMDecision(BaseModel):
    """What the model is required to return. Validated before it is persisted.

    `extra="forbid"` means a hallucinated extra key is a validation error rather
    than something silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=2000)
    sources: list[str] = Field(default_factory=list)

    # Set by grounding when the model cited no usable policy and the sources
    # were filled in from what it was shown. Private: never part of the JSON
    # the model sees or the API returns - it only feeds decision_basis().
    _sources_filled_in: bool = PrivateAttr(default=False)


class FollowUpResult(LLMDecision):
    """A reassessment on a continuing ticket: the decision plus a reply to the customer.

    Only follow-ups use this; the first decision on a ticket (and evaluate.py)
    still uses plain LLMDecision, so the assignment's output schema is unchanged.
    """

    reply: str = Field(min_length=1, max_length=2000)
    customer_preference: Literal["refund", "replacement", "none"] = "none"
