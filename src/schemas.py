"""Pydantic request/response models. These are the API's trust boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

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
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    sources: list[str]
    created_at: datetime


class TicketSummary(BaseModel):
    """Row shape for the History list."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    message: str
    created_at: datetime
    action: Action | None = None
    confidence: float | None = None


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
    created_at: datetime
    decision: DecisionOut | None = None


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
