"""Five extraction schemas used by the eval. Mix of strings, ints with ranges,
floats, enums, booleans, string arrays and arrays of nested objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ContactCard(BaseModel):
    name: str
    email: str
    phone: str
    company: str


class LineItem(BaseModel):
    description: str
    quantity: int = Field(ge=1, le=100)
    unit_price: float


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    currency: Literal["USD", "EUR", "GBP"]
    total: float
    line_items: list[LineItem] = Field(min_length=1, max_length=5)


class Event(BaseModel):
    title: str
    date: str
    attendees: list[str] = Field(max_length=6)
    is_virtual: bool


class SupportTicket(BaseModel):
    customer_id: str
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(max_length=120)


class ProductReview(BaseModel):
    product: str
    rating: int = Field(ge=1, le=5)
    pros: list[str] = Field(max_length=4)
    cons: list[str] = Field(max_length=4)
    would_recommend: bool


SCHEMAS: dict[str, type[BaseModel]] = {
    m.__name__: m for m in (ContactCard, Invoice, Event, SupportTicket, ProductReview)
}
