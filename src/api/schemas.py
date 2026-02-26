from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class TopMatch(BaseModel):
    name: str
    set_code: str
    confidence: float


class RecognizedCard(BaseModel):
    name: str
    set_code: str
    collector_number: str
    confidence: float
    top_matches: list[TopMatch]


class RecognizeResponse(BaseModel):
    cards: list[RecognizedCard]


class CardMetadataItem(BaseModel):
    image_path: str
    name: str
    set: str
    set_code: str
    collector_number: str
    mana_cost: str | None = None
    type_line: str | None = None
    oracle_text: str | None = None


class CardsResponse(BaseModel):
    cards: list[CardMetadataItem]
