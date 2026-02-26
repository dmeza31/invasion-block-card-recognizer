from __future__ import annotations

import json
import logging
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from api.schemas import (
    CardMetadataItem,
    CardsResponse,
    HealthResponse,
    RecognizeResponse,
    RecognizedCard,
    TopMatch,
)
from recognizer.detector import CardDetector
from recognizer.recognizer import CardRecognizer

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

INDEX_PATH = Path("data/embeddings/invasion_block.index")
METADATA_PATH = Path("data/embeddings/card_metadata.json")
CATALOG_PATH = Path("data/card_metadata.json")
REFERENCE_IMAGES_PATH = Path("data/reference_images")
DETECTIONS_OUTPUT_PATH = Path("data/detections")

app = FastAPI(title="MTG Invasion Recognizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence_from_similarity(similarity: float) -> float:
    return max(0.0, min(1.0, float(similarity)))


def _load_card_catalog(catalog_path: Path) -> list[CardMetadataItem]:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Card catalog not found: {catalog_path.as_posix()}")

    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Card catalog JSON must be a dictionary keyed by image path")

    catalog: list[CardMetadataItem] = []

    for image_path, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        image_path_str = str(image_path)
        image_path_obj = Path(image_path_str)
        derived_set_code = image_path_obj.parent.name.lower() if image_path_obj.parent.name else ""

        item = CardMetadataItem(
            image_path=image_path_str,
            name=str(payload.get("name", "")),
            set=str(payload.get("set", "")),
            set_code=str(payload.get("set_code", derived_set_code) or derived_set_code).lower(),
            collector_number=str(payload.get("collector_number", "")),
            mana_cost=(str(payload["mana_cost"]) if payload.get("mana_cost") is not None else None),
            type_line=(str(payload["type_line"]) if payload.get("type_line") is not None else None),
            oracle_text=(str(payload["oracle_text"]) if payload.get("oracle_text") is not None else None),
        )
        catalog.append(item)

    return catalog


@app.on_event("startup")
def startup_event() -> None:
    app.state.card_detector = CardDetector()
    app.state.card_catalog = None

    try:
        app.state.card_catalog = _load_card_catalog(CATALOG_PATH)
    except Exception:
        logger.exception("Failed to load card catalog from %s", CATALOG_PATH.as_posix())

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        logger.warning(
            "Recognizer artifacts not found at startup (index=%s, metadata=%s). "
            "Endpoints will return 500 until artifacts exist.",
            INDEX_PATH.as_posix(),
            METADATA_PATH.as_posix(),
        )
        app.state.card_recognizer = None
        return

    app.state.card_recognizer = CardRecognizer(
        index_path=INDEX_PATH.as_posix(),
        metadata_path=METADATA_PATH.as_posix(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/cards", response_model=CardsResponse)
def list_cards(
    set_code: str | None = Query(default=None),
    name: str | None = Query(default=None),
) -> CardsResponse:
    card_catalog: list[CardMetadataItem] | None = getattr(app.state, "card_catalog", None)
    if card_catalog is None:
        raise HTTPException(status_code=500, detail="Card metadata catalog is not initialized")

    normalized_set = set_code.strip().lower() if set_code else None
    normalized_name = name.strip().lower() if name else None

    filtered = card_catalog
    if normalized_set:
        filtered = [item for item in filtered if item.set_code.lower() == normalized_set]
    if normalized_name:
        filtered = [item for item in filtered if normalized_name in item.name.lower()]

    return CardsResponse(cards=filtered)


@app.get("/cards/{set_code}/{collector_number}/image")
def get_card_image(set_code: str, collector_number: str) -> FileResponse:
    set_folder = REFERENCE_IMAGES_PATH / set_code.lower()
    if not set_folder.exists():
        raise HTTPException(status_code=404, detail="Card image not found")

    matches = sorted(set_folder.glob(f"{collector_number}_*"))
    image_candidates = [path for path in matches if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not image_candidates:
        raise HTTPException(status_code=404, detail="Card image not found")

    image_path = image_candidates[0]
    media_type = "image/jpeg"
    if image_path.suffix.lower() == ".png":
        media_type = "image/png"

    return FileResponse(path=image_path, media_type=media_type, filename=image_path.name)


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB")

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            rgb = uploaded.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        logger.exception("Failed to parse uploaded image")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    card_recognizer: CardRecognizer | None = getattr(app.state, "card_recognizer", None)
    card_detector: CardDetector | None = getattr(app.state, "card_detector", None)

    if card_recognizer is None or card_detector is None:
        raise HTTPException(status_code=500, detail="Recognizer is not initialized")

    try:
        image_rgb = np.array(rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detected_cards = card_detector.detect_and_crop(image_bgr)

        cards: list[RecognizedCard] = []

        for detected_bgr in detected_cards:
            detected_rgb = cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)
            detected_pil = Image.fromarray(detected_rgb)

            predictions = card_recognizer.recognize(detected_pil, top_k=5)
            if not predictions:
                continue

            top_prediction = predictions[0]
            top_matches = [
                TopMatch(
                    name=str(match.get("name", "")),
                    set_code=str(match.get("set_code", "")),
                    confidence=_confidence_from_similarity(float(match.get("similarity_score", 0.0))),
                )
                for match in predictions
            ]

            cards.append(
                RecognizedCard(
                    name=str(top_prediction.get("name", "")),
                    set_code=str(top_prediction.get("set_code", "")),
                    collector_number=str(top_prediction.get("collector_number", "")),
                    confidence=_confidence_from_similarity(
                        float(top_prediction.get("similarity_score", 0.0))
                    ),
                    top_matches=top_matches,
                )
            )

        return RecognizeResponse(cards=cards)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Recognition pipeline failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> FileResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB")

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            rgb = uploaded.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        logger.exception("Failed to parse uploaded image")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    card_detector: CardDetector | None = getattr(app.state, "card_detector", None)
    if card_detector is None:
        raise HTTPException(status_code=500, detail="Detector is not initialized")

    try:
        image_rgb = np.array(rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detected_cards = card_detector.detect_and_crop(image_bgr)

        detected_bgr = detected_cards[0]
        DETECTIONS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        original_stem = Path(file.filename or "upload").stem
        output_name = f"detected_{original_stem}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = DETECTIONS_OUTPUT_PATH / output_name

        if not cv2.imwrite(output_path.as_posix(), detected_bgr):
            raise HTTPException(status_code=500, detail="Failed to write detected image")

        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename=output_name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Detection pipeline failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
