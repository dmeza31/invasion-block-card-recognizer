from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
SET_CODES = ["inv", "pls", "apc"]
REQUEST_DELAY_SECONDS = 0.1


def sanitize_filename(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe filename fragment.

    Replaces unsupported characters with underscores and ensures a non-empty
    fallback value is returned.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def get_image_url(card: dict[str, Any]) -> str | None:
    """Extract the best available normal-sized image URL for a Scryfall card.

    Uses top-level ``image_uris.normal`` when present, otherwise falls back to
    the first face in ``card_faces`` containing ``image_uris.normal``.
    """
    image_uris = card.get("image_uris")
    if isinstance(image_uris, dict):
        normal_url = image_uris.get("normal")
        if isinstance(normal_url, str):
            return normal_url

    card_faces = card.get("card_faces")
    if isinstance(card_faces, list):
        for face in card_faces:
            if not isinstance(face, dict):
                continue
            face_image_uris = face.get("image_uris")
            if not isinstance(face_image_uris, dict):
                continue
            normal_url = face_image_uris.get("normal")
            if isinstance(normal_url, str):
                return normal_url

    return None


def build_card_metadata(card: dict[str, Any]) -> dict[str, Any]:
    """Build normalized metadata fields for a downloaded card image.

    For double-faced cards, missing top-level values are derived by combining
    values from ``card_faces``.
    """
    name = card.get("name", "")
    set_name = card.get("set_name", card.get("set", "")).strip()
    collector_number = str(card.get("collector_number", "")).strip()

    mana_cost = card.get("mana_cost")
    type_line = card.get("type_line")
    oracle_text = card.get("oracle_text")

    card_faces = card.get("card_faces")
    if isinstance(card_faces, list) and card_faces:
        if not mana_cost:
            face_costs = [face.get("mana_cost", "") for face in card_faces if isinstance(face, dict)]
            mana_cost = " // ".join(cost for cost in face_costs if cost)
        if not type_line:
            face_types = [face.get("type_line", "") for face in card_faces if isinstance(face, dict)]
            type_line = " // ".join(type_name for type_name in face_types if type_name)
        if not oracle_text:
            face_text = [face.get("oracle_text", "") for face in card_faces if isinstance(face, dict)]
            oracle_text = "\n//\n".join(text for text in face_text if text)

    return {
        "name": name,
        "set": set_name,
        "collector_number": collector_number,
        "mana_cost": mana_cost or "",
        "type_line": type_line or "",
        "oracle_text": oracle_text or "",
    }


class RateLimitedSession:
    """HTTP session wrapper that enforces a minimum delay between requests."""

    def __init__(self, delay_seconds: float = REQUEST_DELAY_SECONDS) -> None:
        """Initialize a session with the provided inter-request delay."""
        self.delay_seconds = delay_seconds
        self._last_request_time = 0.0
        self.session = requests.Session()

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Send an HTTP request while respecting configured rate limits.

        Sleeps as needed so requests are spaced by at least ``delay_seconds``.
        Raises for non-success HTTP responses.
        """
        elapsed = time.monotonic() - self._last_request_time
        sleep_for = self.delay_seconds - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        response = self.session.request(method, url, timeout=30, **kwargs)
        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response


def download_invasion_block_data() -> None:
    """Download Invasion block images and write consolidated card metadata.

    Fetches cards from Scryfall for INV/PLS/APC with pagination, downloads
    normal-sized images to ``data/reference_images/{set_code}``, and writes
    ``data/card_metadata.json`` keyed by relative image path.
    """
    project_root = Path(__file__).resolve().parents[2]
    images_root = project_root / "data" / "reference_images"
    metadata_path = project_root / "data" / "card_metadata.json"

    images_root.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    client = RateLimitedSession()

    metadata_by_image_path: dict[str, dict[str, Any]] = {}
    total_downloaded = 0

    for set_code in SET_CODES:
        logger.info("Starting download for set '%s'", set_code)
        set_output_dir = images_root / set_code
        set_output_dir.mkdir(parents=True, exist_ok=True)

        next_page: str | None = SCRYFALL_SEARCH_URL
        params: dict[str, Any] | None = {
            "q": f"set:{set_code}",
            "unique": "prints",
        }
        set_seen = 0
        page_num = 0

        while next_page:
            page_num += 1
            logger.info("Fetching set '%s' page %d", set_code, page_num)
            response = client.request("GET", next_page, params=params)
            payload = response.json()

            cards = payload.get("data", [])
            if not isinstance(cards, list):
                logger.warning("Unexpected payload format for set '%s' page %d", set_code, page_num)
                cards = []

            for card in cards:
                if not isinstance(card, dict):
                    continue

                image_url = get_image_url(card)
                if not image_url:
                    logger.warning(
                        "Skipping card without image URL: set=%s collector_number=%s name=%s",
                        set_code,
                        card.get("collector_number", "unknown"),
                        card.get("name", "unknown"),
                    )
                    continue

                collector_number = sanitize_filename(str(card.get("collector_number", "unknown")))
                card_name = sanitize_filename(str(card.get("name", "unknown")))
                output_file = set_output_dir / f"{collector_number}_{card_name}.jpg"

                try:
                    image_response = client.request("GET", image_url)
                    output_file.write_bytes(image_response.content)
                except requests.RequestException as exc:
                    logger.error("Failed to download image for %s (%s): %s", card_name, set_code, exc)
                    continue

                relative_image_path = output_file.relative_to(project_root).as_posix()
                metadata_by_image_path[relative_image_path] = build_card_metadata(card)

                set_seen += 1
                total_downloaded += 1
                if set_seen % 25 == 0:
                    logger.info("Downloaded %d cards for set '%s'", set_seen, set_code)

            next_page = payload.get("next_page") if payload.get("has_more") else None
            params = None

        logger.info("Finished set '%s'. Downloaded %d cards.", set_code, set_seen)

    metadata_path.write_text(json.dumps(metadata_by_image_path, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote metadata for %d images to %s", total_downloaded, metadata_path.as_posix())


def main() -> None:
    """Configure logging and execute the downloader workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    download_invasion_block_data()


if __name__ == "__main__":
    main()
