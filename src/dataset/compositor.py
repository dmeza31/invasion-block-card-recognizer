from __future__ import annotations

import argparse
import io
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"


def _list_background_images(backgrounds_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in backgrounds_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_card_images(reference_dir: Path) -> list[Path]:
    """Return all card image files under a reference image directory."""

    return sorted(
        path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_perspective_coeffs(src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]]) -> list[float]:
    matrix: list[list[float]] = []
    target: list[float] = []

    for (src_x, src_y), (dst_x, dst_y) in zip(src_points, dst_points):
        matrix.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y])
        matrix.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y])
        target.extend([dst_x, dst_y])

    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64))
    return coeffs.tolist()


def _apply_perspective_warp(card_image: Image.Image, max_warp_ratio: float = 0.06) -> Image.Image:
    width, height = card_image.size
    max_dx = width * max_warp_ratio
    max_dy = height * max_warp_ratio

    src = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    dst = [
        (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)),
        (width + random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)),
        (width + random.uniform(-max_dx, max_dx), height + random.uniform(-max_dy, max_dy)),
        (random.uniform(-max_dx, max_dx), height + random.uniform(-max_dy, max_dy)),
    ]

    coeffs = _find_perspective_coeffs(dst, src)
    return card_image.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
    )


def _create_drop_shadow(card_rgba: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    width, height = card_rgba.size
    alpha_mask = card_rgba.split()[-1]

    shadow_opacity = random.randint(50, 140)
    shadow_color = (0, 0, 0, shadow_opacity)
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(shadow)
    draw.bitmap((0, 0), alpha_mask, fill=shadow_color)

    blur_radius = random.uniform(3.0, 10.0)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    offset = (random.randint(-20, 20), random.randint(-20, 20))
    return shadow, offset


def _fit_card_to_background(card: Image.Image, background_size: tuple[int, int]) -> Image.Image:
    bg_width, bg_height = background_size
    target_ratio = random.uniform(0.4, 0.8)

    target_height = int(bg_height * target_ratio)
    aspect_ratio = card.width / card.height
    target_width = max(1, int(target_height * aspect_ratio))

    if target_width > bg_width * 0.9:
        target_width = int(bg_width * 0.9)
        target_height = max(1, int(target_width / aspect_ratio))

    return card.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _save_composite(image: Image.Image, output_path: Path, index: int, count: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix:
        if count == 1:
            destination = output_path
        else:
            destination = output_path.with_name(f"{output_path.stem}_{index + 1}{output_path.suffix}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        destination = output_path / f"composite_{index + 1}.jpg"

    image.convert("RGB").save(destination, format="JPEG", quality=95)
    return destination


def composite_on_backgrounds(
    card_image_path: str | Path,
    backgrounds_dir: str | Path,
    output_path: str | Path,
    count: int = 10,
) -> list[Path]:
    """Composite a clean card image over random backgrounds for robust training.

    Steps applied for each composite:
    1) resize card to ~40-80% of background height,
    2) rotate card by ±20°,
    3) apply slight perspective warp,
    4) place card at random valid position,
    5) optionally add a random drop shadow,
    6) save composited result.
    """

    logger = logging.getLogger(__name__)

    card_path = Path(card_image_path)
    bg_dir = Path(backgrounds_dir)
    destination = Path(output_path)

    if not card_path.exists():
        raise FileNotFoundError(f"Card image not found: {card_path}")
    if not bg_dir.exists():
        raise FileNotFoundError(f"Background directory not found: {bg_dir}")

    background_images = _list_background_images(bg_dir)
    if not background_images:
        raise ValueError(f"No background images found in {bg_dir}")

    with Image.open(card_path) as card_file:
        clean_card = card_file.convert("RGBA")

    started_at = time.perf_counter()
    saved_paths: list[Path] = []

    for idx in range(count):
        bg_path = random.choice(background_images)
        with Image.open(bg_path) as bg_file:
            background = bg_file.convert("RGBA")

        transformed_card = _fit_card_to_background(clean_card, background.size)
        transformed_card = transformed_card.rotate(
            random.uniform(-20, 20),
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        transformed_card = _apply_perspective_warp(transformed_card)

        card_width, card_height = transformed_card.size
        bg_width, bg_height = background.size

        max_x = max(0, bg_width - card_width)
        max_y = max(0, bg_height - card_height)
        pos_x = random.randint(0, max_x) if max_x else 0
        pos_y = random.randint(0, max_y) if max_y else 0

        composite = background.copy()

        if random.random() < 0.7:
            shadow, (offset_x, offset_y) = _create_drop_shadow(transformed_card)
            shadow_x = max(0, min(pos_x + offset_x, max_x))
            shadow_y = max(0, min(pos_y + offset_y, max_y))
            composite.alpha_composite(shadow, dest=(shadow_x, shadow_y))

        composite.alpha_composite(transformed_card, dest=(pos_x, pos_y))

        saved = _save_composite(composite, destination, idx, count)
        saved_paths.append(saved)
        logger.debug("Saved composite %d/%d -> %s", idx + 1, count, saved.as_posix())

    elapsed = time.perf_counter() - started_at
    logger.info(
        "Completed card compositing: source=%s generated=%d elapsed=%.2fs",
        card_path.as_posix(),
        len(saved_paths),
        elapsed,
    )

    return saved_paths


def _generate_fallback_background(size: tuple[int, int], index: int) -> Image.Image:
    width, height = size
    rng = random.Random(index * 9973)

    start = np.array([rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)], dtype=np.float32)
    end = np.array([rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)], dtype=np.float32)

    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / max(1, height - 1)
        row_color = (1.0 - ratio) * start + ratio * end
        gradient[y, :, :] = row_color.astype(np.uint8)

    image = Image.fromarray(gradient, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(50):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = min(width - 1, x1 + rng.randint(10, 180))
        y2 = min(height - 1, y1 + rng.randint(10, 180))
        color = tuple(rng.randint(20, 235) for _ in range(3))
        alpha = rng.randint(30, 110)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        o_draw = ImageDraw.Draw(overlay)
        o_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


def _fetch_pexels_background_urls(api_key: str, limit: int = 80) -> list[str]:
    """Fetch candidate background image URLs from the Pexels API."""

    response = requests.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        params={
            "query": "table desk wood playmat texture",
            "per_page": max(1, min(limit, 80)),
            "orientation": "landscape",
        },
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()
    photos = payload.get("photos", [])

    urls: list[str] = []
    for photo in photos:
        if not isinstance(photo, dict):
            continue
        src = photo.get("src", {})
        if not isinstance(src, dict):
            continue
        candidate = src.get("large2x") or src.get("large") or src.get("original")
        if isinstance(candidate, str):
            urls.append(candidate)

    return urls


def _candidate_background_urls(index: int, pexels_urls: list[str]) -> list[str]:
    """Return prioritized background image URLs for one sample index.

    Uses Pexels results first. Returns an empty list when no API images are
    available so the caller can fall back to synthetic background generation.
    """

    if not pexels_urls:
        return []

    selected_url = pexels_urls[(index - 1) % len(pexels_urls)]
    return [selected_url]


def _download_background_image(
    url: str,
    logger: logging.Logger,
    index: int,
    count: int,
    max_attempts: int = 3,
) -> Image.Image | None:
    """Attempt to download a single background image with retry/backoff."""

    headers = {
        "User-Agent": "mtg-invasion-recognizer/1.0 (+https://github.com/)",
    }

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=20, headers=headers)
        except requests.RequestException as exc:
            logger.warning(
                "Background request failed (%d/%d, attempt %d/%d): %s",
                index,
                count,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)
            continue

        if not response.ok:
            logger.warning(
                "Background request returned HTTP %d (%d/%d, attempt %d/%d)",
                response.status_code,
                index,
                count,
                attempt,
                max_attempts,
            )
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if "image" not in content_type or not response.content:
            logger.warning(
                "Background response was not an image (%d/%d, attempt %d/%d, content-type=%s)",
                index,
                count,
                attempt,
                max_attempts,
                content_type or "unknown",
            )
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)
            continue

        try:
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning(
                "Background decode failed (%d/%d, attempt %d/%d): %s",
                index,
                count,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)

    return None


def download_sample_backgrounds(
    output_dir: str | Path,
    count: int = 20,
    reuse_existing: bool = True,
) -> list[Path]:
    """Prepare background images once, reusing local files when available."""

    logger = logging.getLogger(__name__)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if count <= 0:
        return []

    saved_paths: list[Path] = []
    missing_indices: list[int] = []

    for index in range(1, count + 1):
        destination = out_dir / f"background_{index:03d}.jpg"
        if reuse_existing and destination.exists():
            saved_paths.append(destination)
        else:
            missing_indices.append(index)

    if reuse_existing and not missing_indices:
        logger.info("Reusing %d existing backgrounds from %s", len(saved_paths), out_dir.as_posix())
        return saved_paths

    pexels_api_key = os.getenv("PEXELS_API_KEY", "").strip()
    pexels_urls: list[str] = []

    if pexels_api_key:
        try:
            pexels_urls = _fetch_pexels_background_urls(pexels_api_key, limit=max(40, count * 2))
            logger.info("Fetched %d candidate backgrounds from Pexels", len(pexels_urls))
        except requests.RequestException as exc:
            logger.warning("Failed to fetch Pexels background URLs: %s", exc)
    else:
        logger.warning("PEXELS_API_KEY is not set; using generated fallback backgrounds")

    for index in missing_indices:
        destination = out_dir / f"background_{index:03d}.jpg"
        downloaded_image: Image.Image | None = None

        for provider_url in _candidate_background_urls(index, pexels_urls):
            downloaded_image = _download_background_image(provider_url, logger, index, count)
            if downloaded_image is not None:
                break

        if downloaded_image is not None:
            downloaded_image.save(destination, format="JPEG", quality=92)
            logger.info("Downloaded background %d/%d", index, count)
        else:
            fallback_image = _generate_fallback_background((1600, 900), index)
            fallback_image.save(destination, format="JPEG", quality=92)
            logger.info("Generated fallback background %d/%d", index, count)

        saved_paths.append(destination)

    saved_paths = sorted(saved_paths)
    logger.info("Prepared %d background images in %s", len(saved_paths), out_dir.as_posix())
    return saved_paths[:count]


def composite_dataset(
    reference_dir: str | Path,
    backgrounds_dir: str | Path,
    output_dir: str | Path,
    count_per_image: int = 10,
    background_count: int = 20,
    refresh_backgrounds: bool = False,
    log_every: int = 25,
) -> list[Path]:
    """Composite every reference image using one shared background pool."""

    logger = logging.getLogger(__name__)
    reference_root = Path(reference_dir)
    output_root = Path(output_dir)

    if not reference_root.exists():
        raise FileNotFoundError(f"Reference directory not found: {reference_root}")

    card_images = _list_card_images(reference_root)
    if not card_images:
        raise FileNotFoundError(f"No reference card images found under {reference_root}")

    if log_every <= 0:
        log_every = 1

    run_started_at = time.perf_counter()

    background_started_at = time.perf_counter()
    download_sample_backgrounds(
        output_dir=backgrounds_dir,
        count=background_count,
        reuse_existing=not refresh_backgrounds,
    )
    background_elapsed = time.perf_counter() - background_started_at

    all_outputs: list[Path] = []
    total_cards = len(card_images)
    total_expected = total_cards * count_per_image
    logger.info(
        "Starting dataset compositing: cards=%d expected_outputs=%d backgrounds=%d background_prep=%.2fs",
        total_cards,
        total_expected,
        background_count,
        background_elapsed,
    )

    cards_processed = 0
    failed_cards: list[str] = []

    for idx, card_path in enumerate(card_images, start=1):
        per_card_output = output_root / card_path.relative_to(reference_root).parent / card_path.stem
        logger.info("[%d/%d] Compositing %s", idx, total_cards, card_path.as_posix())
        try:
            generated = composite_on_backgrounds(
                card_image_path=card_path,
                backgrounds_dir=backgrounds_dir,
                output_path=per_card_output,
                count=count_per_image,
            )
            all_outputs.extend(generated)
            cards_processed += 1
        except Exception as exc:
            failed_cards.append(card_path.as_posix())
            logger.exception("Failed compositing card %s: %s", card_path.as_posix(), exc)
            continue

        if idx % log_every == 0 or idx == total_cards:
            elapsed = time.perf_counter() - run_started_at
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining_cards = total_cards - idx
            eta_seconds = remaining_cards / rate if rate > 0 else 0.0
            logger.info(
                "Progress: %d/%d cards (%.1f%%) outputs=%d elapsed=%.1fs eta=%.1fs",
                idx,
                total_cards,
                (idx / total_cards) * 100,
                len(all_outputs),
                elapsed,
                eta_seconds,
            )

    total_elapsed = time.perf_counter() - run_started_at
    logger.info(
        "Dataset compositing complete. cards_ok=%d cards_failed=%d outputs=%d elapsed=%.2fs output_dir=%s",
        cards_processed,
        len(failed_cards),
        len(all_outputs),
        total_elapsed,
        output_root.as_posix(),
    )

    return all_outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full-dataset compositing."""

    parser = argparse.ArgumentParser(description="Composite MTG card images over realistic backgrounds")
    parser.add_argument("--reference-dir", default="data/reference_images", help="Input directory with card images")
    parser.add_argument("--backgrounds-dir", default="data/backgrounds", help="Directory for cached backgrounds")
    parser.add_argument("--output-dir", default="data/composited", help="Output directory for composited images")
    parser.add_argument("--count-per-image", type=int, default=10, help="Composited images to generate per card")
    parser.add_argument("--background-count", type=int, default=20, help="Number of backgrounds to prepare")
    parser.add_argument("--log-every", type=int, default=25, help="Log progress every N cards")
    parser.add_argument(
        "--refresh-backgrounds",
        action="store_true",
        help="Force refreshing missing/reusable backgrounds from providers",
    )
    return parser.parse_args()


def main() -> None:
    """Run folder-wide compositing with one-time background preparation."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    composite_dataset(
        reference_dir=args.reference_dir,
        backgrounds_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        count_per_image=args.count_per_image,
        background_count=args.background_count,
        refresh_backgrounds=args.refresh_backgrounds,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
