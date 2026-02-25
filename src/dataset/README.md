# Dataset Module Documentation

This document explains the dataset-related modules in this folder and how they work together:

1. `downloader.py`
2. `compositor.py`
3. `augmentor.py.py`

---

## 1) `downloader.py`

### Objective
Download clean reference card images for the Invasion block from Scryfall (`inv`, `pls`, `apc`) and build a metadata index used by downstream preprocessing/training pipelines.

### Methods

- `sanitize_filename(value: str) -> str`
  - **What it does:** Normalizes text into a safe filename fragment.
  - **Why it is important:** Prevents invalid filenames and cross-platform path issues when saving card images.

- `get_image_url(card: dict[str, Any]) -> str | None`
  - **What it does:** Resolves the image URL from either top-level `image_uris.normal` or `card_faces[*].image_uris.normal`.
  - **Why it is important:** Ensures both normal and multi-faced cards can be downloaded correctly.

- `build_card_metadata(card: dict[str, Any]) -> dict[str, Any]`
  - **What it does:** Extracts normalized metadata fields (`name`, `set`, `collector_number`, `mana_cost`, `type_line`, `oracle_text`), with face-based fallback logic.
  - **Why it is important:** Produces a consistent metadata schema for training and inference lookups.

- `RateLimitedSession.__init__(delay_seconds: float = REQUEST_DELAY_SECONDS)`
  - **What it does:** Initializes a reusable `requests.Session` with rate-limit tracking.
  - **Why it is important:** Centralizes client behavior and avoids repeated setup overhead.

- `RateLimitedSession.request(method: str, url: str, **kwargs) -> requests.Response`
  - **What it does:** Executes HTTP requests while enforcing minimum delay and raising on non-2xx responses.
  - **Why it is important:** Complies with API rate limits and standardizes request error handling.

- `download_invasion_block_data() -> None`
  - **What it does:** Main downloader pipeline; paginates Scryfall results, downloads images, writes `data/reference_images/...` and `data/card_metadata.json`.
  - **Why it is important:** Creates the canonical dataset source that all later dataset transformations depend on.

- `main() -> None`
  - **What it does:** Configures logging and runs the downloader pipeline.
  - **Why it is important:** Provides a script entrypoint for `python -m dataset.downloader`.

### Execution Flow
1. Initialize logger and rate-limited HTTP session.
2. For each set code (`inv`, `pls`, `apc`), call Scryfall search endpoint.
3. Follow pagination via `next_page` while `has_more` is true.
4. For each card, resolve image URL (including double-faced fallback), sanitize filename, and download image.
5. Build metadata per image and accumulate in-memory map.
6. Write final metadata map to `data/card_metadata.json`.

---

## 2) `compositor.py`

### Objective
Generate realistic synthetic photo-style card examples by placing clean card images on varied backgrounds with geometric and visual perturbations. This improves robustness for real-world capture conditions.

### Methods

- `_list_background_images(backgrounds_dir: Path) -> list[Path]`
  - **What it does:** Collects background files recursively.
  - **Why it is important:** Provides the background pool for compositing.

- `_list_card_images(reference_dir: Path) -> list[Path]`
  - **What it does:** Collects reference card images recursively.
  - **Why it is important:** Enables full-folder batch compositing.

- `_find_perspective_coeffs(src_points, dst_points) -> list[float]`
  - **What it does:** Solves perspective transform coefficients.
  - **Why it is important:** Powers realistic card warp simulations.

- `_apply_perspective_warp(card_image, max_warp_ratio=0.06) -> Image.Image`
  - **What it does:** Applies mild random perspective distortion.
  - **Why it is important:** Simulates camera angle/perspective differences.

- `_create_drop_shadow(card_rgba) -> tuple[Image.Image, tuple[int, int]]`
  - **What it does:** Generates blurred shadow mask and random offset.
  - **Why it is important:** Simulates depth/lighting effects seen in real photos.

- `_fit_card_to_background(card, background_size) -> Image.Image`
  - **What it does:** Resizes card relative to background dimensions.
  - **Why it is important:** Simulates varying camera distance and framing.

- `_save_composite(image, output_path, index, count) -> Path`
  - **What it does:** Saves composited output with naming conventions.
  - **Why it is important:** Keeps outputs deterministic and organized.

- `composite_on_backgrounds(card_image_path, backgrounds_dir, output_path, count=10) -> list[Path]`
  - **What it does:** Single-card compositing routine; random background selection, transforms, optional shadow, save loop.
  - **Why it is important:** Core synthesis unit used by both single-card and dataset-wide workflows.

- `_generate_fallback_background(size, index) -> Image.Image`
  - **What it does:** Generates synthetic texture/gradient fallback backgrounds.
  - **Why it is important:** Keeps pipeline operational when remote background providers fail.

- `_fetch_pexels_background_urls(api_key, limit=80) -> list[str]`
  - **What it does:** Queries Pexels API and extracts candidate image URLs.
  - **Why it is important:** Supplies realistic external backgrounds.

- `_candidate_background_urls(index, pexels_urls) -> list[str]`
  - **What it does:** Chooses provider URL(s) for an index.
  - **Why it is important:** Encapsulates source-selection policy cleanly.

- `_download_background_image(url, logger, index, count, max_attempts=3) -> Image.Image | None`
  - **What it does:** Downloads image with retries/backoff and validation.
  - **Why it is important:** Makes background acquisition resilient and observable.

- `download_sample_backgrounds(output_dir, count=20, reuse_existing=True) -> list[Path]`
  - **What it does:** Reuses cached backgrounds, downloads missing ones, and falls back to generated textures.
  - **Why it is important:** Avoids redundant external API calls and speeds repeated runs.

- `composite_dataset(reference_dir, backgrounds_dir, output_dir, count_per_image=10, background_count=20, refresh_backgrounds=False, log_every=25) -> list[Path]`
  - **What it does:** Dataset-wide orchestration; prepares backgrounds once, composites each card folder-wide, and logs progress.
  - **Why it is important:** Scalable batch mode equivalent to augmentor-style processing.

- `parse_args() -> argparse.Namespace`
  - **What it does:** CLI argument parser for dataset compositing.
  - **Why it is important:** Enables reproducible command-line runs.

- `main() -> None`
  - **What it does:** Configures logging, parses args, invokes batch compositing.
  - **Why it is important:** Entry point for `python -m dataset.compositor`.

### Execution Flow
1. Parse CLI args and initialize logging.
2. Discover all reference card images under `reference_dir`.
3. Prepare backgrounds once via `download_sample_backgrounds`:
   - reuse local cache if available,
   - otherwise pull from Pexels,
   - fallback to generated backgrounds when needed.
4. Iterate all reference images and call `composite_on_backgrounds` for each.
5. For each synthetic image: background selection → resize → rotate → perspective warp → optional shadow → random placement → save.
6. Log progress and summary counts in console.

---

## 3) `augmentator.py` *(implemented as `augmentor.py`)*

### Objective
Create synthetic training images from clean references using torchvision transforms and build train/val/test directory splits with per-class stratification.

### Methods

- `build_augmentation_pipeline(include_gaussian_noise: bool = True) -> transforms.Compose`
  - **What it does:** Defines augmentation chain (rotation, perspective, color jitter, blur, affine, optional noise).
  - **Why it is important:** Encapsulates augmentation policy in one reusable pipeline.

- `_add_gaussian_noise(image: Image.Image, std: float = 0.04) -> Image.Image`
  - **What it does:** Adds controlled Gaussian noise to image tensors.
  - **Why it is important:** Improves robustness to sensor/compression noise.

- `_iter_reference_images(input_dir: Path) -> list[Path]`
  - **What it does:** Recursively enumerates source images.
  - **Why it is important:** Supports full dataset processing.

- `_parse_card_parts(image_path: Path, input_dir: Path) -> tuple[str, str, str]`
  - **What it does:** Derives set code, collector number, and card name from filename/path.
  - **Why it is important:** Ensures deterministic class folder naming.

- `generate_augmented_dataset(input_dir, output_dir, num_variants=30) -> None`
  - **What it does:** Generates multiple augmented variants per reference card and writes them to class-specific folders.
  - **Why it is important:** Expands limited clean data into a richer training set.

- `_safe_split_counts(total, train_ratio, val_ratio) -> tuple[int, int, int]`
  - **What it does:** Computes robust split sizes even for small class counts.
  - **Why it is important:** Prevents invalid or empty splits for edge-case classes.

- `create_splits(output_dir, train_ratio=0.8, val_ratio=0.1) -> None`
  - **What it does:** Builds `train/`, `val/`, `test/` folders with per-class shuffled stratified copies.
  - **Why it is important:** Produces model-ready dataset layout for training and evaluation.

- `parse_args() -> argparse.Namespace`
  - **What it does:** Parses CLI options for augmentation and splitting.
  - **Why it is important:** Enables configurable reproducible runs from terminal/Makefile.

- `main() -> None`
  - **What it does:** Sets logging, seeds RNG, executes augmentation then split creation.
  - **Why it is important:** Entry point for `python -m dataset.augmentor`.

### Execution Flow
1. Parse CLI parameters and initialize logging.
2. Build augmentation pipeline.
3. Enumerate all reference images.
4. For each image, generate `num_variants` transformed outputs and save under class-specific directory.
5. After generation, compute train/val/test counts per class.
6. Copy files into split directories and log class-level split counts.

---

## Typical End-to-End Dataset Preparation Order

1. Run downloader to create clean references and metadata.
2. Run compositor to generate real-world scene composites (optional but recommended for robustness).
3. Run augmentor/augmentator to increase synthetic diversity and build train/val/test splits.
