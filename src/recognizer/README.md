# Recognizer Module Documentation

This document explains the recognition-related modules in this folder and how they work together:

1. `embedder.py`
2. `index_builder.py`
3. `recognizer.py`
4. `detector.py`
5. `build_index.py`
6. `__init__.py`

---

## 1) `embedder.py`

### Objective
Convert card images into dense semantic vectors using CLIP (`ViT-B/32`) so similarity search can be done efficiently with FAISS.

### Methods

- `CardEmbedder.__init__(device: str | None = None)`
  - **What it does:** Selects compute device (`cuda` if available, otherwise `cpu`), loads CLIP model + preprocess pipeline, switches model to eval mode.
  - **Why it is important:** Centralizes model loading and preprocessing once, so embedding operations are consistent and efficient.

- `CardEmbedder.embed_image(image: Image.Image) -> np.ndarray`
  - **What it does:** Preprocesses a PIL image, runs CLIP image encoder, and returns an L2-normalized `float32` vector.
  - **Why it is important:** Produces normalized vectors suitable for cosine-style similarity via inner-product FAISS search.

- `CardEmbedder.embed_directory(image_dir: str) -> tuple[np.ndarray, list[dict]]`
  - **What it does:** Recursively loads all `.jpg` images, embeds each image, and returns:
    - embedding matrix (`N x D`)
    - metadata list (set code, collector number, card name, image path)
  - **Why it is important:** This is the batch embedding step used to build the retrieval index from the reference card corpus.

- `CardEmbedder._parse_metadata(root: Path, image_path: Path) -> tuple[str, str, str]`
  - **What it does:** Parses image path/filename into `(set_code, collector_number, card_name)`.
  - **Why it is important:** Keeps index vectors aligned with human-readable card identity fields used during recognition output.

### Additional compatibility detail
- The module contains a small `pkg_resources` compatibility shim for `openai-clip` import behavior.
- **Why it matters:** Prevents runtime import failures from legacy package expectations and keeps CLIP loading stable.

---

## 2) `index_builder.py`

### Objective
Build and persist a FAISS index from reference image embeddings, and load index artifacts for inference.

### Methods

- `build_index_pipeline(image_dir, index_path, metadata_path, device=None) -> tuple[Any, list[dict[str, Any]]]`
  - **What it does:** End-to-end indexing pipeline:
    1. Initializes `CardEmbedder`
    2. Embeds reference images
    3. Builds FAISS `IndexFlatIP`
    4. Saves index file and aligned metadata JSON
  - **Why it is important:** Creates the searchable retrieval backend used by runtime recognition.

- `load_index(index_path, metadata_path) -> tuple[Any, list[dict[str, Any]]]`
  - **What it does:** Loads FAISS index from disk and parses metadata JSON list.
  - **Why it is important:** Provides the runtime entrypoint for retrieval without recomputing embeddings.

- `main() -> None`
  - **What it does:** Configures logging and runs `build_index_pipeline` with default paths.
  - **Why it is important:** Script entrypoint for local index generation (used by Make targets).

### Important implementation note
- `faiss` is imported inside functions rather than at module top-level.
- **Why it matters:** Avoids potential native import-order instability in mixed ML stacks.

---

## 3) `recognizer.py`

### Objective
Perform card recognition by embedding a query image and searching nearest neighbors in the FAISS index.

### Methods

- `CardRecognizer.__init__(index_path: str, metadata_path: str)`
  - **What it does:** Loads FAISS index + metadata and initializes `CardEmbedder`.
  - **Why it is important:** Prepares everything needed for low-latency repeated queries.

- `CardRecognizer.recognize(image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]`
  - **What it does:** Embeds input image, queries FAISS, maps result ids to metadata, and returns sorted predictions.
  - **Why it is important:** Core inference method used by APIs, tests, and interactive workflows.

- `CardRecognizer.recognize_from_bytes(image_bytes: bytes, top_k: int = 5) -> list[dict[str, Any]]`
  - **What it does:** Converts encoded image bytes to RGB PIL image and delegates to `recognize`.
  - **Why it is important:** Convenient for web/API pipelines where images arrive as uploaded bytes.

---

## 4) `detector.py`

### Objective
Detect card-like quadrilaterals in photos and normalize each detected card crop into a standard rectangle (`488x680`) before recognition.

### Methods

- `CardDetector.preprocess(image: np.ndarray) -> np.ndarray`
  - **What it does:** Applies CLAHE histogram equalization (grayscale or LAB luminance channel) for lighting normalization.
  - **Why it is important:** Improves robustness under uneven illumination and shadows.

- `CardDetector._order_points(points: np.ndarray) -> np.ndarray`
  - **What it does:** Orders 4 contour points as top-left, top-right, bottom-right, bottom-left.
  - **Why it is important:** Ensures perspective transform is geometrically correct.

- `CardDetector.detect_and_crop(image: np.ndarray) -> list[np.ndarray]`
  - **What it does:** Detection pipeline:
    1. Grayscale conversion
    2. Gaussian blur
    3. Canny edges
    4. Contour detection and quadrilateral filtering (`area > 10%`)
    5. Perspective warp to `488x680`
    6. Return all detected crops
    7. Fallback to full input image if no card contour is detected
  - **Why it is important:** Handles real-world photos where card framing is imperfect or perspective-distorted.

---

## Execution Flow

### A) Index Build Flow (offline)
1. Run index build pipeline (`python -m recognizer.index_builder` / Make target).
2. `CardEmbedder` loads CLIP and embeds all reference images.
3. `index_builder` creates FAISS `IndexFlatIP` and writes:
   - `data/embeddings/invasion_block.index`
   - `data/embeddings/card_metadata.json`
4. Artifacts are now ready for inference.

### B) Recognition Flow (direct image)
1. Initialize `CardRecognizer` with index + metadata paths.
2. Call `recognize(...)` (PIL image) or `recognize_from_bytes(...)` (encoded bytes).
3. Query image is embedded via CLIP.
4. FAISS returns top-k nearest vectors.
5. Vector ids map to metadata and final prediction payload is returned.

### C) Recognition Flow (photo with detector, optional)
1. Read photo with OpenCV.
2. Optionally call `CardDetector.preprocess(...)` for lighting normalization.
3. Call `CardDetector.detect_and_crop(...)` to extract one or more rectified card crops.
4. Run `CardRecognizer.recognize(...)` on each crop.
5. Merge/select predictions according to your application policy (e.g., best score or any-crop hit).

---

## Typical End-to-End Usage Order

1. Build index artifacts from reference images.
2. Initialize recognizer in API/UI/test runtime.
3. For tightly cropped images, recognize directly.
4. For real photos, detect/rectify cards first, then recognize.