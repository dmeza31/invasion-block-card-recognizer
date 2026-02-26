from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from recognizer.embedder import CardEmbedder


def build_index_pipeline(
    image_dir: str | Path = "data/reference_images",
    index_path: str | Path = "data/embeddings/invasion_block.index",
    metadata_path: str | Path = "data/embeddings/card_metadata.json",
    device: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build and persist a FAISS IndexFlatIP from reference card images."""

    logger = logging.getLogger(__name__)
    started_at = time.perf_counter()

    image_dir_path = Path(image_dir)
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)

    logger.info("Initializing CardEmbedder (device=%s)", device or "auto")
    embedder = CardEmbedder(device=device)

    logger.info("Computing embeddings from %s", image_dir_path.as_posix())
    embed_start = time.perf_counter()
    embeddings, metadata = embedder.embed_directory(str(image_dir_path))
    embed_elapsed = time.perf_counter() - embed_start

    if embeddings.size == 0:
        raise ValueError(f"No embeddings generated from directory: {image_dir_path}")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    logger.info(
        "Embedding complete: vectors=%d dim=%d elapsed=%.2fs",
        embeddings.shape[0],
        embeddings.shape[1],
        embed_elapsed,
    )

    import faiss

    logger.info("Building FAISS IndexFlatIP")
    index_start = time.perf_counter()
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    index_elapsed = time.perf_counter() - index_start
    logger.info("FAISS index built: ntotal=%d elapsed=%.2fs", faiss_index.ntotal, index_elapsed)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving index to %s", index_file.as_posix())
    faiss.write_index(faiss_index, index_file.as_posix())

    logger.info("Saving metadata to %s", metadata_file.as_posix())
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    total_elapsed = time.perf_counter() - started_at
    logger.info("Index build pipeline complete in %.2fs", total_elapsed)

    return faiss_index, metadata


def load_index(
    index_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load FAISS index and metadata list for querying."""

    index_file = Path(index_path)
    metadata_file = Path(metadata_path)

    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    import faiss

    index = faiss.read_index(index_file.as_posix())

    raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Metadata file must contain a list of metadata dictionaries")

    return index, raw


def main() -> None:
    """Run the full build pipeline with timing logs."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_index_pipeline(
        image_dir="data/reference_images",
        index_path="data/embeddings/invasion_block.index",
        metadata_path="data/embeddings/card_metadata.json",
        device=None,
    )


if __name__ == "__main__":
    main()
