from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from recognizer.embedder import CardEmbedder
from recognizer.index_builder import load_index


class CardRecognizer:
    """Recognize MTG cards by querying CLIP embeddings against a FAISS index."""

    def __init__(self, index_path: str, metadata_path: str) -> None:
        """Load index artifacts and initialize the embedder.

        Args:
            index_path: Path to the serialized FAISS index.
            metadata_path: Path to metadata JSON aligned to FAISS vector ids.
        """

        self.index, self.metadata = load_index(index_path=index_path, metadata_path=metadata_path)
        self.embedder = CardEmbedder()

    def recognize(self, image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
        """Recognize a card image and return top-k nearest matches.

        Args:
            image: Input card image as PIL image.
            top_k: Number of nearest neighbors to return.

        Returns:
            A list of dictionaries sorted by descending similarity score, with keys:
            ``name``, ``set_code``, ``collector_number``, ``similarity_score``, ``image_path``.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vector = self.embedder.embed_image(image)
        query = np.expand_dims(vector, axis=0).astype(np.float32, copy=False)

        scores, indices = self.index.search(query, top_k)

        results: list[dict[str, Any]] = []
        for similarity_score, row_index in zip(scores[0], indices[0]):
            if row_index < 0 or row_index >= len(self.metadata):
                continue

            metadata = self.metadata[row_index]
            results.append(
                {
                    "name": metadata.get("card_name") or metadata.get("name", ""),
                    "set_code": metadata.get("set_code", ""),
                    "collector_number": metadata.get("collector_number", ""),
                    "similarity_score": float(similarity_score),
                    "image_path": metadata.get("image_path", ""),
                }
            )

        results.sort(key=lambda item: item["similarity_score"], reverse=True)
        return results

    def recognize_from_bytes(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, Any]]:
        """Recognize a card from raw image bytes.

        Args:
            image_bytes: Raw bytes of an encoded image.
            top_k: Number of nearest neighbors to return.

        Returns:
            The same top-k result list produced by :meth:`recognize`.
        """

        with Image.open(BytesIO(image_bytes)) as img:
            pil_image = img.convert("RGB")

        return self.recognize(image=pil_image, top_k=top_k)
