from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path

import numpy as np
from packaging import version as packaging_version
import torch
from PIL import Image

if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = types.SimpleNamespace(
        packaging=types.SimpleNamespace(version=packaging_version)
    )

import clip


class CardEmbedder:
    """Generate card image embeddings using OpenAI CLIP ViT-B/32."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.logger = logging.getLogger(__name__)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.logger.info("Loaded CLIP model ViT-B/32 on device=%s", self.device)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed one PIL image and return an L2-normalized float32 vector."""

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_directory(self, image_dir: str) -> tuple[np.ndarray, list[dict]]:
        """Embed all .jpg images under a directory tree and return embeddings+metadata."""

        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory not found: {root}")

        image_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".jpg"
        )

        if not image_paths:
            self.logger.warning("No .jpg files found under %s", root.as_posix())
            return np.empty((0, 0), dtype=np.float32), []

        embeddings: list[np.ndarray] = []
        metadata: list[dict] = []

        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            vector = self.embed_image(image)
            embeddings.append(vector)

            set_code, collector_number, card_name = self._parse_metadata(root, image_path)
            metadata.append(
                {
                    "set_code": set_code,
                    "collector_number": collector_number,
                    "card_name": card_name,
                    "image_path": image_path.as_posix(),
                }
            )

            if index % 50 == 0 or index == len(image_paths):
                self.logger.info("Embedded %d/%d images", index, len(image_paths))

        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        return embedding_matrix, metadata

    @staticmethod
    def _parse_metadata(root: Path, image_path: Path) -> tuple[str, str, str]:
        relative = image_path.relative_to(root)
        set_code = relative.parts[0] if len(relative.parts) > 1 else "unknown"

        stem = image_path.stem
        collector_number = "unknown"
        card_name = stem

        match = re.match(r"^([^_]+)_(.+)$", stem)
        if match:
            collector_number = match.group(1)
            card_name = match.group(2)

        card_name = card_name.replace("_", " ").strip()
        return set_code, collector_number, card_name
