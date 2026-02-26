from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from recognizer.recognizer import CardRecognizer


class TestCardRecognizerReferenceNoDetector:
    """Benchmark CardRecognizer on reference images without detector preprocessing."""

    INDEX_PATH = Path("data/embeddings/invasion_block.index")
    METADATA_PATH = Path("data/embeddings/card_metadata.json")
    REFERENCE_DIR = Path("data/reference_images")

    @staticmethod
    def _parse_truth_from_reference_path(image_path: Path, reference_root: Path) -> tuple[str, str] | None:
        """Parse expected (set_code, collector_number) from reference image path."""

        try:
            relative = image_path.relative_to(reference_root)
        except ValueError:
            return None

        if len(relative.parts) < 2:
            return None

        set_code = relative.parts[0]
        stem = image_path.stem
        if "_" not in stem:
            return None

        collector_number = stem.split("_", 1)[0]
        return set_code, collector_number

    @staticmethod
    def _accuracy_band(accuracy: float) -> str:
        if accuracy >= 90:
            return "excellent"
        if accuracy >= 75:
            return "strong"
        if accuracy >= 50:
            return "fair"
        return "needs improvement"

    def test_random_reference_accuracy_no_detector(self) -> None:
        """Randomly sample reference images and evaluate direct recognition accuracy."""

        if not self.INDEX_PATH.exists() or not self.METADATA_PATH.exists():
            pytest.skip("Index/metadata artifacts not found. Run `make build-index` first.")

        if not self.REFERENCE_DIR.exists():
            pytest.skip("Reference images directory not found at data/reference_images.")

        images = sorted(path for path in self.REFERENCE_DIR.rglob("*.jpg") if path.is_file())
        if not images:
            pytest.skip("No reference .jpg images found under data/reference_images.")

        sample_size = int(os.getenv("RECOGNIZER_REFERENCE_SAMPLE_SIZE", "100"))
        sample_size = max(1, min(sample_size, len(images)))
        sampled = random.sample(images, sample_size)

        recognizer = CardRecognizer(
            index_path=self.INDEX_PATH.as_posix(),
            metadata_path=self.METADATA_PATH.as_posix(),
        )

        total = 0
        top1_correct = 0
        top5_correct = 0
        unparsable_paths = 0
        no_prediction_count = 0

        for image_path in sampled:
            expected = self._parse_truth_from_reference_path(image_path, self.REFERENCE_DIR)
            if expected is None:
                unparsable_paths += 1
                continue

            expected_set_code, expected_collector_number = expected

            image_bytes = image_path.read_bytes()
            predictions = recognizer.recognize_from_bytes(image_bytes, top_k=5)
            if not predictions:
                no_prediction_count += 1
                continue

            total += 1

            top1 = predictions[0]
            if (
                str(top1.get("set_code", "")).lower() == expected_set_code.lower()
                and str(top1.get("collector_number", "")) == expected_collector_number
            ):
                top1_correct += 1

            if any(
                str(pred.get("set_code", "")).lower() == expected_set_code.lower()
                and str(pred.get("collector_number", "")) == expected_collector_number
                for pred in predictions
            ):
                top5_correct += 1

        assert total > 0, "No valid evaluation samples were processed."

        top1_accuracy = (top1_correct / total) * 100
        top5_accuracy = (top5_correct / total) * 100

        top1_band = self._accuracy_band(top1_accuracy)
        top5_band = self._accuracy_band(top5_accuracy)

        print("\n=== Recognizer Reference Benchmark (No Detector) ===")
        print("Summary:")
        print(f"  - Requested sample size: {sample_size}")
        print(f"  - Evaluated samples: {total}")
        print(f"  - Skipped (unparsable path): {unparsable_paths}")
        print(f"  - Skipped (no predictions): {no_prediction_count}")
        print("Metrics:")
        print(f"  - Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total})")
        print("    Explanation: exact expected card is the #1 prediction.")
        print(f"  - Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})")
        print("    Explanation: expected card appears anywhere in the top 5 predictions.")
        print("Interpretation:")
        print(f"  - Top-1 quality: {top1_band}")
        print(f"  - Top-5 quality: {top5_band}")
