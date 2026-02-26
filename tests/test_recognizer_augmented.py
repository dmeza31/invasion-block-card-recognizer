from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from recognizer.recognizer import CardRecognizer


class TestCardRecognizerAugmented:
    """Benchmark CardRecognizer against augmented images."""

    INDEX_PATH = Path("data/embeddings/invasion_block.index")
    METADATA_PATH = Path("data/embeddings/card_metadata.json")
    AUGMENTED_DIR = Path("data/augmented_images")
    SPLIT_DIRS = {"train", "val", "test"}

    @staticmethod
    def _parse_truth_from_augmented_path(image_path: Path, augmented_root: Path) -> tuple[str, str] | None:
        """Parse expected (set_code, collector_number) from augmented image path.

        Supported structures:
            data/augmented_images/{set_code}_{collector_number}_{card_name}/aug_X.jpg
            data/augmented_images/{split}/{set_code}_{collector_number}_{card_name}/aug_X.jpg
        """

        try:
            relative = image_path.relative_to(augmented_root)
        except ValueError:
            return None

        if len(relative.parts) < 2:
            return None

        first = relative.parts[0]
        if first in TestCardRecognizerAugmented.SPLIT_DIRS:
            if len(relative.parts) < 3:
                return None
            class_folder = relative.parts[1]
        else:
            class_folder = first

        pieces = class_folder.split("_", 2)
        if len(pieces) < 2:
            return None

        set_code = pieces[0]
        collector_number = pieces[1]
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

    def test_random_augmented_accuracy(self) -> None:
        """Randomly sample augmented images and print Top-1/Top-5 accuracy."""

        if not self.INDEX_PATH.exists() or not self.METADATA_PATH.exists():
            pytest.skip("Index/metadata artifacts not found. Run `make build-index` first.")

        if not self.AUGMENTED_DIR.exists():
            pytest.skip("Augmented dataset not found. Run `make augment-dataset` first.")

        images = sorted(path for path in self.AUGMENTED_DIR.rglob("*.jpg") if path.is_file())
        if not images:
            pytest.skip("No augmented .jpg images found under data/augmented_images.")

        sample_size = int(os.getenv("RECOGNIZER_AUGMENTED_SAMPLE_SIZE", "100"))
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
            expected = self._parse_truth_from_augmented_path(image_path, self.AUGMENTED_DIR)
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

        print("\n=== Recognizer Augmented Benchmark ===")
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
