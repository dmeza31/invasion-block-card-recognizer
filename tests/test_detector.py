from __future__ import annotations

import cv2
import numpy as np

from recognizer.detector import CardDetector


class TestCardDetector:
    """Tests for OpenCV-based card detection and normalization."""

    def test_preprocess_applies_clahe_and_preserves_shape(self) -> None:
        """Preprocess keeps image shape/dtype while normalizing luminance."""

        detector = CardDetector()

        image = np.full((240, 320, 3), 80, dtype=np.uint8)
        cv2.rectangle(image, (40, 50), (280, 190), (95, 95, 95), thickness=-1)

        processed = detector.preprocess(image)

        assert processed.shape == image.shape
        assert processed.dtype == image.dtype
        assert not np.array_equal(processed, image)

    def test_detect_and_crop_warps_detected_quadrilateral_to_card_size(self) -> None:
        """A large quadrilateral card region is detected and warped to 488x680."""

        detector = CardDetector()

        image = np.zeros((900, 900, 3), dtype=np.uint8)

        points = np.array(
            [
                [180, 120],
                [700, 180],
                [650, 800],
                [140, 740],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, points, color=(255, 255, 255))

        crops = detector.detect_and_crop(image)

        assert len(crops) >= 1
        first = crops[0]
        assert first.shape[1] == CardDetector.OUTPUT_WIDTH
        assert first.shape[0] == CardDetector.OUTPUT_HEIGHT

    def test_detect_and_crop_returns_full_image_when_no_card_found(self) -> None:
        """When no valid card contour exists, full input image is returned as fallback."""

        detector = CardDetector()

        image = np.zeros((360, 480, 3), dtype=np.uint8)

        crops = detector.detect_and_crop(image)

        assert len(crops) == 1
        assert crops[0].shape == image.shape
        assert np.array_equal(crops[0], image)
