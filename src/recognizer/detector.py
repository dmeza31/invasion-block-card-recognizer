from __future__ import annotations

import cv2
import numpy as np


class CardDetector:
    """Detect and crop card-like quadrilaterals from photos using OpenCV."""

    OUTPUT_WIDTH = 488
    OUTPUT_HEIGHT = 680

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Normalize lighting with CLAHE histogram equalization.

        Args:
            image: Input image as a NumPy array.

        Returns:
            Lighting-normalized image.
        """

        if image.size == 0:
            raise ValueError("image cannot be empty")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if image.ndim == 2:
            return clahe.apply(image)

        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            normalized = cv2.merge((l_channel, a_channel, b_channel))
            return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

        raise ValueError("image must be a 2D grayscale or 3D color array")

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        """Order quadrilateral points as top-left, top-right, bottom-right, bottom-left."""

        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)

        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def detect_and_crop(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect cards in a photo and return perspective-corrected crops.

        Pipeline:
        1) Convert to grayscale.
        2) Apply Gaussian blur.
        3) Run Canny edge detection.
        4) Find contours and keep quadrilaterals with area > 10% of image area.
        5) Warp each card to 488x680.
        6) Return all crops.

        If no valid card contour is detected, returns the full image as fallback.

        Args:
            image: Input image as a NumPy array.

        Returns:
            List of cropped card images.
        """

        if image.size == 0:
            raise ValueError("image cannot be empty")

        if image.ndim == 2:
            source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            source = image.copy()
        else:
            raise ValueError("image must be a 2D grayscale or 3D color array")

        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = float(source.shape[0] * source.shape[1])
        min_area = image_area * 0.10

        dst = np.array(
            [
                [0, 0],
                [self.OUTPUT_WIDTH - 1, 0],
                [self.OUTPUT_WIDTH - 1, self.OUTPUT_HEIGHT - 1],
                [0, self.OUTPUT_HEIGHT - 1],
            ],
            dtype=np.float32,
        )

        card_crops: list[np.ndarray] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            points = approx.reshape(4, 2).astype(np.float32)
            ordered = self._order_points(points)

            matrix = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(source, matrix, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
            card_crops.append(warped)

        if not card_crops:
            return [source]

        return card_crops
