from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_DIR_NAMES = {"train", "val", "test"}


def build_augmentation_pipeline(include_gaussian_noise: bool = True) -> transforms.Compose:
    """Create the torchvision pipeline that simulates real-world card photos."""

    pipeline_steps: list[transforms.Transform] = [
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.7),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.5,
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10,
        ),
    ]

    if include_gaussian_noise:
        pipeline_steps.append(transforms.RandomApply([transforms.Lambda(_add_gaussian_noise)], p=0.5))

    return transforms.Compose(pipeline_steps)


def _add_gaussian_noise(image: Image.Image, std: float = 0.04) -> Image.Image:
    """Overlay Gaussian noise on an image and return a PIL image."""

    tensor = transforms.functional.to_tensor(image)
    noise = torch.randn_like(tensor) * std
    noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return transforms.functional.to_pil_image(noisy_tensor)


def _iter_reference_images(input_dir: Path) -> list[Path]:
    """Return all supported image files under the input directory tree."""

    image_paths = [
        image_path
        for image_path in input_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def _parse_card_parts(image_path: Path, input_dir: Path) -> tuple[str, str, str]:
    """Extract set code, collector number, and card name from image path."""

    relative_path = image_path.relative_to(input_dir)
    set_code = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

    stem = image_path.stem
    if "_" in stem:
        collector_number, card_name = stem.split("_", 1)
    else:
        collector_number, card_name = stem, "unknown"

    return set_code, collector_number, card_name


def generate_augmented_dataset(input_dir: str | Path, output_dir: str | Path, num_variants: int = 30) -> None:
    """Generate augmented image variants for each reference image.

    Each source image creates a class directory named
    ``{set_code}_{collector_number}_{card_name}`` and writes ``aug_{i}.jpg``
    files inside it.
    """

    source_root = Path(input_dir)
    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    pipeline = build_augmentation_pipeline(include_gaussian_noise=True)

    image_paths = _iter_reference_images(source_root)
    logger.info("Found %d reference images in %s", len(image_paths), source_root.as_posix())

    generated_count = 0
    for index, image_path in enumerate(image_paths, start=1):
        set_code, collector_number, card_name = _parse_card_parts(image_path, source_root)
        class_dir_name = f"{set_code}_{collector_number}_{card_name}"
        class_output_dir = target_root / class_dir_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image_file:
            base_image = image_file.convert("RGB")

            for variant_idx in range(num_variants):
                augmented_image = pipeline(base_image)
                output_path = class_output_dir / f"aug_{variant_idx + 1}.jpg"
                augmented_image.save(output_path, format="JPEG", quality=95)
                generated_count += 1

        if index % 25 == 0 or index == len(image_paths):
            logger.info(
                "Processed %d/%d source images (generated %d variants)",
                index,
                len(image_paths),
                generated_count,
            )

    logger.info("Augmented dataset generation complete. Total generated images: %d", generated_count)


def _safe_split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """Compute robust train/val/test counts for one class."""

    if total == 0:
        return 0, 0, 0

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                train_count = max(1, train_count - 1)
    elif total == 2:
        train_count, val_count, test_count = 1, 0, 1
    else:
        train_count, val_count, test_count = 1, 0, 0

    return train_count, val_count, test_count


def create_splits(output_dir: str | Path, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    """Create stratified train/val/test splits from augmented class folders."""

    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1")

    logger = logging.getLogger(__name__)
    root = Path(output_dir)

    class_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name not in SPLIT_DIR_NAMES
    ]

    split_roots = {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }

    for split_path in split_roots.values():
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        class_images = [
            image_path
            for image_path in sorted(class_dir.glob("*.jpg"))
            if image_path.is_file()
        ]

        random.shuffle(class_images)
        train_count, val_count, test_count = _safe_split_counts(len(class_images), train_ratio, val_ratio)

        train_images = class_images[:train_count]
        val_images = class_images[train_count : train_count + val_count]
        test_images = class_images[train_count + val_count : train_count + val_count + test_count]

        split_map = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, images in split_map.items():
            class_split_dir = split_roots[split_name] / class_dir.name
            class_split_dir.mkdir(parents=True, exist_ok=True)
            for image_path in images:
                shutil.copy2(image_path, class_split_dir / image_path.name)

        logger.info(
            "Class '%s' split -> train=%d, val=%d, test=%d",
            class_dir.name,
            len(train_images),
            len(val_images),
            len(test_images),
        )

    logger.info("Finished creating stratified splits at %s", root.as_posix())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for augmentation and split generation."""

    parser = argparse.ArgumentParser(description="Generate augmented MTG card dataset and stratified splits")
    parser.add_argument(
        "--input-dir",
        default="data/reference_images",
        help="Directory containing clean reference images",
    )
    parser.add_argument(
        "--output-dir",
        default="data/augmented_images",
        help="Directory where augmented images and splits are written",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=30,
        help="Number of augmented variants to generate per source image",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def main() -> None:
    """Run augmentation and split generation from command-line options."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    random.seed(args.seed)
    generate_augmented_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_variants=args.num_variants,
    )
    create_splits(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
