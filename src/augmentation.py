
from pathlib import Path
from collections import Counter
import csv

import cv2
import numpy as np

# Project configuration
ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
RESULTS_DIR = ROOT / "results"

TARGET_PER_CLASS = 200
EXPECTED_CLASSES = 13065
EXPECTED_TRAIN_IMAGES = 522127

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Random image augmentation
def augment_image(image, seed=42):
    """
    Apply random rotation, scaling and shearing.
    Only use this function for training images.
    """
    rng = np.random.default_rng(seed)

    height, width = image.shape
    center = ((width - 1) / 2, (height - 1) / 2)

    # Generate random transformation parameters
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.90, 1.08)
    shear = rng.uniform(-0.12, 0.12)

    # Rotation and scaling
    rotation = cv2.getRotationMatrix2D(
        center, angle, scale
    )

    rotation_matrix = np.vstack([
        rotation,
        [0, 0, 1]
    ])

    # Horizontal shear around image center
    shear_matrix = np.array([
        [1, shear, -shear * center[1]],
        [0, 1, 0],
        [0, 0, 1]
    ])

    transform = shear_matrix @ rotation_matrix

    # Estimate the image background
    corners = np.concatenate([
        image[:8, :8].ravel(),
        image[:8, -8:].ravel(),
        image[-8:, :8].ravel(),
        image[-8:, -8:].ravel()
    ])

    background = int(np.median(corners))

    # Apply OpenCV transformation
    augmented = cv2.warpAffine(
        image,
        transform[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background
    )

    return augmented


# Step 2: Read the training dataset
def analyze_augmentation_plan():
    class_counts = Counter()

    with open(
        TRAIN_CSV, "r",
        encoding="utf-8",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            class_id = int(row["class_id"])
            class_counts[class_id] += 1

    total_original = sum(class_counts.values())

    if len(class_counts) != EXPECTED_CLASSES:
        raise ValueError("Incorrect number of classes!")

    if total_original != EXPECTED_TRAIN_IMAGES:
        raise ValueError("Incorrect training image count!")

    # Calculate additional augmented samples needed
    additional_samples = sum(
        max(0, TARGET_PER_CLASS - count)
        for count in class_counts.values()
    )

    total_target = total_original + additional_samples

    print("Training classes:", len(class_counts))
    print("Original training images:", total_original)
    print("Additional augmented samples:", additional_samples)
    print("Target training samples:", total_target)

    if total_target != EXPECTED_CLASSES * TARGET_PER_CLASS:
        raise ValueError("Augmentation target mismatch!")

    print("Augmentation Plan PASSED!")


# Step 3: Test the augmentation function
def test_augmentation():
    image_path = (
        ROOT / "data" / "extracted"
        / "cleaned_data" / "1000"
        / "sample_0000.png"
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(image_path)

    augmented = augment_image(image, seed=2026)

    output_path = (
        RESULTS_DIR / "augmentation_random_test.png"
    )

    success = cv2.imwrite(
        str(output_path),
        augmented
    )

    if not success:
        raise RuntimeError("Failed to save image!")

    print("Random augmentation saved:", output_path)
    print("Random Augmentation Test PASSED!")


if __name__ == "__main__":
    print("DIT5411 - Image Augmentation")
    print("=" * 45)

    analyze_augmentation_plan()

    print("=" * 45)

    test_augmentation()

    print("=" * 45)
    print("Augmentation Module Test PASSED!")
