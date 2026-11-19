
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Select one original training image
IMAGE_PATH = (
    ROOT / "data" / "extracted"
    / "cleaned_data" / "1000" / "sample_0000.png"
)


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_PATH = RESULTS_DIR / "augmentation_preview_1000.png"


# Step 1: Load original image
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH}"
    )

print("Original image shape:", image.shape)


# Step 2: Create augmentation function
def augment_image(img, angle=0, scale=1.0, shear=0.0):

    height, width = img.shape

    center = (
        (width - 1) / 2,
        (height - 1) / 2
    )

    # Rotation and scaling
    rotation = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    rotation_3x3 = np.vstack([
        rotation,
        [0, 0, 1]
    ])

    # Horizontal shearing around image center
    shear_matrix = np.array([
        [1, shear, -shear * center[1]],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combine transformations
    transform = shear_matrix @ rotation_3x3

    # Estimate background intensity from corners
    corners = np.concatenate([
        img[:8, :8].ravel(),
        img[:8, -8:].ravel(),
        img[-8:, :8].ravel(),
        img[-8:, -8:].ravel()
    ])

    background = int(np.median(corners))

    result = cv2.warpAffine(
        img,
        transform[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background
    )

    return result


# Step 3: Generate different image transformations
examples = [
    ("Original", image),
    ("Rotate -8", augment_image(image, angle=-8)),
    ("Rotate +8", augment_image(image, angle=8)),
    ("Shear -0.12", augment_image(image, shear=-0.12)),
    ("Shear +0.12", augment_image(image, shear=0.12)),
    ("Scale 0.90", augment_image(image, scale=0.90)),
    ("Scale 1.08", augment_image(image, scale=1.08)),
]

# Step 4: Draw comparison
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, (title, img) in zip(axes.flat, examples):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

# Hide unused subplot
for ax in list(axes.flat)[len(examples):]:
    ax.axis("off")

plt.tight_layout()

# Step 5: Save image
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print("Preview saved:", OUTPUT_PATH)
print("Augmentation Preview PASSED!")
