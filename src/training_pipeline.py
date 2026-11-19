
from pathlib import Path
from collections import Counter
import csv

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augment_image


# ==========================================
# 1. Project Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"

IMAGE_SIZE = 96
BATCH_SIZE = 16
TARGET_PER_CLASS = 200

# Small-scale test: 3 character classes
CLASS_IDS = [1, 1000, 13065]


# ==========================================
# 2. Load Original Training Records
# ==========================================

def load_records():

    records = {
        class_id: []
        for class_id in CLASS_IDS
    }

    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )

    with open(
        TRAIN_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            class_id = int(row["class_id"])

            if class_id in records:

                sample_position = int(
                    row["sample_position"]
                )

                image_path = str(
                    ROOT / row["image_path"]
                )

                records[class_id].append(
                    (sample_position, image_path)
                )

    # Verify records and sample order
    for class_id, samples in records.items():

        samples.sort(
            key=lambda item: item[0]
        )

        if not samples:
            raise ValueError(
                f"No training samples for class {class_id}"
            )

        positions = [
            sample[0]
            for sample in samples
        ]

        expected_positions = list(
            range(len(samples))
        )

        if positions != expected_positions:
            raise ValueError(
                f"Invalid sample order: {class_id}"
            )

        records[class_id] = [
            sample[1]
            for sample in samples
        ]

    return records


# ==========================================
# 3. Generate 200 Samples Per Class
# ==========================================

def sample_generator(records, epoch=0):

    for slot in range(TARGET_PER_CLASS):

        for class_id in CLASS_IDS:

            paths = records[class_id]
            original_count = len(paths)

            if slot < original_count:

                # Use original training image
                path = paths[slot]
                needs_augmentation = False

            else:

                # Reuse an original image
                # with a new transformation
                index = (
                    slot - original_count
                ) % original_count

                path = paths[index]
                needs_augmentation = True

            # Reproducible random seed
            seed = (
                epoch * 20000000
                + class_id * 1000
                + slot
            )

            yield (
                path,
                np.int32(class_id - 1),
                np.bool_(needs_augmentation),
                np.int64(seed)
            )


# ==========================================
# 4. Convert TensorFlow/Numpy Values
# ==========================================

def to_python_scalar(value):
    """
    Convert NumPy scalar values when needed.

    TensorFlow may pass a string as Python bytes
    rather than a NumPy scalar.
    """

    if hasattr(value, "item"):
        return value.item()

    return value


# ==========================================
# 5. OpenCV Image Processing
# ==========================================

def process_image(
    path,
    label,
    augment_flag,
    seed
):

    # Fix: handle both bytes and NumPy scalars
    raw_path = to_python_scalar(path)

    if isinstance(raw_path, bytes):
        image_path = raw_path.decode("utf-8")
    else:
        image_path = str(raw_path)

    # Convert other values safely
    label_value = int(
        to_python_scalar(label)
    )

    needs_augmentation = bool(
        to_python_scalar(augment_flag)
    )

    random_seed = int(
        to_python_scalar(seed)
    )

    # Load grayscale image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    # Apply augmentation only to training copies
    if needs_augmentation:

        image = augment_image(
            image,
            seed=random_seed
        )

    # Resize to 96 x 96
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize image pixels to [0, 1]
    image = image.astype(
        np.float32
    ) / 255.0

    # Add channel dimension
    image = image[..., np.newaxis]

    return (
        image,
        np.int32(label_value)
    )


# ==========================================
# 6. TensorFlow Image Processing
# ==========================================

def tensorflow_process(
    path,
    label,
    augment_flag,
    seed
):

    image, target = tf.numpy_function(
        process_image,
        [
            path,
            label,
            augment_flag,
            seed
        ],
        [
            tf.float32,
            tf.int32
        ]
    )

    # Set TensorFlow tensor shapes
    image.set_shape(
        (IMAGE_SIZE, IMAGE_SIZE, 1)
    )

    target.set_shape(())

    return image, target


# ==========================================
# 7. Build TensorFlow Dataset
# ==========================================


def build_dataset(records, epoch=0):

    total_samples = len(CLASS_IDS) * TARGET_PER_CLASS

    dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator(records, epoch=epoch),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

    # Explicitly declare the number of samples
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(
            total_samples
        )
    )

    # Shuffle training records each iteration
    dataset = dataset.shuffle(
        buffer_size=total_samples,
        seed=2026,
        reshuffle_each_iteration=True
    )

    # Read and augment images with OpenCV
    dataset = dataset.map(
        tensorflow_process,
        num_parallel_calls=2
    )

    # Create batches
    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    dataset = dataset.prefetch(1)

    return dataset


    # Connect OpenCV preprocessing
    dataset = dataset.map(
        tensorflow_process,
        num_parallel_calls=2
    )

    # Create batches
    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Prefetch one batch
    dataset = dataset.prefetch(1)

    return dataset


# ==========================================
# 8. Test Complete Data Pipeline
# ==========================================

def test_pipeline():

    print(
        "DIT5411 - TensorFlow Data Pipeline",
        flush=True
    )

    print("=" * 45)

    # Load training records
    records = load_records()

    planned_counts = Counter()
    planned_augmented = 0

    # Count all planned samples
    for (
        path,
        label,
        is_augmented,
        seed
    ) in sample_generator(records):

        planned_counts[int(label)] += 1

        if is_augmented:
            planned_augmented += 1

    print(
        "Classes:",
        len(planned_counts)
    )

    print(
        "Samples per class:",
        dict(planned_counts)
    )

    print(
        "Total planned samples:",
        sum(planned_counts.values())
    )

    print(
        "Planned augmented samples:",
        planned_augmented
    )

    # Validate planned sample counts
    assert len(planned_counts) == len(CLASS_IDS)

    assert all(
        count == TARGET_PER_CLASS
        for count in planned_counts.values()
    )

    assert sum(
        planned_counts.values()
    ) == 600

    print(
        "Sample Planning PASSED!"
    )

    print("=" * 45)

    # Build TensorFlow dataset
    dataset = build_dataset(records)

    print(
        "Testing TensorFlow batches...",
        flush=True
    )

    total_processed = 0
    batch_count = 0
    actual_counts = Counter()

    # Process every batch (600 images)
    for images, labels in dataset:

        batch_count += 1

        current_batch_size = int(
            images.shape[0]
        )

        # Verify image dimensions
        assert images.shape[1:] == (
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )

        # Verify image values
        assert bool(
            tf.reduce_all(
                tf.math.is_finite(images)
            )
        )

        min_pixel = float(
            tf.reduce_min(images).numpy()
        )

        max_pixel = float(
            tf.reduce_max(images).numpy()
        )

        assert 0.0 <= min_pixel <= 1.0
        assert 0.0 <= max_pixel <= 1.0

        # Verify labels
        for label in labels.numpy():

            actual_counts[int(label)] += 1

        total_processed += current_batch_size

        # Print first batch information
        if batch_count == 1:

            print("\nFirst batch:")

            print(
                "Image shape:",
                images.shape
            )

            print(
                "Label shape:",
                labels.shape
            )

            print(
                "Min pixel:",
                min_pixel
            )

            print(
                "Max pixel:",
                max_pixel
            )

            print(
                "Labels:",
                labels.numpy().tolist()
            )

    print("=" * 45)

    print(
        "Total processed images:",
        total_processed
    )

    print(
        "Total batches:",
        batch_count
    )

    print(
        "Actual samples per class:",
        dict(actual_counts)
    )

    # Final verification
    assert total_processed == 600

    assert actual_counts == planned_counts

    print("=" * 45)

    print(
        "TensorFlow Data Pipeline PASSED!"
    )


# ==========================================
# 9. Main Program
# ==========================================

if __name__ == "__main__":

    test_pipeline()
