from pathlib import Path
import csv
import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augment_image


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 13065
TARGET_PER_CLASS = 200

IMAGE_SIZE = 96
BATCH_SIZE = 16
BENCHMARK_STEPS = 8
SEED = 2026

tf.keras.utils.set_random_seed(SEED)


def audit_dataset():

    print("Stage 6A - Full Dataset Audit", flush=True)
    print("=" * 55)

    train_counts = np.zeros(
        NUM_CLASSES, dtype=np.int32
    )

    test_counts = np.zeros(
        NUM_CLASSES, dtype=np.int32
    )

    first_train_image = [None] * NUM_CLASSES
    train_paths = set()

    train_file = SPLITS / "train.csv"
    test_file = SPLITS / "test.csv"

    with open(
        train_file, encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            class_id = int(row["class_id"])
            label = int(row["label_index"])

            if not 1 <= class_id <= NUM_CLASSES:
                raise ValueError("Invalid class ID")

            if label != class_id - 1:
                raise ValueError("Invalid training label")

            path = str(ROOT / row["image_path"])

            if path in train_paths:
                raise ValueError("Duplicate training image")

            train_paths.add(path)
            train_counts[label] += 1

            if first_train_image[label] is None:
                first_train_image[label] = path

    overlap = 0
    test_path_set = set()

    with open(
        test_file, encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            class_id = int(row["class_id"])
            label = int(row["label_index"])

            if not 1 <= class_id <= NUM_CLASSES:
                raise ValueError("Invalid class ID")

            if label != class_id - 1:
                raise ValueError("Invalid test label")

            path = str(ROOT / row["image_path"])

            if path in test_path_set:
                raise ValueError("Duplicate test image")

            test_path_set.add(path)
            test_counts[label] += 1

            if path in train_paths:
                overlap += 1

    train_total = int(np.sum(train_counts))
    test_total = int(np.sum(test_counts))

    train_classes = int(
        np.count_nonzero(train_counts)
    )

    test_classes = int(
        np.count_nonzero(test_counts)
    )

    rare_classes = int(
        np.sum(train_counts < 40)
    )

    no_test_classes = int(
        np.sum(test_counts == 0)
    )

    augmented_needed = int(
        np.sum(
            np.maximum(
                0,
                TARGET_PER_CLASS - train_counts
            )
        )
    )

    final_train_total = (
        train_total + augmented_needed
    )

    print("Training images:", train_total)
    print("Testing images:", test_total)

    print("Training classes:", train_classes)
    print("Testing classes:", test_classes)

    print("Classes below 40:", rare_classes)
    print("Classes without test images:", no_test_classes)

    print("Train/test path overlap:", overlap)

    print(
        "Augmented samples needed:",
        augmented_needed
    )

    print(
        "Target training samples:",
        final_train_total
    )

    assert train_total == 522127
    assert test_total == 162550

    assert train_classes == NUM_CLASSES
    assert test_classes == 13022

    assert rare_classes == 43
    assert no_test_classes == 43

    assert overlap == 0
    assert final_train_total == 2613000

    # Verify a sample of real image files.
    sampled_ids = np.linspace(
        0, NUM_CLASSES - 1,
        num=256,
        dtype=np.int32
    )

    for class_index in sampled_ids:

        path = first_train_image[
            int(class_index)
        ]

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Unreadable image: {path}"
            )

    print("Sampled image checks: 256 PASSED")
    print("Full Dataset Audit PASSED!")

    report = {
        "total_classes": NUM_CLASSES,
        "training_images": train_total,
        "testing_images": test_total,
        "training_classes": train_classes,
        "testing_classes": test_classes,
        "rare_classes": rare_classes,
        "classes_without_test": no_test_classes,
        "train_test_overlap": overlap,
        "augmentation_required": augmented_needed,
        "target_training_samples": final_train_total,
        "sampled_images_checked": 256,
        "full_file_integrity_check": False
    }

    output = RESULTS / "full_dataset_audit.json"

    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Audit saved:", output)

    return first_train_image


def build_benchmark_model():

    # Lightweight CNN with the FINAL output size.
    # This is a speed test, not the final model.

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        tf.keras.layers.Conv2D(
            8, 3, strides=2, activation="relu"
        ),

        tf.keras.layers.Conv2D(
            16, 3, strides=2, activation="relu"
        ),

        tf.keras.layers.Conv2D(
            32, 3, strides=2, activation="relu"
        ),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            NUM_CLASSES
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
    )

    return model


def benchmark_cpu(first_train_image):

    print("=" * 55)
    print("Stage 6A - CPU Benchmark", flush=True)

    rng = np.random.default_rng(SEED)

    total = BATCH_SIZE * BENCHMARK_STEPS

    selected_classes = rng.choice(
        NUM_CLASSES,
        size=total,
        replace=False
    )

    model = build_benchmark_model()

    print("Output classes:", NUM_CLASSES)
    print("Model parameters:", model.count_params())
    print("Batch size:", BATCH_SIZE)
    print("Measured training batches:", BENCHMARK_STEPS)

    def prepare_batch(step):

        start = step * BATCH_SIZE
        end = start + BATCH_SIZE

        labels = selected_classes[start:end]

        images = np.empty(
            (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1),
            dtype=np.float32
        )

        for index, label in enumerate(labels):

            image_path = first_train_image[
                int(label)
            ]

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise FileNotFoundError(image_path)

            image = augment_image(
                image,
                seed=SEED + start + index
            )

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )

            image = 1.0 - (
                image.astype(np.float32) / 255.0
            )

            images[index, :, :, 0] = image

        return images, labels.astype(np.int32)

    # Warm-up: excludes first-step compilation overhead.
    x, y = prepare_batch(0)
    model.train_on_batch(x, y)

    start_time = time.perf_counter()

    for step in range(BENCHMARK_STEPS):

        x, y = prepare_batch(step)

        loss = model.train_on_batch(x, y)

        print(
            f"Benchmark batch {step + 1}/"
            f"{BENCHMARK_STEPS} completed",
            flush=True
        )

    elapsed = time.perf_counter() - start_time

    images_per_second = total / elapsed

    print("=" * 55)
    print("Benchmark seconds:", round(elapsed, 2))

    print(
        "Images per second:",
        round(images_per_second, 2)
    )

    print(
        "Benchmark final loss:",
        round(float(loss), 4)
    )

    print("CPU Benchmark PASSED!")

    print(
        "NOTE: This is a small-model speed test, "
        "not full-model training."
    )

    report = {
        "model_type": "lightweight_cnn_benchmark",
        "final_output_classes": NUM_CLASSES,
        "model_parameters": int(model.count_params()),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "measured_batches": BENCHMARK_STEPS,
        "measured_images": total,
        "elapsed_seconds": elapsed,
        "images_per_second": images_per_second,
        "cpu_threads_reported": os.cpu_count(),
        "note": (
            "Short CPU speed measurement only. "
            "Not an accuracy result or a reliable "
            "full-training time estimate."
        )
    }

    output = RESULTS / "full_cpu_benchmark.json"

    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Benchmark saved:", output)


if __name__ == "__main__":

    first_images = audit_dataset()
    benchmark_cpu(first_images)

    print("=" * 55)
    print("Stage 6A PASSED!")