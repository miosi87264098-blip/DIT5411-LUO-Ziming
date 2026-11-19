from pathlib import Path
from collections import Counter
import csv
import json
import math
import time

import numpy as np
import tensorflow as tf

import train_full_13065_v2 as pipeline


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CHECKPOINT = (
    MODELS / "full_13065_v2_full_epoch_003.keras"
)

TEST_CSV = ROOT / "data" / "splits" / "test.csv"

NUM_CLASSES = 13065
EXPECTED_IMAGES = 162550
BATCH_SIZE = 32
REPORT_INTERVAL = 250

RESULTS.mkdir(parents=True, exist_ok=True)


def main():

    print("Stage 7 - Independent Test Evaluation", flush=True)
    print("=" * 60)

    print("Loading model:", CHECKPOINT, flush=True)

    model = tf.keras.models.load_model(
        CHECKPOINT,
        compile=False
    )

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError("Wrong model output size")

    print("Output classes:", NUM_CLASSES)
    print("Test images:", EXPECTED_IMAGES)
    print("Batch size:", BATCH_SIZE)
    print("Progress report every:", REPORT_INTERVAL, "batches")

    per_class_total = np.zeros(
        NUM_CLASSES, dtype=np.int64
    )

    per_class_correct = np.zeros(
        NUM_CLASSES, dtype=np.int64
    )

    top1_correct = 0
    top5_correct = 0
    processed = 0
    batch_number = 0

    confusion_pairs = Counter()
    error_samples = []

    started = time.perf_counter()

    def evaluate_batch(paths, labels):

        nonlocal top1_correct
        nonlocal top5_correct
        nonlocal processed
        nonlocal batch_number

        count = len(paths)

        images = np.empty(
            (count, 96, 96, 1),
            dtype=np.float32
        )

        actual = np.asarray(
            labels,
            dtype=np.int32
        )

        for i, path in enumerate(paths):

            image, checked_label = pipeline.load_image(
                path,
                int(actual[i]),
                False,
                2026
            )

            if int(checked_label) != int(actual[i]):
                raise ValueError("Label mismatch")

            images[i] = image

        logits = model(
            images,
            training=False
        ).numpy()

        if not np.isfinite(logits).all():
            raise ValueError("Non-finite predictions")

        predicted = np.argmax(
            logits,
            axis=1
        )

        top5 = np.argpartition(
            logits, -5, axis=1
        )[:, -5:]

        hits1 = predicted == actual

        hits5 = np.any(
            top5 == actual[:, None],
            axis=1
        )

        top1_correct += int(np.sum(hits1))
        top5_correct += int(np.sum(hits5))

        per_class_total[:] += np.bincount(
            actual,
            minlength=NUM_CLASSES
        )

        per_class_correct[:] += np.bincount(
            actual[hits1],
            minlength=NUM_CLASSES
        )

        for i in range(count):
            if not hits1[i]:
                true_class = int(actual[i]) + 1
                pred_class = int(predicted[i]) + 1

                confusion_pairs[
                    (true_class, pred_class)
                ] += 1

                if len(error_samples) < 200:
                    candidates = top5[i]

                    candidates = sorted(
                        candidates.tolist(),
                        key=lambda c: float(
                            logits[i, c]
                        ),
                        reverse=True
                    )

                    error_samples.append({
                        "image_path": paths[i],
                        "true_class_id": true_class,
                        "predicted_class_id": pred_class,
                        "top5_class_ids": [
                            int(c) + 1
                            for c in candidates
                        ]
                    })

        processed += count
        batch_number += 1

        if (
            batch_number % REPORT_INTERVAL == 0
            or processed == EXPECTED_IMAGES
        ):
            elapsed = time.perf_counter() - started

            rate = processed / max(elapsed, 0.001)

            remaining = (
                EXPECTED_IMAGES - processed
            ) / max(rate, 0.001)

            percentage = (
                processed / EXPECTED_IMAGES * 100
            )

            accuracy = (
                top1_correct / processed * 100
            )

            print(
                f"Progress: {processed:,}/"
                f"{EXPECTED_IMAGES:,} "
                f"({percentage:.1f}%) | "
                f"Top-1: {accuracy:.2f}% | "
                f"ETA: {remaining / 60:.1f} min",
                flush=True
            )

    paths = []
    labels = []

    with TEST_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "image_path",
            "class_id",
            "label_index"
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError("Missing CSV columns")

        for row in reader:

            label = int(row["label_index"])
            class_id = int(row["class_id"])

            if not 0 <= label < NUM_CLASSES:
                raise ValueError("Invalid label")

            if class_id != label + 1:
                raise ValueError("Invalid class mapping")

            path = ROOT / row["image_path"]

            paths.append(str(path))
            labels.append(label)

            if len(paths) == BATCH_SIZE:
                evaluate_batch(paths, labels)

                paths.clear()
                labels.clear()

    if paths:
        evaluate_batch(paths, labels)

    if processed != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} test images, "
            f"processed {processed}"
        )

    elapsed = time.perf_counter() - started

    valid = per_class_total > 0

    class_accuracy = np.full(
        NUM_CLASSES, np.nan,
        dtype=np.float64
    )

    class_accuracy[valid] = (
        per_class_correct[valid]
        / per_class_total[valid]
    )

    top1_accuracy = top1_correct / processed
    top5_accuracy = top5_correct / processed
    macro_accuracy = float(
        np.mean(class_accuracy[valid])
    )

    report = {
        "model": CHECKPOINT.name,
        "evaluation_type": "independent_test_set",
        "output_classes": NUM_CLASSES,
        "tested_classes": int(valid.sum()),
        "classes_without_test_images": int((~valid).sum()),
        "test_images": processed,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_class_accuracy": macro_accuracy,
        "evaluation_seconds": elapsed,
        "batch_size": BATCH_SIZE,
        "random_top1_baseline": 1.0 / NUM_CLASSES,
        "common_confusions": [
            {
                "true_class_id": int(pair[0]),
                "predicted_class_id": int(pair[1]),
                "count": int(count)
            }
            for pair, count
            in confusion_pairs.most_common(30)
        ]
    }

    json_path = (
        RESULTS / "full_13065_v2_test_metrics.json"
    )

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report, file,
            indent=2,
            ensure_ascii=False
        )

    class_path = (
        RESULTS / "full_13065_v2_test_per_class.csv"
    )

    with class_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "class_id",
            "test_images",
            "correct",
            "accuracy"
        ])

        for label in range(NUM_CLASSES):
            count = int(per_class_total[label])
            correct = int(per_class_correct[label])

            accuracy_value = (
                correct / count
                if count > 0 else ""
            )

            writer.writerow([
                label + 1,
                count,
                correct,
                accuracy_value
            ])

    error_path = (
        RESULTS / "full_13065_v2_test_errors_sample.csv"
    )

    with error_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_path",
            "true_class_id",
            "predicted_class_id",
            "top5_class_ids"
        ])

        for item in error_samples:
            writer.writerow([
                item["image_path"],
                item["true_class_id"],
                item["predicted_class_id"],
                " ".join(
                    map(str, item["top5_class_ids"])
                )
            ])

    print("=" * 60)
    print("Independent Test Results")
    print("Test images:", processed)
    print("Tested classes:", int(valid.sum()))

    print(
        "Top-1 Accuracy:",
        f"{top1_accuracy * 100:.2f}%"
    )

    print(
        "Top-5 Accuracy:",
        f"{top5_accuracy * 100:.2f}%"
    )

    print(
        "Macro Class Accuracy:",
        f"{macro_accuracy * 100:.2f}%"
    )

    print(
        "Evaluation minutes:",
        round(elapsed / 60, 2)
    )

    print("Metrics:", json_path)
    print("Per-class results:", class_path)
    print("Error samples:", error_path)
    print("Stage 7 Test Evaluation PASSED!")


if __name__ == "__main__":
    main()