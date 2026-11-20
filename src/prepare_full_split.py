from pathlib import Path
import csv
import json
import random

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE_TRAIN = SPLITS / "train.csv"
SOURCE_TEST = SPLITS / "test.csv"

OUTPUT_TRAIN = SPLITS / "full_train.csv"
OUTPUT_VAL = SPLITS / "full_val.csv"

NUM_CLASSES = 13065
TARGET_PER_CLASS = 200
SEED = 2026


def validation_size(count):
    if count < 2:
        raise ValueError("Not enough training samples")
    return max(1, min(8, count // 5))


def main():
    print("Stage 6B - Full Train/Validation Split")
    print("=" * 55, flush=True)

    # Pass 1: Count original training images.
    counts = [0] * (NUM_CLASSES + 1)

    with SOURCE_TRAIN.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("Missing CSV columns")

        required = {"image_path", "class_id", "label_index"}

        if not required.issubset(set(fieldnames)):
            raise ValueError("Missing required columns")

        for row in reader:
            class_id = int(row["class_id"])
            label = int(row["label_index"])

            if not 1 <= class_id <= NUM_CLASSES:
                raise ValueError("Invalid class ID")

            if label != class_id - 1:
                raise ValueError("Invalid label index")

            counts[class_id] += 1

    if sum(counts) != 522127:
        raise ValueError("Unexpected original training count")

    if any(count == 0 for count in counts[1:]):
        raise ValueError("Some classes have no training samples")

    # Select validation records reproducibly.
    val_indices = {}

    for class_id in range(1, NUM_CLASSES + 1):
        count = counts[class_id]
        val_count = validation_size(count)

        rng = random.Random(SEED + class_id)

        val_indices[class_id] = set(
            rng.sample(range(count), val_count)
        )

    # Pass 2: Write train and validation manifests.
    seen = [0] * (NUM_CLASSES + 1)
    train_counts = [0] * (NUM_CLASSES + 1)
    val_counts = [0] * (NUM_CLASSES + 1)

    train_tmp = SPLITS / "full_train.csv.tmp"
    val_tmp = SPLITS / "full_val.csv.tmp"

    train_paths = set()
    val_paths = set()

    with (
        SOURCE_TRAIN.open(
            "r", encoding="utf-8-sig", newline=""
        ) as source,
        train_tmp.open(
            "w", encoding="utf-8", newline=""
        ) as train_file,
        val_tmp.open(
            "w", encoding="utf-8", newline=""
        ) as val_file
    ):
        reader = csv.DictReader(source)

        train_writer = csv.DictWriter(
            train_file, fieldnames=fieldnames
        )

        val_writer = csv.DictWriter(
            val_file, fieldnames=fieldnames
        )

        train_writer.writeheader()
        val_writer.writeheader()

        for row in reader:
            class_id = int(row["class_id"])

            position = seen[class_id]
            seen[class_id] += 1

            path = row["image_path"]

            if position in val_indices[class_id]:
                val_writer.writerow(row)
                val_counts[class_id] += 1

                if path in train_paths or path in val_paths:
                    raise ValueError("Duplicate image path")

                val_paths.add(path)

            else:
                train_writer.writerow(row)
                train_counts[class_id] += 1

                if path in train_paths or path in val_paths:
                    raise ValueError("Duplicate image path")

                train_paths.add(path)

    # Verify completeness and class coverage.
    for class_id in range(1, NUM_CLASSES + 1):
        assert seen[class_id] == counts[class_id]
        assert train_counts[class_id] > 0
        assert val_counts[class_id] > 0
        assert (
            train_counts[class_id] + val_counts[class_id]
            == counts[class_id]
        )

    train_total = sum(train_counts)
    val_total = sum(val_counts)

    assert train_total + val_total == 522127
    assert train_paths.isdisjoint(val_paths)

    # Original test set remains untouched.
    test_counts = [0] * (NUM_CLASSES + 1)
    test_total = 0

    with SOURCE_TEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            class_id = int(row["class_id"])
            label = int(row["label_index"])
            path = row["image_path"]

            if not 1 <= class_id <= NUM_CLASSES:
                raise ValueError("Invalid test class")

            if label != class_id - 1:
                raise ValueError("Invalid test label")

            if path in train_paths or path in val_paths:
                raise ValueError("Test data leakage detected")

            test_counts[class_id] += 1
            test_total += 1

    assert test_total == 162550

    # Check all classes can reach 200 train samples.
    augmentation_needed = sum(
        TARGET_PER_CLASS - train_counts[class_id]
        for class_id in range(1, NUM_CLASSES + 1)
    )

    assert (
        train_total + augmentation_needed
        == NUM_CLASSES * TARGET_PER_CLASS
    )

    # Only publish files after all checks pass.
    train_tmp.replace(OUTPUT_TRAIN)
    val_tmp.replace(OUTPUT_VAL)

    no_test = sum(
        test_counts[class_id] == 0
        for class_id in range(1, NUM_CLASSES + 1)
    )

    report = {
        "classes": NUM_CLASSES,
        "original_train_candidates": 522127,
        "full_train_originals": train_total,
        "full_validation_images": val_total,
        "original_test_images": test_total,
        "classes_without_test_images": no_test,
        "augmentation_required_after_validation": (
            augmentation_needed
        ),
        "target_training_samples": (
            NUM_CLASSES * TARGET_PER_CLASS
        ),
        "random_seed": SEED,
        "validation_policy": (
            "20 percent approximately, "
            "maximum 8 images per class"
        ),
        "leakage_check_scope": "file_paths",
        "split_leakage_detected": False
    }

    report_path = RESULTS / "full_split_audit.json"

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report, file,
            indent=2,
            ensure_ascii=False
        )

    print("Original training candidates:", 522127)
    print("Full training originals:", train_total)
    print("Full validation images:", val_total)
    print("Original testing images:", test_total)
    print("Training classes:", NUM_CLASSES)
    print("Validation classes:", NUM_CLASSES)
    print("Classes without test images:", no_test)
    print("Augmentation needed:", augmentation_needed)
    print(
        "Target training samples:",
        NUM_CLASSES * TARGET_PER_CLASS
    )

    print("Split file-path disjointness check: PASSED!")
    print("Output train:", OUTPUT_TRAIN)
    print("Output validation:", OUTPUT_VAL)
    print("Report:", report_path)
    print("=" * 55)
    print("Stage 6B PASSED!")


if __name__ == "__main__":
    main()
