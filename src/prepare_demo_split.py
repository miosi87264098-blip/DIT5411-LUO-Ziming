
from pathlib import Path
import csv
import random

# Project paths
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"

SOURCE_TRAIN = SPLIT_DIR / "train.csv"
SOURCE_TEST = SPLIT_DIR / "test.csv"

DEMO_TRAIN = SPLIT_DIR / "demo_train.csv"
DEMO_VAL = SPLIT_DIR / "demo_val.csv"
DEMO_TEST = SPLIT_DIR / "demo_test.csv"

# Selected classes for the MLP demo
CLASS_IDS = [1, 1000, 13065]

TRAIN_PER_CLASS = 32
VAL_PER_CLASS = 8
SEED = 2026

FIELDNAMES = [
    "image_path",
    "class_id",
    "label_index",
    "sample_position"
]


def load_records(csv_path):
    records = {
        class_id: []
        for class_id in CLASS_IDS
    }

    with open(
        csv_path, "r",
        encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            class_id = int(row["class_id"])

            if class_id in records:
                records[class_id].append(row)

    return records


def make_output_row(row, class_id):
    return {
        "image_path": row["image_path"],
        "class_id": class_id,
        "label_index": CLASS_IDS.index(class_id),
        "sample_position": int(row["sample_position"])
    }


def save_csv(path, records):
    with open(
        path, "w",
        encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(records)


def main():
    print("DIT5411 - Demo Dataset Split")
    print("=" * 45)

    original_train = load_records(SOURCE_TRAIN)
    original_test = load_records(SOURCE_TEST)

    train_records = []
    val_records = []
    test_records = []

    for class_id in CLASS_IDS:
        samples = original_train[class_id]

        # The demo uses 40 original training samples
        if len(samples) != 40:
            raise ValueError(
                f"Class {class_id}: expected 40 samples, "
                f"got {len(samples)}"
            )

        # Deterministic per-class random split
        rng = random.Random(SEED + class_id)
        samples = samples.copy()
        rng.shuffle(samples)

        val_samples = samples[:VAL_PER_CLASS]
        train_samples = samples[VAL_PER_CLASS:]

        assert len(train_samples) == TRAIN_PER_CLASS
        assert len(val_samples) == VAL_PER_CLASS

        for row in train_samples:
            train_records.append(
                make_output_row(row, class_id)
            )

        for row in val_samples:
            val_records.append(
                make_output_row(row, class_id)
            )

        for row in original_test[class_id]:
            test_records.append(
                make_output_row(row, class_id)
            )

        print(
            f"Class {class_id}: "
            f"Train={len(train_samples)}, "
            f"Val={len(val_samples)}, "
            f"Test={len(original_test[class_id])}"
        )

    # Verify no image is shared between datasets
    train_paths = {
        row["image_path"] for row in train_records
    }
    val_paths = {
        row["image_path"] for row in val_records
    }
    test_paths = {
        row["image_path"] for row in test_records
    }

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # Verify that referenced image files exist
    for path in train_paths | val_paths | test_paths:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)

    save_csv(DEMO_TRAIN, train_records)
    save_csv(DEMO_VAL, val_records)
    save_csv(DEMO_TEST, test_records)

    print("=" * 45)
    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Testing images:", len(test_records))

    print("Data Leakage Check: PASSED")
    print("Demo Dataset Split PASSED!")


if __name__ == "__main__":
    main()
