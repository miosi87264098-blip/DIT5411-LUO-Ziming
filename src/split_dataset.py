
from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "data" / "extracted" / "manifest.csv"
COUNTS_FILE = ROOT / "results" / "class_counts.csv"

SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
RARE_CSV = RESULTS_DIR / "rare_classes.csv"

EXPECTED_CLASSES = 13065
EXPECTED_IMAGES = 684677
TRAIN_PER_CLASS = 40

print("DIT5411 - Dataset Splitting")
print("=" * 45)

# Step 1: Read class statistics
class_counts = {}

with open(COUNTS_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        class_id = int(row["class_id"])
        class_counts[class_id] = int(row["total_samples"])

if set(class_counts) != set(range(1, EXPECTED_CLASSES + 1)):
    raise ValueError("Unexpected class IDs!")

# Step 2: Calculate training limits
train_limits = {
    class_id: min(TRAIN_PER_CLASS, count)
    for class_id, count in class_counts.items()
}

rare_classes = {
    class_id: count
    for class_id, count in class_counts.items()
    if count < TRAIN_PER_CLASS
}

# Step 3: Split using the original sample order
seen = Counter()
train_total = 0
test_total = 0

header = [
    "image_path",
    "class_id",
    "label_index",
    "sample_position",
    "original_sample_index"
]

with (
    open(MANIFEST, "r", encoding="utf-8", newline="") as source,
    open(TRAIN_CSV, "w", encoding="utf-8", newline="") as train_file,
    open(TEST_CSV, "w", encoding="utf-8", newline="") as test_file
):
    reader = csv.DictReader(source)
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    train_writer.writerow(header)
    test_writer.writerow(header)

    for row in reader:
        class_id = int(row["class_id"])
        position = int(row["sample_position"])

        if class_id not in class_counts:
            raise ValueError(f"Unknown class: {class_id}")

        if position != seen[class_id]:
            raise ValueError(
                f"Unexpected sample order for class {class_id}"
            )

        relative_path = (
            f"data/extracted/cleaned_data/"
            f"{class_id}/sample_{position:04d}.png"
        )

        record = [
            relative_path,
            class_id,
            class_id - 1,
            position,
            row["original_sample_index"]
        ]

        if position < train_limits[class_id]:
            train_writer.writerow(record)
            train_total += 1
        else:
            test_writer.writerow(record)
            test_total += 1

        seen[class_id] += 1

# Step 4: Validate the result
if dict(seen) != class_counts:
    raise RuntimeError("Class sample counts do not match!")

if train_total + test_total != EXPECTED_IMAGES:
    raise RuntimeError("Total image count mismatch!")

# Step 5: Save the exceptional classes
with open(RARE_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_id", "available_samples"])

    for class_id, count in sorted(rare_classes.items()):
        writer.writerow([class_id, count])

testable_classes = sum(
    count > TRAIN_PER_CLASS
    for count in class_counts.values()
)

print("Total classes:", len(class_counts))
print("Training images:", train_total)
print("Testing images:", test_total)
print("Classes below 40:", len(rare_classes))
print("Classes represented in test set:", testable_classes)

print("\nFiles created:")
print(TRAIN_CSV)
print(TEST_CSV)
print(RARE_CSV)

print("=" * 45)
print("Dataset Splitting PASSED!")
