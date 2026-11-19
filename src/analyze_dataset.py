
from pathlib import Path
from collections import Counter
import csv

# Project paths
ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "data" / "extracted" / "manifest.csv"
DATA_DIR = ROOT / "data" / "extracted" / "cleaned_data"
RESULTS_DIR = ROOT / "results"

EXPECTED_CLASSES = 13065
EXPECTED_IMAGES = 684677

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("DIT5411 - Dataset Analysis")
print("=" * 45)

# Step 1: Check dataset files
if not MANIFEST.is_file():
    raise FileNotFoundError("manifest.csv not found!")

if not DATA_DIR.is_dir():
    raise FileNotFoundError("cleaned_data not found!")

# Step 2: Count images in each class
class_counts = Counter()

with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        class_id = int(row["class_id"])
        class_counts[class_id] += 1

total_classes = len(class_counts)
total_images = sum(class_counts.values())

print("Total classes:", total_classes)
print("Total images:", total_images)

if total_classes != EXPECTED_CLASSES:
    raise ValueError("Unexpected class count!")

if total_images != EXPECTED_IMAGES:
    raise ValueError("Unexpected image count!")

# Step 3: Analyze image distribution
counts = list(class_counts.values())

print("\nImage distribution:")
print("Minimum:", min(counts))
print("Maximum:", max(counts))
print("Average:", round(total_images / total_classes, 2))

# Step 4: Check assignment requirements
below_40 = sum(n < 40 for n in counts)
exactly_40 = sum(n == 40 for n in counts)
above_40 = sum(n > 40 for n in counts)

train_count = sum(min(40, n) for n in counts)
test_count = sum(max(n - 40, 0) for n in counts)

print("\nAssignment requirements:")
print("Classes below 40 samples:", below_40)
print("Classes with exactly 40 samples:", exactly_40)
print("Classes above 40 samples:", above_40)

print("Potential training images:", train_count)
print("Potential testing images:", test_count)

# Step 5: Save class statistics
OUTPUT_CSV = RESULTS_DIR / "class_counts.csv"

with open(
    OUTPUT_CSV, "w",
    encoding="utf-8",
    newline=""
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "total_samples",
        "training_samples",
        "testing_samples"
    ])

    for class_id in sorted(class_counts):
        n = class_counts[class_id]

        writer.writerow([
            class_id,
            n,
            min(40, n),
            max(0, n - 40)
        ])

print("\nFirst 10 classes:")
for class_id in sorted(class_counts)[:10]:
    print(
        f"Class {class_id}: "
        f"{class_counts[class_id]} samples"
    )

print("\nResults saved:", OUTPUT_CSV)
print("=" * 45)
print("Dataset Analysis PASSED!")
