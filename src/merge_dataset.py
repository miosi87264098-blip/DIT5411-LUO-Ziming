
from pathlib import Path
import shutil
import zipfile

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handwritting_data_all"
)

# Output ZIP file
OUTPUT_FILE = DATA_DIR / "all_data.zip"

# 12 ZIP parts
parts = [
    DATA_DIR / f"all_data.zip.{i:03d}"
    for i in range(1, 13)
]

print("DIT5411 - Dataset Preparation")
print("=" * 40)

# Step 1: Check all ZIP parts
for part in parts:
    if not part.is_file():
        raise FileNotFoundError(
            f"Missing file: {part.name}"
        )

    print(f"Found: {part.name}")

# Step 2: Merge ZIP parts
if OUTPUT_FILE.exists():
    print("ZIP already exists. Skipping merge.")
else:
    temp_file = DATA_DIR / "all_data.zip.partial"

    with open(temp_file, "wb") as output:
        for part in parts:
            print(f"Merging: {part.name}")

            with open(part, "rb") as source:
                shutil.copyfileobj(
                    source,
                    output,
                    length=1024 * 1024
                )

    temp_file.rename(OUTPUT_FILE)

# Step 3: Check ZIP file
print("=" * 40)
print("Output:", OUTPUT_FILE)

size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print("Size (MB):", round(size_mb, 2))

if zipfile.is_zipfile(OUTPUT_FILE):
    print("ZIP structure check: PASSED")
else:
    raise RuntimeError("ZIP structure check: FAILED")

print("Dataset merge finished.")
