
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv
import os
import re
import shutil

# Project paths
ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = (
    ROOT / "data" / "raw"
    / "handwritting_data_all" / "all_data.zip"
)

OUTPUT_DIR = ROOT / "data" / "extracted"
DATA_DIR = OUTPUT_DIR / "cleaned_data"

MANIFEST = OUTPUT_DIR / "manifest.csv"
MARKER = OUTPUT_DIR / ".extraction_complete"

EXPECTED_IMAGES = 684677
EXPECTED_CLASSES = 13065

print("DIT5411 - Safe Dataset Extraction")
print("=" * 45, flush=True)

if MARKER.exists():
    print("Dataset already extracted.")
    raise SystemExit(0)

if not ZIP_PATH.exists():
    raise FileNotFoundError(ZIP_PATH)

# Step 1: Read ZIP without forcing Big5
with ZipFile(ZIP_PATH, "r") as zf:

    classes = defaultdict(list)
    total_size = 0
    missing_indices = 0

    for info in zf.infolist():

        if info.is_dir():
            continue

        if not info.filename.lower().endswith(".png"):
            raise ValueError(
                f"Unexpected file: {info.filename}"
            )

        # Obtain numeric class ID
        parts = info.filename.split("/", 2)

        if (
            len(parts) != 3
            or parts[0] != "cleaned_data"
            or not parts[1].isdigit()
        ):
            raise ValueError(
                f"Unexpected image path: {info.filename}"
            )

        class_id = int(parts[1])

        if not 1 <= class_id <= EXPECTED_CLASSES:
            raise ValueError(
                f"Invalid class ID: {class_id}"
            )

        # Recover original sample number
        match = re.search(
            r"_(\d+)\.png$",
            parts[2],
            re.IGNORECASE
        )

        sample_index = (
            int(match.group(1))
            if match else None
        )

        if sample_index is None:
            missing_indices += 1

        classes[class_id].append(
            (sample_index, info.filename, info)
        )

        total_size += info.file_size

    image_count = sum(
        len(items) for items in classes.values()
    )

    print("Images:", image_count)
    print("Classes:", len(classes))
    print("Missing sample indices:", missing_indices)

    if (
        image_count != EXPECTED_IMAGES
        or len(classes) != EXPECTED_CLASSES
    ):
        raise RuntimeError(
            "Dataset count validation failed!"
        )

    # Step 2: Check available disk space
    free_space = shutil.disk_usage(ROOT).free

    if free_space < total_size + 2 * 1024**3:
        raise RuntimeError("Not enough disk space!")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    manifest_temp = OUTPUT_DIR / "manifest.csv.partial"

    # Step 3: Extract using safe numeric filenames
    with open(
        manifest_temp, "w",
        newline="", encoding="utf-8"
    ) as manifest_file:

        writer = csv.writer(manifest_file)

        writer.writerow([
            "class_id",
            "sample_position",
            "original_sample_index",
            "original_zip_filename",
            "original_filename_hex"
        ])

        for class_id in sorted(classes):

            class_dir = DATA_DIR / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            # Sort by original numeric sample index
            items = sorted(
                classes[class_id],
                key=lambda item: (
                    item[0] if item[0] is not None
                    else float("inf"),
                    item[1]
                )
            )

            for position, (
                sample_index, original_name, info
            ) in enumerate(items):

                output_file = (
                    class_dir /
                    f"sample_{position:04d}.png"
                )

                temp_file = output_file.with_suffix(".tmp")

                # Reading to EOF verifies ZIP CRC
                try:
                    with zf.open(info) as source:
                        with open(temp_file, "wb") as target:
                            shutil.copyfileobj(source, target)

                    os.replace(temp_file, output_file)

                except Exception:
                    temp_file.unlink(missing_ok=True)
                    raise

                writer.writerow([
                    class_id,
                    position,
                    sample_index,
                    original_name,
                    original_name.encode("cp437").hex()
                ])

                extracted += 1

                if extracted % 25000 == 0:
                    print(
                        f"Extracted {extracted}/{image_count}",
                        flush=True
                    )

    # Step 4: Final validation
    if extracted != EXPECTED_IMAGES:
        raise RuntimeError("Image count mismatch!")

    manifest_temp.replace(MANIFEST)

    MARKER.write_text(
        "Extraction completed successfully.\n",
        encoding="utf-8"
    )

print("=" * 45)
print("Extracted images:", extracted)
print("Character classes:", len(classes))
print("Manifest:", MANIFEST)
print("Dataset Extraction PASSED!")
