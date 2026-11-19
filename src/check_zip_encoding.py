
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handwritting_data_all"
    / "all_data.zip"
)

print("DIT5411 - ZIP Encoding Diagnosis")
print("=" * 45)

total_images = 0
invalid_big5 = 0
utf8_names = 0
affected_classes = Counter()
examples = []

# Use the default CP437 decoding to inspect raw filenames
with ZipFile(ZIP_PATH, "r") as zf:
    for info in zf.infolist():

        if info.is_dir():
            continue

        if not info.filename.lower().endswith(".png"):
            continue

        total_images += 1

        # ZIP entries explicitly marked as UTF-8
        if info.flag_bits & 0x800:
            utf8_names += 1
            continue

        # Recover original filename bytes
        raw_name = info.filename.encode("cp437")

        try:
            raw_name.decode("big5")

        except UnicodeDecodeError as error:
            invalid_big5 += 1

            parts = info.filename.split("/")
            if len(parts) >= 3:
                affected_classes[parts[1]] += 1

            if len(examples) < 5:
                examples.append({
                    "filename": info.filename,
                    "raw_hex": raw_name.hex(),
                    "error": str(error)
                })

print("Total images:", total_images)
print("UTF-8 filename entries:", utf8_names)
print("Big5 decoding failures:", invalid_big5)
print("Affected classes:", len(affected_classes))

print("\nFirst 5 problematic filenames:")

for example in examples:
    print("-" * 45)
    print("Filename:", repr(example["filename"]))
    print("Raw bytes:", example["raw_hex"])
    print("Error:", example["error"])

print("=" * 45)
print("Encoding diagnosis completed.")
