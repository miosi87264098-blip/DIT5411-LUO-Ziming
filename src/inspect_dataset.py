
from pathlib import Path
from zipfile import ZipFile
import shutil

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset ZIP file
ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handwritting_data_all"
    / "all_data.zip"
)

print("DIT5411 - Dataset Inspection")
print("=" * 45)

# Check ZIP file
if not ZIP_PATH.exists():
    raise FileNotFoundError("Dataset ZIP file not found!")

with ZipFile(ZIP_PATH, "r") as zip_file:

    # Read archive information
    all_files = zip_file.infolist()

    image_files = [
        item for item in all_files
        if not item.is_dir()
        and item.filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        )
    ]

    # Calculate uncompressed size
    total_size = sum(
        item.file_size for item in all_files
    )

    print("Total entries:", len(all_files))
    print("Image files:", len(image_files))

    print(
        "Uncompressed size (GB):",
        round(total_size / (1024 ** 3), 2)
    )

    print("\nFirst 10 archive entries:")
    for item in all_files[:10]:
        print(repr(item.filename))

# Check disk space
disk = shutil.disk_usage(PROJECT_ROOT)

print("\nDisk information:")
print(
    "Available space (GB):",
    round(disk.free / (1024 ** 3), 2)
)

print(
    "Required extraction space (GB):",
    round(total_size / (1024 ** 3), 2)
)

print("=" * 45)
print("Dataset inspection completed!")
