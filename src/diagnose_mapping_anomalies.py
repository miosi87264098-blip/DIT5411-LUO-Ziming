from pathlib import Path
from collections import Counter
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "extracted" / "manifest.csv"
OUT = ROOT / "results" / "stage8" / "mapping_v2"

REVIEW = OUT / "character_mapping_review.csv"
SUMMARY = OUT / "character_mapping_summary.json"

mismatches = []
affected_classes = Counter()
actual_folders = Counter()
mismatch_types = Counter()
total = 0

print("Stage 8C-3 - Mapping Anomaly Diagnosis")
print("=" * 65, flush=True)

with MANIFEST.open(
    "r", encoding="utf-8-sig", newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        total += 1
        class_id = int(row["class_id"])
        raw = bytes.fromhex(row["original_filename_hex"])

        # ZIP entries use / as their path separator.
        parts = raw.split(b"/")

        folder = (
            parts[-2] if len(parts) >= 2 else b""
        )

        expected = str(class_id).encode("ascii")

        if folder != expected:
            affected_classes[class_id] += 1

            folder_text = folder.decode(
                "ascii", errors="backslashreplace"
            )

            actual_folders[folder_text] += 1

            try:
                actual_id = int(folder.decode("ascii"))
                issue_type = (
                    "different_numeric_folder"
                    if actual_id != class_id
                    else "other"
                )
            except (ValueError, UnicodeError):
                issue_type = "non_numeric_folder"

            mismatch_types[issue_type] += 1

            if len(mismatches) < 25:
                filename_hex = parts[-1].hex()

                mismatches.append({
                    "manifest_class_id": class_id,
                    "source_folder": folder_text,
                    "source_filename_hex": filename_hex,
                    "source_path_hex": raw.hex(),
                    "issue_type": issue_type
                })

        if total % 200000 == 0:
            print(
                f"Inspected {total:,} manifest entries",
                flush=True
            )

with REVIEW.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    review_rows = list(csv.DictReader(file))

with SUMMARY.open(
    "r", encoding="utf-8"
) as file:
    old_summary = json.load(file)

status_counts = Counter(
    row["status"] for row in review_rows
)

report = {
    "total_manifest_rows": total,
    "mismatch_count": sum(mismatch_types.values()),
    "affected_class_count": len(affected_classes),
    "mismatch_types": dict(mismatch_types),
    "top_affected_classes": affected_classes.most_common(20),
    "top_source_folders": actual_folders.most_common(20),
    "mismatch_examples": mismatches,
    "review_class_count": len(review_rows),
    "review_status_counts": dict(status_counts),
    "review_examples": review_rows[:30],
    "previous_high_confidence_count": old_summary.get(
        "high_confidence"
    )
}

json_path = OUT / "mapping_anomaly_diagnosis.json"
txt_path = OUT / "mapping_anomaly_diagnosis.txt"

json_path.write_text(
    json.dumps(
        report, indent=2, ensure_ascii=False
    ),
    encoding="utf-8"
)

lines = [
    "DIT5411 - Character Mapping Anomaly Diagnosis",
    "=" * 65,
    f"Manifest rows: {total}",
    f"Directory mismatches: {report['mismatch_count']}",
    f"Affected classes: {len(affected_classes)}",
    f"Review classes: {len(review_rows)}",
    "",
    "Mismatch types:",
    json.dumps(
        dict(mismatch_types),
        ensure_ascii=False, indent=2
    ),
    "",
    "Top affected classes:",
]

for class_id, count in affected_classes.most_common(20):
    lines.append(
        f"Class {class_id}: {count} mismatched records"
    )

lines.extend(["", "First 25 mismatch examples:"])

for item in mismatches:
    lines.append(
        f"Class {item['manifest_class_id']} | "
        f"Source folder: {item['source_folder']} | "
        f"Type: {item['issue_type']} | "
        f"Filename hex: {item['source_filename_hex']}"
    )

lines.extend([
    "",
    "Classes requiring mapping review:",
    "Status counts:",
    json.dumps(
        dict(status_counts),
        ensure_ascii=False, indent=2
    ),
    "",
    "First 30 review cases:",
])

for row in review_rows[:30]:
    lines.append(
        f"Class {row['class_id']} | "
        f"Status: {row['status']} | "
        f"CP950: {row['cp950_character']!r} | "
        f"Big5-HKSCS: {row['big5hkscs_character']!r} | "
        f"Samples: {row['sample_count']}"
    )

txt_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("=" * 65)
print("Manifest rows:", total)
print("Directory mismatches:", report["mismatch_count"])
print("Affected classes:", len(affected_classes))
print("Mismatch types:", dict(mismatch_types))
print("Classes requiring review:", len(review_rows))
print("Review status counts:", dict(status_counts))

print("")
print("First 10 directory mismatches:")

for item in mismatches[:10]:
    print(
        f"Class {item['manifest_class_id']} "
        f"-> source folder {item['source_folder']} "
        f"[{item['issue_type']}]"
    )

print("")
print("First 15 review classes:")

for row in review_rows[:15]:
    print(
        f"Class {row['class_id']}: "
        f"{row['status']} | "
        f"CP950={row['cp950_character']!r} | "
        f"HKSCS={row['big5hkscs_character']!r}"
    )

print("")
print("Text report:", txt_path)
print("JSON report:", json_path)
print("Stage 8C-3 Diagnosis PASSED!")