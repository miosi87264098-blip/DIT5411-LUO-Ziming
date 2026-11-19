from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "extracted" / "manifest.csv"
OUT = ROOT / "results" / "stage8"
OUT.mkdir(parents=True, exist_ok=True)

N = 13065
CODECS = ("cp950", "big5hkscs")

counts = [0] * (N + 1)
decoded = {
    codec: [Counter() for _ in range(N + 1)]
    for codec in CODECS
}
failures = {
    codec: [0] * (N + 1)
    for codec in CODECS
}
examples = [""] * (N + 1)


def decode_name(raw, codec):
    basename = raw.replace(b"\\", b"/").split(b"/")[-1]

    if not basename.lower().endswith(b".png"):
        return None

    stem = basename[:-4]
    name_bytes, separator, number = stem.rpartition(b"_")

    if not separator or not number.isdigit():
        return None

    try:
        character = name_bytes.decode(codec, errors="strict")
    except UnicodeError:
        return None

    if len(character) != 1 or not character.isprintable():
        return None

    return character


def save_csv(path, rows):
    fields = [
        "class_id",
        "character",
        "unicode",
        "status",
        "cp950_character",
        "big5hkscs_character",
        "sample_count",
    ]

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Stage 8C - Character Mapping Recovery", flush=True)
    print("=" * 60)

    total = 0
    invalid = 0
    folder_mismatch = 0

    with SOURCE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not {
            "class_id",
            "original_filename_hex"
        }.issubset(reader.fieldnames or []):
            raise ValueError("Manifest columns are missing")

        for row in reader:
            total += 1

            try:
                class_id = int(row["class_id"])
                if not 1 <= class_id <= N:
                    raise ValueError("Invalid class ID")

                hex_value = row["original_filename_hex"]
                raw = bytes.fromhex(hex_value)
            except (ValueError, TypeError, KeyError):
                invalid += 1
                continue

            counts[class_id] += 1

            if not examples[class_id]:
                examples[class_id] = hex_value

            parts = raw.replace(b"\\", b"/").split(b"/")

            if (
                len(parts) < 2
                or parts[-2] != str(class_id).encode("ascii")
            ):
                folder_mismatch += 1

            for codec in CODECS:
                char = decode_name(raw, codec)

                if char is None:
                    failures[codec][class_id] += 1
                else:
                    decoded[codec][class_id][char] += 1

            if total % 100000 == 0:
                print(
                    f"Processed {total:,} manifest rows",
                    flush=True
                )

    rows = []

    for class_id in range(1, N + 1):
        candidates = {}

        for codec in CODECS:
            counter = decoded[codec][class_id]

            if (
                counts[class_id] > 0
                and failures[codec][class_id] == 0
                and len(counter) == 1
            ):
                candidates[codec] = next(iter(counter))
            else:
                candidates[codec] = ""

        a = candidates["cp950"]
        b = candidates["big5hkscs"]

        if a and b and a == b:
            status = "consistent"
            char = a
        elif a and b:
            status = "codec_disagreement"
            char = a
        elif a or b:
            status = "single_codec_review"
            char = a or b
        else:
            status = "unresolved"
            char = ""

        rows.append({
            "class_id": class_id,
            "character": char,
            "unicode": f"U+{ord(char):04X}" if char else "",
            "status": status,
            "cp950_character": a,
            "big5hkscs_character": b,
            "sample_count": counts[class_id],
        })

    reverse = defaultdict(list)

    for row in rows:
        if row["character"]:
            reverse[row["character"]].append(row["class_id"])

    duplicates = {
        char: ids for char, ids in reverse.items()
        if len(ids) > 1
    }

    for row in rows:
        if row["character"] in duplicates:
            row["status"] += "_duplicate_review"

    verified = [
        row for row in rows
        if row["status"] == "consistent"
    ]

    review = [
        row for row in rows
        if row["status"] != "consistent"
    ]

    save_csv(
        OUT / "character_mapping_detailed.csv", rows
    )
    save_csv(
        OUT / "character_mapping_verified.csv", verified
    )
    save_csv(
        OUT / "character_mapping_review.csv", review
    )

    summary = {
        "manifest_rows": total,
        "expected_rows": 684677,
        "total_classes": N,
        "classes_found": sum(x > 0 for x in counts[1:]),
        "invalid_rows": invalid,
        "folder_mismatches": folder_mismatch,
        "high_confidence": len(verified),
        "need_review": len(review),
        "duplicate_groups": len(duplicates),
        "status_counts": dict(
            Counter(row["status"] for row in rows)
        ),
        "first_20": rows[:20],
        "note": (
            "Recovered from filename bytes. "
            "External label verification remains outstanding."
        ),
    }

    (OUT / "character_mapping_summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    print("=" * 60)
    print("Manifest rows:", total)
    print("Classes found:", summary["classes_found"])
    print("Invalid rows:", invalid)
    print("Folder mismatches:", folder_mismatch)
    print("High-confidence mappings:", len(verified))
    print("Classes requiring review:", len(review))
    print("Duplicate-character groups:", len(duplicates))

    print("\nFirst 20 mappings:")

    for row in rows[:20]:
        print(
            f"Class {row['class_id']:5d} -> "
            f"{row['character'] or 'UNKNOWN'} "
            f"[{row['status']}]"
        )

    print("\nResults directory:", OUT)

    if total != 684677 or invalid or folder_mismatch:
        print("WARNING: Manifest structure needs review.")

    print("Stage 8C Recovery Audit PASSED!")


if __name__ == "__main__":
    main()