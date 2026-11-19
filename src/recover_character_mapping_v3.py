from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "extracted" / "manifest.csv"
OUT = ROOT / "results" / "stage8" / "mapping_v3"

OUT.mkdir(parents=True, exist_ok=True)

N = 13065
CODECS = ("cp950", "big5hkscs")

counters = [
    [Counter(), Counter()]
    for _ in range(N + 1)
]

failures = [
    [0, 0] for _ in range(N + 1)
]

counts = [0] * (N + 1)
reconstructed = [0] * (N + 1)
examples = [""] * (N + 1)

total = 0
malformed = 0
reconstructed_total = 0


def extract_character_bytes(raw, class_id):
    prefix = (
        b"cleaned_data/"
        + str(class_id).encode("ascii")
        + b"/"
    )

    if not raw.startswith(prefix):
        return None, False

    suffix = raw[len(prefix):]

    match = re.fullmatch(
        rb"(.+)_([0-9]+)\.png",
        suffix,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return None, False

    name_bytes = match.group(1)

    # Hypothesis: a Big5 trailing 0x5C byte
    # was normalized to '/' in the ZIP path.
    #
    # Example: CA 2F -> CA 5C -> 佢
    #
    # Do not apply to arbitrary filenames.
    if (
        len(name_bytes) == 2
        and name_bytes[0] >= 0x81
        and name_bytes[1] == 0x2F
    ):
        return name_bytes[:1] + b"\\", True

    return name_bytes, False


def save_csv(path, rows):
    fields = [
        "class_id",
        "character",
        "unicode",
        "status",
        "cp950_character",
        "big5hkscs_character",
        "sample_count",
        "reconstructed_samples",
        "example_hex",
    ]

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global total, malformed, reconstructed_total

    print("Stage 8C-4 - Character Mapping V3")
    print("=" * 60, flush=True)

    with SOURCE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            total += 1

            try:
                class_id = int(row["class_id"])

                if not 1 <= class_id <= N:
                    raise ValueError("Invalid class")

                raw = bytes.fromhex(
                    row["original_filename_hex"]
                )

            except (ValueError, KeyError, TypeError):
                malformed += 1
                continue

            counts[class_id] += 1

            if not examples[class_id]:
                examples[class_id] = raw.hex()

            name_bytes, was_reconstructed = (
                extract_character_bytes(
                    raw, class_id
                )
            )

            if was_reconstructed:
                reconstructed[class_id] += 1
                reconstructed_total += 1

            if name_bytes is None:
                malformed += 1
                failures[class_id][0] += 1
                failures[class_id][1] += 1
                continue

            for index, codec in enumerate(CODECS):
                try:
                    character = name_bytes.decode(
                        codec, errors="strict"
                    )

                    if (
                        len(character) != 1
                        or not character.isprintable()
                    ):
                        raise ValueError(
                            "Not a single printable character"
                        )

                    counters[class_id][index][character] += 1

                except (UnicodeError, ValueError):
                    failures[class_id][index] += 1

            if total % 100000 == 0:
                print(
                    f"Processed {total:,} rows",
                    flush=True
                )

    rows = []

    for class_id in range(1, N + 1):
        candidates = []

        for index in range(2):
            counter = counters[class_id][index]

            if (
                counts[class_id] > 0
                and failures[class_id][index] == 0
                and len(counter) == 1
            ):
                candidates.append(next(iter(counter)))
            else:
                candidates.append("")

        cp, hk = candidates

        if cp and hk and cp == hk:
            char = cp

            if reconstructed[class_id]:
                status = "consistent_reconstructed_review"
            else:
                status = "consistent_direct"

        elif cp and hk:
            char = ""
            status = "codec_disagreement"

        elif cp or hk:
            char = cp or hk
            status = "single_codec_review"

        else:
            char = ""
            status = "unresolved"

        rows.append({
            "class_id": class_id,
            "character": char,
            "unicode": (
                f"U+{ord(char):04X}"
                if char else ""
            ),
            "status": status,
            "cp950_character": cp,
            "big5hkscs_character": hk,
            "sample_count": counts[class_id],
            "reconstructed_samples": reconstructed[class_id],
            "example_hex": examples[class_id],
        })

    reverse = defaultdict(list)

    for row in rows:
        if row["character"]:
            reverse[row["character"]].append(
                row["class_id"]
            )

    duplicate_groups = {
        char: ids
        for char, ids in reverse.items()
        if len(ids) > 1
    }

    for row in rows:
        if row["character"] in duplicate_groups:
            row["status"] += "_duplicate_review"

    direct = [
        row for row in rows
        if row["status"] == "consistent_direct"
    ]

    reconstructed_review = [
        row for row in rows
        if "reconstructed" in row["status"]
    ]

    unresolved = [
        row for row in rows
        if (
            row["status"] != "consistent_direct"
            and "reconstructed" not in row["status"]
        )
    ]

    save_csv(
        OUT / "character_mapping_all.csv",
        rows
    )

    save_csv(
        OUT / "character_mapping_direct.csv",
        direct
    )

    save_csv(
        OUT / "character_mapping_reconstructed_review.csv",
        reconstructed_review
    )

    save_csv(
        OUT / "character_mapping_unresolved.csv",
        unresolved
    )

    statuses = dict(
        Counter(row["status"] for row in rows)
    )

    summary = {
        "manifest_rows": total,
        "classes": N,
        "malformed_records": malformed,
        "reconstructed_records": reconstructed_total,
        "reconstructed_classes": sum(
            x > 0 for x in reconstructed
        ),
        "consistent_direct_classes": len(direct),
        "reconstructed_review_classes": (
            len(reconstructed_review)
        ),
        "other_review_classes": len(unresolved),
        "duplicate_character_groups": (
            len(duplicate_groups)
        ),
        "status_counts": statuses,
        "first_20": rows[:20],
        "class_104": rows[103],
        "other_review": unresolved,
        "note": (
            "0x2F to 0x5C recovery is a hypothesis "
            "requiring validation against source labels."
        )
    }

    (OUT / "mapping_v3_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("")
    print("=" * 60)
    print("Manifest records:", total)
    print("Malformed records:", malformed)
    print(
        "Reconstructed filename records:",
        reconstructed_total
    )
    print(
        "Affected reconstructed classes:",
        summary["reconstructed_classes"]
    )
    print(
        "Consistent direct mappings:",
        len(direct)
    )
    print(
        "Consistent reconstructed candidates:",
        len(reconstructed_review)
    )
    print(
        "Other classes requiring review:",
        len(unresolved)
    )
    print(
        "Duplicate character groups:",
        len(duplicate_groups)
    )
    print("Status breakdown:", statuses)

    print("")
    print("Class 104:")
    print(
        json.dumps(
            rows[103],
            ensure_ascii=False
        )
    )

    print("")
    print("Remaining unresolved / conflicting classes:")

    for row in unresolved[:30]:
        print(
            f"Class {row['class_id']}: "
            f"{row['status']} "
            f"CP950={row['cp950_character']!r} "
            f"HKSCS={row['big5hkscs_character']!r}"
        )

    print("")
    print("Output folder:", OUT)

    if total != 684677 or malformed:
        print("WARNING: Inspect manifest anomalies.")

    print("Stage 8C-4 Audit PASSED!")


if __name__ == "__main__":
    main()