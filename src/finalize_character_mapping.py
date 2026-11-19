from pathlib import Path
from collections import Counter
import csv
import json
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (
    ROOT / "results" / "stage8" / "mapping_v3"
    / "character_mapping_all.csv"
)
MANIFEST = ROOT / "data" / "extracted" / "manifest.csv"
OUT = ROOT / "results" / "stage9"
OUT.mkdir(parents=True, exist_ok=True)

N = 13065

# These are visually supported candidates.
# They are NOT confirmed original source labels.
SPECIAL = {
    2781: {
        "character": "\u6fc8",
        "expected_bytes": "e9ef75",
        "expected_count": 51,
    },
    12332: {
        "character": "\u5d92",
        "expected_bytes": "70e565",
        "expected_count": 51,
    },
}


def load_rows(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def check_special_raw_bytes():
    seen = {
        class_id: Counter()
        for class_id in SPECIAL
    }

    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            class_id = int(row["class_id"])

            if class_id not in SPECIAL:
                continue

            raw = bytes.fromhex(
                row["original_filename_hex"]
            )

            prefix = (
                b"cleaned_data/"
                + str(class_id).encode("ascii")
                + b"/"
            )

            if not raw.startswith(prefix):
                raise ValueError(
                    f"Unexpected original path for {class_id}"
                )

            suffix = raw[len(prefix):]

            match = re.fullmatch(
                rb"(.+)_([0-9]+)\.png",
                suffix,
                flags=re.IGNORECASE | re.DOTALL
            )

            if match is None:
                raise ValueError(
                    f"Unexpected sample name for {class_id}"
                )

            seen[class_id][match.group(1).hex()] += 1

    for class_id, spec in SPECIAL.items():
        expected = {
            spec["expected_bytes"]:
            spec["expected_count"]
        }

        if dict(seen[class_id]) != expected:
            raise ValueError(
                f"Source bytes changed for {class_id}: "
                f"{dict(seen[class_id])}"
            )

    print("Special-case source-byte audit PASSED!")


def main():
    print("Stage 9A - Final Mapping Consolidation")
    print("=" * 60)

    original = load_rows(SOURCE)

    if len(original) != N:
        raise ValueError("Expected 13065 mapping rows")

    by_id = {}

    for row in original:
        class_id = int(row["class_id"])

        if class_id in by_id:
            raise ValueError(
                f"Duplicate class ID: {class_id}"
            )

        by_id[class_id] = row

    if set(by_id) != set(range(1, N + 1)):
        raise ValueError(
            "Missing or out-of-range class IDs"
        )

    unresolved = {
        class_id
        for class_id, row in by_id.items()
        if row["status"] == "unresolved"
    }

    if unresolved != set(SPECIAL):
        raise ValueError(
            f"Unexpected unresolved classes: {unresolved}"
        )

    check_special_raw_bytes()

    complete = []

    for class_id in range(1, N + 1):
        row = by_id[class_id]
        original_status = row["status"]
        char = row["character"]

        if original_status == "consistent_direct":
            status = "direct_consistent"
            evidence = "Original filename: two-codec agreement"

        elif original_status == "consistent_reconstructed_review":
            status = "reconstructed_candidate"
            evidence = (
                "Filename-byte repair; "
                "two-codec agreement; source review pending"
            )

        elif class_id in SPECIAL:
            char = SPECIAL[class_id]["character"]
            status = "visual_supported_candidate"
            evidence = (
                "Original Big5 glyph bytes and "
                "handwriting appearance agree; "
                "extra ASCII byte remains unexplained"
            )

        else:
            raise ValueError(
                f"Unexpected status: {class_id}, "
                f"{original_status}"
            )

        if len(char) != 1 or not char.isprintable():
            raise ValueError(
                f"Invalid character for class {class_id}"
            )

        complete.append({
            "class_id": class_id,
            "character": char,
            "unicode": f"U+{ord(char):04X}",
            "status": status,
            "sample_count": int(row["sample_count"]),
            "evidence": evidence,
        })

    characters = [row["character"] for row in complete]

    if len(set(characters)) != N:
        duplicates = [
            char for char, count
            in Counter(characters).items()
            if count > 1
        ]
        raise ValueError(
            f"Duplicate characters: {duplicates[:10]}"
        )

    fields = [
        "class_id",
        "character",
        "unicode",
        "status",
        "sample_count",
        "evidence",
    ]

    csv_path = OUT / "character_mapping_for_demo.csv"

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(complete)

    mapping = {
        str(row["class_id"]): row["character"]
        for row in complete
    }

    json_path = OUT / "class_id_to_character.json"

    json_path.write_text(
        json.dumps(
            mapping, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    statuses = dict(
        Counter(row["status"] for row in complete)
    )

    summary = {
        "classes": N,
        "mapped_candidates": len(complete),
        "unique_characters": len(set(characters)),
        "status_counts": statuses,
        "visually_supported_special_cases": [
            {
                "class_id": class_id,
                "character": SPECIAL[class_id]["character"],
                "source_label_fully_verified": False,
            }
            for class_id in SPECIAL
        ],
        "limitation": (
            "This mapping contains inferred candidates. "
            "It is not fully validated against an "
            "authoritative original label manifest."
        ),
    }

    summary_path = OUT / "mapping_quality_summary.json"
    summary_path.write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    report = [
        "# Traditional Chinese Character Mapping",
        "",
        "## Coverage",
        "",
        "- CNN output classes: 13,065",
        "- Mapped character candidates: 13,065",
        "- Unique character candidates: 13,065",
        "",
        "## Evidence Levels",
        "",
        f"- Direct consistent: "
        f"{statuses.get('direct_consistent', 0)}",
        f"- Reconstructed candidates: "
        f"{statuses.get('reconstructed_candidate', 0)}",
        f"- Visually supported candidates: "
        f"{statuses.get('visual_supported_candidate', 0)}",
        "",
        "## Two Special Cases",
        "",
        "- Class 2781: U+6FC8, Big5 E9EF; "
        "the original filename additionally contains ASCII 'u'.",
        "- Class 12332: U+5D92, Big5 E565; "
        "the original filename additionally contains ASCII 'p'.",
        "",
        "The handwriting and encoded glyph bytes support",
        "these candidates, but the extra bytes have not",
        "been explained by the source dataset author.",
        "",
        "## Important Limitations",
        "",
        "- This is a reconstructed mapping, not a fully",
        "  authoritative original label dictionary.",
        "- Model accuracy metrics remain unchanged.",
        "- Inference code must verify whether model output",
        "  indices are zero-based before using this mapping.",
    ]

    report_path = OUT / "mapping_quality_report.md"
    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("Classes:", N)
    print("Mapped candidates:", len(complete))
    print("Unique characters:", len(set(characters)))
    print("Status counts:", statuses)

    print("")
    print("Class 1:", mapping["1"])
    print("Class 2781:", mapping["2781"])
    print("Class 12332:", mapping["12332"])
    print("Class 13065:", mapping["13065"])

    print("")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Quality summary:", summary_path)
    print("Quality report:", report_path)

    print("")
    print("Stage 9A Mapping Consolidation PASSED!")


if __name__ == "__main__":
    main()