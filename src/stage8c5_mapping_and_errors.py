from pathlib import Path
from collections import Counter
import csv
import re

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent.parent
STAGE8 = ROOT / "results" / "stage8"
MAPDIR = STAGE8 / "mapping_v3"

MANIFEST = ROOT / "data" / "extracted" / "manifest.csv"
MAPPING = MAPDIR / "character_mapping_all.csv"
ERRORS = ROOT / "results" / (
    "full_13065_v2_test_errors_sample.csv"
)

UNRESOLVED = (2781, 12332)


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def describe_bytes(data):
    codecs = (
        "cp950",
        "big5hkscs",
        "big5",
        "gb18030",
        "utf-8",
    )

    results = []

    for codec in codecs:
        try:
            decoded = data.decode(
                codec, errors="strict"
            )

            points = " ".join(
                f"U+{ord(char):04X}"
                for char in decoded
            )

            results.append(
                f"  {codec}: {decoded!r} ({points})"
            )

        except UnicodeError:
            results.append(
                f"  {codec}: decode failed"
            )

    return results


def inspect_unresolved():
    print("Inspecting two unresolved classes...")

    candidates = {
        class_id: Counter()
        for class_id in UNRESOLVED
    }

    examples = {}
    row_counts = Counter()

    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            class_id = int(row["class_id"])

            if class_id not in candidates:
                continue

            row_counts[class_id] += 1

            raw = bytes.fromhex(
                row["original_filename_hex"]
            )

            prefix = (
                b"cleaned_data/"
                + str(class_id).encode("ascii")
                + b"/"
            )

            if not raw.startswith(prefix):
                name = raw
            else:
                suffix = raw[len(prefix):]

                match = re.fullmatch(
                    rb"(.+)_([0-9]+)\.png",
                    suffix,
                    flags=re.IGNORECASE | re.DOTALL
                )

                name = (
                    match.group(1)
                    if match else suffix
                )

            key = name.hex()
            candidates[class_id][key] += 1
            examples.setdefault(class_id, raw.hex())

    lines = [
        "DIT5411 - Remaining Character Mapping Audit",
        "=" * 65,
        "",
        "This is forensic inspection, not label verification.",
        "",
    ]

    for class_id in UNRESOLVED:
        lines.extend([
            f"Class ID: {class_id}",
            f"Manifest samples: {row_counts[class_id]}",
            f"Example original ZIP path hex:",
            examples.get(class_id, "NOT FOUND"),
            "",
            "Distinct raw character byte sequences:",
        ])

        for name_hex, count in (
            candidates[class_id].most_common(20)
        ):
            lines.append(
                f"Bytes: {name_hex} | count={count}"
            )

            lines.extend(
                describe_bytes(bytes.fromhex(name_hex))
            )
            lines.append("")

        lines.append("-" * 65)

    lines.extend([
        "",
        "IMPORTANT:",
        "Successful decoding with an alternative codec",
        "does not prove the decoded text is the true",
        "character label. Do not automatically assign it.",
    ])

    output = MAPDIR / "remaining_two_classes_audit.txt"
    output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(lines))
    print("Forensic report:", output)


def get_chinese_font():
    candidates = [
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
    ]

    for path in candidates:
        if path.is_file():
            return FontProperties(fname=str(path))

    print(
        "WARNING: Chinese font not found. "
        "Some characters may not display."
    )
    return FontProperties()


def build_chinese_error_montage(mapping):
    errors = read_csv(ERRORS)

    font = get_chinese_font()

    selected = []
    seen_classes = set()
    seen_paths = set()

    for item in errors:
        class_id = int(item["true_class_id"])

        if class_id not in seen_classes:
            selected.append(item)
            seen_classes.add(class_id)
            seen_paths.add(item["image_path"])

        if len(selected) >= 24:
            break

    if len(selected) < 24:
        for item in errors:
            if item["image_path"] not in seen_paths:
                selected.append(item)
                seen_paths.add(item["image_path"])

            if len(selected) >= 24:
                break

    def label(class_id):
        entry = mapping.get(class_id)

        if entry is None or not entry["character"]:
            return f"UNKNOWN ({class_id})"

        char = entry["character"]
        status = entry["status"]

        marker = (
            "*" if "reconstructed" in status else ""
        )

        return f"{char}{marker} ({class_id})"

    fig, axes = plt.subplots(
        4, 6, figsize=(20, 13)
    )

    missing_images = 0

    for ax, item in zip(axes.flat, selected):
        path = Path(item["image_path"])

        if not path.is_absolute():
            path = ROOT / path

        image = None

        if path.is_file():
            encoded = np.fromfile(
                str(path), dtype=np.uint8
            )

            image = cv2.imdecode(
                encoded, cv2.IMREAD_GRAYSCALE
            )

        if image is not None:
            ax.imshow(
                image, cmap="gray",
                vmin=0, vmax=255
            )
        else:
            missing_images += 1
            ax.text(
                0.5, 0.5, "Image unavailable",
                transform=ax.transAxes,
                ha="center", va="center"
            )

        true_id = int(item["true_class_id"])
        pred_id = int(item["predicted_class_id"])

        ax.set_title(
            f"True: {label(true_id)}\n"
            f"Pred: {label(pred_id)}",
            fontsize=10,
            fontproperties=font
        )

        ax.axis("off")

    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        "CNN V2 - Chinese Handwriting Error Analysis\n"
        "* = Reconstructed mapping candidate; "
        "external verification pending",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output = (
        STAGE8 / "error_analysis_chinese_labels.png"
    )

    fig.savefig(
        output, dpi=180, bbox_inches="tight"
    )

    plt.close(fig)

    print("Chinese error montage:", output)
    print("Examples displayed:", len(selected))
    print("Missing images:", missing_images)

    return output


def main():
    print("Stage 8C-5 - Final Mapping Inspection")
    print("=" * 65)

    rows = read_csv(MAPPING)

    if len(rows) != 13065:
        raise ValueError("Mapping class count is invalid")

    mapping = {
        int(row["class_id"]): row
        for row in rows
    }

    direct = sum(
        x["status"] == "consistent_direct"
        for x in rows
    )

    reconstructed = sum(
        x["status"] == "consistent_reconstructed_review"
        for x in rows
    )

    unresolved = [
        int(x["class_id"])
        for x in rows
        if x["status"] == "unresolved"
    ]

    print("Direct consistent classes:", direct)
    print("Reconstructed candidate classes:", reconstructed)
    print("Unresolved classes:", unresolved)

    inspect_unresolved()

    output = build_chinese_error_montage(mapping)

    print("=" * 65)
    print("Mapping inspection completed.")
    print("Montage:", output)
    print("Stage 8C-5 PASSED!")


if __name__ == "__main__":
    main()