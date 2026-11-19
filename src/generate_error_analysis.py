from pathlib import Path
import csv
import json

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "stage8"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def resolve_image(raw_path):
    path = Path(raw_path)

    if path.is_absolute():
        return path

    return ROOT / path


def read_image(path):
    if not path.is_file():
        return None

    try:
        data = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        return cv2.imdecode(
            data,
            cv2.IMREAD_GRAYSCALE
        )
    except (OSError, ValueError, cv2.error):
        return None


def build_error_montage(errors):
    print("Generating error image montage...")

    selected = []
    used_classes = set()
    used_paths = set()

    # Prefer diverse true classes.
    for item in errors:
        class_id = item["true_class_id"]

        if class_id not in used_classes:
            selected.append(item)
            used_classes.add(class_id)
            used_paths.add(item["image_path"])

        if len(selected) == 24:
            break

    # Fill remaining positions if needed.
    if len(selected) < 24:
        for item in errors:
            path = item["image_path"]

            if path not in used_paths:
                selected.append(item)
                used_paths.add(path)

            if len(selected) == 24:
                break

    fig, axes = plt.subplots(
        4, 6, figsize=(18, 13)
    )

    missing = 0

    for ax, item in zip(axes.flat, selected):
        path = resolve_image(
            item["image_path"]
        )

        image = read_image(path)

        if image is None:
            missing += 1
            ax.text(
                0.5, 0.5,
                "Image unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
        else:
            ax.imshow(
                image,
                cmap="gray"
            )

        ax.set_title(
            f"True: {item['true_class_id']}\n"
            f"Pred: {item['predicted_class_id']}",
            fontsize=10
        )

        ax.axis("off")

    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        "CNN V2 - Misclassified Handwriting Samples\n"
        "Numeric Class IDs (Character Mapping Unverified)",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output = OUTPUT / "error_analysis_montage.png"

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Error montage:", output)
    print("Displayed images:", len(selected))
    print("Unavailable images:", missing)

    return len(selected), missing


def audit_character_mapping():
    print("")
    print("Checking character mapping...")

    manifest = (
        ROOT / "data" / "extracted" / "manifest.csv"
    )

    candidates = [
        ROOT / "data" / "class_mapping.csv",
        ROOT / "data" / "labels.csv",
        ROOT / "data" / "class_to_char.csv",
        ROOT / "data" / "extracted" / "class_mapping.csv",
        ROOT / "data" / "extracted" / "labels.csv",
        ROOT / "data" / "raw" / "labels.txt",
    ]

    lines = [
        "DIT5411 - Character Mapping Audit",
        "=" * 60,
        "",
        "This is an inspection, not mapping verification.",
        "",
    ]

    found = []

    for candidate in candidates:
        if candidate.is_file():
            found.append(str(candidate))

    lines.append("Potential mapping files:")

    if found:
        lines.extend(found)
    else:
        lines.append(
            "No candidate file found at the checked paths."
        )

    lines.append("")

    if manifest.is_file():
        lines.append(f"Manifest: {manifest}")

        with manifest.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            columns = reader.fieldnames or []

            lines.append("")
            lines.append(
                "Manifest columns: " + repr(columns)
            )

            lines.append("")
            lines.append("First five manifest rows:")

            for index, row in enumerate(reader):
                if index >= 5:
                    break

                short_row = {
                    key: str(value)[:160]
                    for key, value in row.items()
                }

                lines.append(
                    json.dumps(
                        short_row,
                        ensure_ascii=False
                    )
                )
    else:
        lines.append(
            "Manifest file was not found."
        )

    lines.extend([
        "",
        "IMPORTANT:",
        "A numeric class ID is not a verified Chinese",
        "character until an authoritative mapping",
        "has been recovered and checked.",
        "",
        "Original filenames may contain encoded data;",
        "do not assume these are actual character labels.",
    ])

    output = OUTPUT / "character_mapping_audit.txt"

    output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("Mapping audit:", output)

    for line in lines:
        print(line)

    return output


def main():
    print(
        "Stage 8B - Error Analysis",
        flush=True
    )
    print("=" * 60)

    metrics = load_json(
        RESULTS / "full_13065_v2_test_metrics.json"
    )

    classes = load_csv(
        RESULTS / "full_13065_v2_test_per_class.csv"
    )

    errors = load_csv(
        RESULTS / "full_13065_v2_test_errors_sample.csv"
    )

    assert int(metrics["test_images"]) == 162550
    assert int(metrics["tested_classes"]) == 13022
    assert len(classes) == 13065

    tested = []

    for row in classes:
        count = int(row["test_images"])

        if count > 0:
            tested.append({
                "class_id": int(row["class_id"]),
                "total": count,
                "correct": int(row["correct"]),
                "accuracy": float(row["accuracy"])
            })

    assert len(tested) == 13022
    assert sum(x["total"] for x in tested) == 162550

    low_accuracy = sorted(
        tested,
        key=lambda x: (
            x["accuracy"],
            -x["total"],
            x["class_id"]
        )
    )

    below_30 = sum(
        x["accuracy"] < 0.30
        for x in tested
    )

    above_80 = sum(
        x["accuracy"] >= 0.80
        for x in tested
    )

    zero_correct = sum(
        x["correct"] == 0
        for x in tested
    )

    # Full-test confusion counts, not just
    # the first 200 saved error examples.
    confusions = metrics.get(
        "common_confusions", []
    )

    confusion_csv = (
        OUTPUT / "top_confusion_pairs.csv"
    )

    with confusion_csv.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "true_class_id",
            "predicted_class_id",
            "error_count"
        ])

        for item in confusions:
            writer.writerow([
                item["true_class_id"],
                item["predicted_class_id"],
                item["count"]
            ])

    displayed, missing = build_error_montage(
        errors
    )

    mapping_file = audit_character_mapping()

    top1 = 100 * float(
        metrics["top1_accuracy"]
    )

    top5 = 100 * float(
        metrics["top5_accuracy"]
    )

    macro = 100 * float(
        metrics["macro_class_accuracy"]
    )

    lines = [
        "# CNN V2 - Error Analysis",
        "",
        "## Final Test Results",
        "",
        f"- Test images: {metrics['test_images']:,}",
        f"- Tested classes: {metrics['tested_classes']:,}",
        f"- Top-1 Accuracy: {top1:.2f}%",
        f"- Top-5 Accuracy: {top5:.2f}%",
        f"- Macro Class Accuracy: {macro:.2f}%",
        "",
        "## Per-Class Performance",
        "",
        f"- Classes below 30% accuracy: {below_30}",
        f"- Classes at or above 80% accuracy: {above_80}",
        f"- Classes with zero correct predictions: {zero_correct}",
        "",
        "## Twenty Most Difficult Tested Classes",
        "",
        "| Class ID | Test Images | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]

    for item in low_accuracy[:20]:
        lines.append(
            f"| {item['class_id']} | "
            f"{item['total']} | "
            f"{item['correct']} | "
            f"{100 * item['accuracy']:.2f}% |"
        )

    lines.extend([
        "",
        "## Common Confusions",
        "",
        "| True Class | Predicted Class | Errors |",
        "|---|---|---:|",
    ])

    for item in confusions[:20]:
        lines.append(
            f"| {item['true_class_id']} | "
            f"{item['predicted_class_id']} | "
            f"{item['count']} |"
        )

    lines.extend([
        "",
        "## Error Examples",
        "",
        "![Misclassified images](error_analysis_montage.png)",
        "",
        f"Displayed examples: {displayed}",
        f"Missing image files: {missing}",
        "",
        "The saved error-example CSV contains only",
        "up to the first 200 errors encountered.",
        "The montage is illustrative, not a random",
        "or representative sample of all errors.",
        "",
        "## Interpretation",
        "",
        "The CNN achieves higher Top-5 than Top-1",
        "accuracy, suggesting that a candidate-ranking",
        "interface could be useful.",
        "",
        "Per-class accuracy varies substantially.",
        "Specific causes, such as visually similar",
        "characters, require character-label",
        "verification before they can be confirmed.",
        "",
        "## Limitations",
        "",
        "- 43 classes have no independent test examples.",
        "- Class-ID-to-character mapping is not yet verified.",
        "- Error examples are a small, biased subset.",
        "- Path-level splitting does not prove the absence",
        "  of image-content duplicates.",
        "",
        "See character_mapping_audit.txt for",
        "the original metadata inspection.",
    ])

    report = OUTPUT / "error_analysis_report.md"

    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("")
    print("=" * 60)
    print("Final Test Top-1:", f"{top1:.2f}%")
    print("Final Test Top-5:", f"{top5:.2f}%")
    print("Macro Accuracy:", f"{macro:.2f}%")

    print("Classes below 30%:", below_30)
    print("Classes at least 80%:", above_80)
    print("Classes with zero correct:", zero_correct)

    print("Error samples available:", len(errors))
    print("Error montage:", OUTPUT / "error_analysis_montage.png")
    print("Confusion CSV:", confusion_csv)
    print("Error report:", report)
    print("Mapping audit:", mapping_file)

    print("Stage 8B PASSED!")


if __name__ == "__main__":
    main()