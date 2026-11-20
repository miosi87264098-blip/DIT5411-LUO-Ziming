from pathlib import Path
import csv
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "stage8"

OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    print("Stage 8 - Final Results and Figures")
    print("=" * 55)

    state = read_json(
        RESULTS / "full_13065_v2_training_state.json"
    )

    test = read_json(
        RESULTS / "full_13065_v2_test_metrics.json"
    )
    split = read_json(RESULTS / "full_split_audit.json")

    history = sorted(
        [
            item for item in state["history"]
            if item["phase"] == "full"
        ],
        key=lambda item: int(item["epoch"])
    )

    if len(history) < 3:
        raise ValueError("Three Full Epochs not found")

    if test["test_images"] != 162550:
        raise ValueError("Unexpected test image count")

    epochs = [int(x["epoch"]) for x in history]

    train_acc = [
        100 * float(x["accuracy"])
        for x in history
    ]

    val_acc = [
        100 * float(x["val_accuracy"])
        for x in history
    ]

    train_loss = [
        float(x["loss"])
        for x in history
    ]

    val_loss = [
        float(x["val_loss"])
        for x in history
    ]

    # Figure 1: Training and validation accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs, train_acc,
        marker="o", label="Training Accuracy"
    )
    plt.plot(
        epochs, val_acc,
        marker="s", label="Validation Accuracy"
    )
    plt.xlabel("Full Training Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN V2 - Accuracy by Epoch")
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "full_training_accuracy.png",
        dpi=180
    )
    plt.close()

    # Figure 2: Training and validation loss
    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs, train_loss,
        marker="o", label="Training Loss"
    )
    plt.plot(
        epochs, val_loss,
        marker="s", label="Validation Loss"
    )
    plt.xlabel("Full Training Epoch")
    plt.ylabel("Loss")
    plt.title("CNN V2 - Loss by Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "full_training_loss.png",
        dpi=180
    )
    plt.close()

    # Figure 3: Class accuracy distribution
    class_accuracies = []

    with (
        RESULTS / "full_13065_v2_test_per_class.csv"
    ).open(
        encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            count = int(row["test_images"])

            if count > 0:
                class_accuracies.append(
                    100 * float(row["accuracy"])
                )

    if len(class_accuracies) != 13022:
        raise ValueError(
            "Unexpected number of tested classes"
        )

    plt.figure(figsize=(9, 5))
    plt.hist(
        class_accuracies,
        bins=np.linspace(0, 100, 21),
        edgecolor="black"
    )
    plt.xlabel("Per-Class Top-1 Accuracy (%)")
    plt.ylabel("Number of Classes")
    plt.title("CNN V2 - Test Accuracy Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "class_accuracy_distribution.png",
        dpi=180
    )
    plt.close()

    top1 = float(test["top1_accuracy"]) * 100
    top5 = float(test["top5_accuracy"]) * 100
    macro = float(test["macro_class_accuracy"]) * 100

    best_val = max(
        history,
        key=lambda x: float(x["val_accuracy"])
    )

    lines = [
        "# DIT5411 - Final Machine Learning Results",
        "",
        "## Dataset",
        "",
        "- Output classes: 13,065",
        "- Original training candidates: 522,127",
        "- Final training originals: 417,719",
        f"- Original validation images: {split['full_validation_images']:,}",
        f"- Original assignment test images: {test['test_images']:,}",
        "- Classes with test samples: 13,022",
        "- Classes without test samples: 43",
        f"- Training items per full epoch: {split['target_training_samples']:,} "
        f"({split['full_train_originals']:,} original and "
        f"{split['augmentation_required_after_validation']:,} transformed)",
        "",
        "## Model",
        "",
        "- Model: CNN V2",
        "- Parameters: 978,553",
        "- Input shape: 96 x 96 x 1",
        "- Output logits: 13,065",
        f"- Completed training: {state['completed']['coverage']} coverage epoch(s) "
        f"and {len(history)} full epochs",
        "- Final checkpoint: full_13065_v2_full_epoch_003.keras",
        "",
        "## Full Training History",
        "",
        "| Epoch | Train Acc. | Val Acc. | Train Loss | Val Loss |",
        "|---|---:|---:|---:|---:|",
    ]

    for x in history:
        lines.append(
            f"| {x['epoch']} | "
            f"{100 * float(x['accuracy']):.2f}% | "
            f"{100 * float(x['val_accuracy']):.2f}% | "
            f"{float(x['loss']):.4f} | "
            f"{float(x['val_loss']):.4f} |"
        )

    comparison_path = OUTPUT / "full_class_comparison.json"
    if comparison_path.is_file():
        v2 = next(
            model for model in read_json(comparison_path)["models"]
            if model["model"] == "CNN V2"
        )
        validation = v2.get("filtered_validation")
        if validation:
            lines.extend([
                "",
                "## Filtered Validation Follow-up",
                "",
                f"- Images: {validation['images']:,} from {validation['classes']:,} classes, "
                "after excluding exact matches with development training",
                f"- Top-1: {validation['top1_accuracy']:.2%} ({validation['top1_correct']:,} correct)",
                f"- Top-5: {validation['top5_accuracy']:.2%} ({validation['top5_correct']:,} correct)",
                f"- Macro class accuracy: {validation['macro_class_accuracy']:.2%}",
            ])

    lines.extend([
        "",
        "## Original Assignment Test Results",
        "",
        f"- Top-1 Accuracy: {top1:.2f}%",
        f"- Top-5 Accuracy: {top5:.2f}%",
        f"- Macro Class Accuracy: {macro:.2f}%",
        f"- Highest original validation accuracy: epoch {best_val['epoch']}",
        f"- Test duration: {float(test['evaluation_seconds']) / 60:.2f} minutes",
    ])
    if comparison_path.is_file():
        lines.extend([
            "",
            "See the [four-model evaluation](full_class_comparison.json) "
            "for the common test run and input hashes.",
        ])
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The 43 rare categories have no held-out test examples.",
        "- Class-ID to actual Chinese-character mapping still requires verification.",
        "- Three-class MLP/CNN experiments are not equivalent to full-class benchmarking.",
        "- The original split check applies to file paths. See the "
        "[content audit and test protocol](../../README.md#7-original-test-results) for the follow-up audit.",
        "- Test results should not be used repeatedly for hyperparameter selection.",
        "",
        "## Generated Figures",
        "",
        "![Training accuracy](full_training_accuracy.png)",
        "",
        "![Training loss](full_training_loss.png)",
        "",
        "![Class accuracy distribution](class_accuracy_distribution.png)",
    ])

    report = OUTPUT / "final_results_summary.md"

    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("Full epochs:", len(history))
    print("Top-1:", f"{top1:.2f}%")
    print("Top-5:", f"{top5:.2f}%")
    print("Macro:", f"{macro:.2f}%")
    print("Tested classes:", len(class_accuracies))
    print("")
    print("Output directory:", OUTPUT)
    print("Generated 3 PNG figures and Markdown report.")
    print("Stage 8 Figures PASSED!")


if __name__ == "__main__":
    main()
