from pathlib import Path
from collections import Counter
import json
import math

import numpy as np
import tensorflow as tf

import train_full_13065_v2 as p


SAMPLE_COUNT = 256
SEED = 2026

OUTPUT = (
    p.RESULTS / "full_13065_v2_diagnostics.json"
)


def prepare_images(groups, labels):
    images = []
    targets = []

    for label in labels:
        label = int(label)
        path = groups[label][0]

        image, actual_label = p.load_image(
            path, label, False, SEED
        )

        if image.shape != (96, 96, 1):
            raise ValueError("Unexpected image shape")

        if not np.isfinite(image).all():
            raise ValueError("Non-finite pixel values")

        images.append(image)
        targets.append(actual_label)

    return (
        np.stack(images).astype(np.float32),
        np.asarray(targets, dtype=np.int32)
    )


def examine(name, images, targets, model, feature_model):
    print("")
    print(f"Evaluating {name}...", flush=True)

    logits = model.predict(
        images,
        batch_size=32,
        verbose=0
    )

    if not np.isfinite(logits).all():
        raise ValueError("Non-finite model outputs")

    probabilities = tf.nn.softmax(
        logits, axis=-1
    ).numpy()

    predictions = np.argmax(
        probabilities, axis=1
    )

    top5 = np.argpartition(
        probabilities, -5, axis=1
    )[:, -5:]

    features = feature_model.predict(
        images,
        batch_size=32,
        verbose=0
    )

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits
        )
    ).numpy()

    top1_hits = int(
        np.sum(predictions == targets)
    )

    top5_hits = int(
        np.sum(
            np.any(
                top5 == targets[:, None],
                axis=1
            )
        )
    )

    top1_confidence = np.max(
        probabilities, axis=1
    )

    entropy = -np.sum(
        probabilities
        * np.log(np.maximum(probabilities, 1e-12)),
        axis=1
    )

    most_common = [
        {
            "class_id": int(label) + 1,
            "prediction_count": int(count)
        }
        for label, count in Counter(
            predictions.tolist()
        ).most_common(5)
    ]

    result = {
        "sample_count": len(targets),
        "loss": float(loss),
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
        "unique_predicted_classes": int(
            np.unique(predictions).size
        ),
        "mean_top1_confidence": float(
            np.mean(top1_confidence)
        ),
        "mean_prediction_entropy": float(
            np.mean(entropy)
        ),
        "mean_image_pixel_std": float(
            np.std(images, axis=(1, 2, 3)).mean()
        ),
        "mean_feature_std_across_images": float(
            np.std(features, axis=0).mean()
        ),
        "most_common_predictions": most_common
    }

    print("Images:", result["sample_count"])
    print("Loss:", round(result["loss"], 4))
    print("Top-1 hits:", top1_hits)
    print("Top-5 hits:", top5_hits)
    print(
        "Unique predicted classes:",
        result["unique_predicted_classes"]
    )
    print(
        "Mean top-1 confidence:",
        round(result["mean_top1_confidence"], 6)
    )
    print(
        "Mean prediction entropy:",
        round(result["mean_prediction_entropy"], 4)
    )
    print(
        "Mean image pixel std:",
        round(result["mean_image_pixel_std"], 4)
    )
    print(
        "Feature diversity:",
        round(
            result["mean_feature_std_across_images"],
            6
        )
    )
    print(
        "Most common predictions:",
        most_common
    )

    return result


def main():
    print(
        "Stage 6D-4 - V2 Model Diagnostics",
        flush=True
    )
    print("=" * 55)

    with p.STATE_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        state = json.load(file)

    checkpoint_name = state["checkpoint"]
    checkpoint = p.MODELS / checkpoint_name

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print("Checkpoint:", checkpoint_name)
    print("Coverage epochs:", state["completed"]["coverage"])

    model = tf.keras.models.load_model(
        checkpoint, compile=False
    )

    if model.output_shape[-1] != p.NUM_CLASSES:
        raise ValueError("Wrong output class count")

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    if len(dense_layers) < 2:
        raise ValueError("Expected hidden Dense layer")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=dense_layers[-2].output
    )

    print("Output classes:", model.output_shape[-1])
    print("Model parameters:", model.count_params())

    train_groups = p.load_groups(
        "full_train.csv", 417719
    )

    val_groups = p.load_groups(
        "full_val.csv", 104408
    )

    rng = np.random.default_rng(SEED)

    chosen_labels = rng.choice(
        p.NUM_CLASSES,
        size=SAMPLE_COUNT,
        replace=False
    )

    train_images, train_targets = prepare_images(
        train_groups, chosen_labels
    )

    val_images, val_targets = prepare_images(
        val_groups, chosen_labels
    )

    train_report = examine(
        "Training sample",
        train_images,
        train_targets,
        model,
        feature_model
    )

    val_report = examine(
        "Validation sample",
        val_images,
        val_targets,
        model,
        feature_model
    )

    random_loss = math.log(p.NUM_CLASSES)
    random_accuracy = 100 / p.NUM_CLASSES

    report = {
        "checkpoint": checkpoint_name,
        "classes": p.NUM_CLASSES,
        "random_baseline_loss": random_loss,
        "random_baseline_accuracy_percent": (
            random_accuracy
        ),
        "training": train_report,
        "validation": val_report,
        "diagnostic_only": True
    }

    with OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("")
    print("=" * 55)
    print(
        "Random baseline loss:",
        round(random_loss, 4)
    )
    print(
        "Random baseline accuracy:",
        round(random_accuracy, 6),
        "%"
    )
    print("Diagnostic report:", OUTPUT)

    if val_report["unique_predicted_classes"] <= 2:
        print(
            "WARNING: Predictions are concentrated "
            "in very few classes."
        )

    if abs(val_report["loss"] - random_loss) < 0.1:
        print(
            "WARNING: Validation loss remains "
            "near the random baseline."
        )

    print("Existing checkpoint was NOT modified.")
    print("Stage 6D-3 Diagnostics PASSED!")


if __name__ == "__main__":
    main()