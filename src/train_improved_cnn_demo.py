from collections import Counter
import csv
import json
import time

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_mlp_demo import (
    ROOT, SPLITS, RESULTS, MODELS,
    CLASS_IDS, IMAGE_SIZE,
    read_records, load_eval_data,
    AugmentedTrainDataset
)

SEED = 2026
EPOCHS = 25

tf.keras.utils.set_random_seed(SEED)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        tf.keras.layers.Conv2D(
            16, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            32, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            96, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(
            len(CLASS_IDS), activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_history(history):
    data = history.history
    epochs = range(1, len(data["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs, data["accuracy"], label="Train"
    )
    axes[0].plot(
        epochs, data["val_accuracy"], label="Validation"
    )
    axes[0].set_title("Improved CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs, data["loss"], label="Train"
    )
    axes[1].plot(
        epochs, data["val_loss"], label="Validation"
    )
    axes[1].set_title("Improved CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output = RESULTS / "cnn_improved_demo_history.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("History saved:", output)


def save_confusion(y_true, y_pred):
    matrix = np.zeros(
        (len(CLASS_IDS), len(CLASS_IDS)),
        dtype=np.int32
    )

    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")

    names = [str(value) for value in CLASS_IDS]
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Improved CNN Confusion Matrix")

    for row in range(len(names)):
        for col in range(len(names)):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output = RESULTS / "cnn_improved_demo_confusion.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("Confusion matrix saved:", output)

    return matrix


def save_comparison(new_metrics):
    sources = [
        ("MLP V2", "mlp_demo_v2_metrics.json"),
        ("CNN V1", "cnn_demo_v1_metrics.json")
    ]

    comparison = []

    for name, filename in sources:
        path = RESULTS / filename
        if path.is_file():
            with open(path, encoding="utf-8") as file:
                metrics = json.load(file)
            comparison.append((
                name, float(metrics["test_accuracy"])
            ))

    comparison.append((
        "Improved CNN",
        float(new_metrics["test_accuracy"])
    ))

    csv_path = RESULTS / "model_comparison_3class_v2.csv"

    with open(
        csv_path, "w",
        newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(comparison)

    names = [item[0] for item in comparison]
    values = [item[1] * 100 for item in comparison]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Three-Model Comparison")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    fig.tight_layout()
    output = RESULTS / "model_comparison_3class_v2.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Comparison chart saved:", output)

    for name, accuracy in comparison:
        print(f"{name}: {accuracy * 100:.2f}%")


def main():
    print("DIT5411 - Improved CNN Demo", flush=True)
    print("=" * 50)

    train_records = read_records(
        SPLITS / "demo_train.csv"
    )
    val_records = read_records(
        SPLITS / "demo_val.csv"
    )
    test_records = read_records(
        SPLITS / "demo_test.csv"
    )

    # Verify class counts.
    for name, records, expected in [
        ("Train", train_records, 32),
        ("Validation", val_records, 8),
        ("Test", test_records, 11)
    ]:
        counts = Counter(label for _, label in records)
        print(name, dict(counts))
        assert all(counts[i] == expected for i in range(3))

    # Verify split independence by image path.
    train_paths = {str(p) for p, _ in train_records}
    val_paths = {str(p) for p, _ in val_records}
    test_paths = {str(p) for p, _ in test_records}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    print("Data Leakage Check: PASSED!")

    train_data = AugmentedTrainDataset(
        train_records,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=2
    )

    assert len(train_data.entries) == 600
    assert train_data.augmented_count == 504

    x_val, y_val = load_eval_data(val_records)
    x_test, y_test = load_eval_data(test_records)

    print("Training samples per epoch:", 600)
    print("Validation images:", len(y_val))
    print("Test images:", len(y_test))

    model = create_model()
    model.summary()

    print("Model parameters:", model.count_params())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    print("\nStarting Improved CNN training...", flush=True)

    start = time.perf_counter()

    history = model.fit(
        train_data,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=False,
        verbose=2
    )

    elapsed = time.perf_counter() - start

    # Evaluate on the held-out test CSV.
    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    correct = int(np.sum(predictions == y_test))

    print("=" * 50)
    print("Final Improved CNN Test Results")
    print("Test Accuracy:", f"{test_accuracy * 100:.2f}%")
    print("Test Loss:", f"{test_loss:.4f}")
    print("Correct Predictions:", f"{correct}/33")
    print("Training seconds:", f"{elapsed:.2f}")

    model_path = MODELS / "cnn_improved_demo.keras"
    model.save(model_path)
    print("Model saved:", model_path)

    save_history(history)
    matrix = save_confusion(y_test, predictions)

    metrics = {
        "model": "Improved CNN Demo",
        "class_ids": CLASS_IDS,
        "image_size": IMAGE_SIZE,
        "training_samples_per_epoch": 600,
        "augmented_samples_per_epoch": 504,
        "validation_images": 24,
        "test_images": 33,
        "model_parameters": int(model.count_params()),
        "epochs_completed": len(history.history["loss"]),
        "training_seconds": float(elapsed),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "correct_test_predictions": correct,
        "confusion_matrix": matrix.tolist(),
        "history": {
            key: [float(x) for x in values]
            for key, values in history.history.items()
        }
    }

    metrics_path = RESULTS / "cnn_improved_demo_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("Metrics saved:", metrics_path)

    save_comparison(metrics)

    print("=" * 50)
    print("Improved CNN Demo PASSED!")


if __name__ == "__main__":
    main()