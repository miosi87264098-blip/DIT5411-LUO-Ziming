from pathlib import Path
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
    ROOT,
    SPLITS,
    RESULTS,
    MODELS,
    CLASS_IDS,
    IMAGE_SIZE,
    read_records,
    load_eval_data,
    AugmentedTrainDataset
)


# ==========================================
# Configuration
# ==========================================

SEED = 2026
EPOCHS = 20
MODEL_NAME = "CNN Demo V1"

tf.keras.utils.set_random_seed(SEED)

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Create CNN architecture
# ==========================================

def create_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        # First convolution block
        tf.keras.layers.Conv2D(
            16,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(2),

        # Second convolution block
        tf.keras.layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(2),

        # Third convolution block
        tf.keras.layers.Conv2D(
            48,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(2),

        # Classification head
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(
            len(CLASS_IDS),
            activation="softmax"
        )

    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# 2. Save accuracy and loss charts
# ==========================================

def save_history(history):

    data = history.history

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    epochs = range(
        1,
        len(data["accuracy"]) + 1
    )

    axes[0].plot(
        epochs,
        data["accuracy"],
        label="Training"
    )

    axes[0].plot(
        epochs,
        data["val_accuracy"],
        label="Validation"
    )

    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        data["loss"],
        label="Training"
    )

    axes[1].plot(
        epochs,
        data["val_loss"],
        label="Validation"
    )

    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    output = RESULTS / "cnn_demo_v1_history.png"

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Training chart saved:", output)


# ==========================================
# 3. Save confusion matrix
# ==========================================

def save_confusion_matrix(y_true, y_pred):

    matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(CLASS_IDS)
    ).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    labels = [
        str(class_id)
        for class_id in CLASS_IDS
    ]

    ax.set_xticks(
        range(len(labels)),
        labels
    )

    ax.set_yticks(
        range(len(labels)),
        labels
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("CNN Demo Confusion Matrix")

    for row in range(len(labels)):
        for col in range(len(labels)):

            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output = RESULTS / "cnn_demo_v1_confusion.png"

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Confusion matrix saved:", output)

    return matrix


# ==========================================
# 4. Compare CNN against previous MLP
# ==========================================

def compare_with_mlp(cnn_metrics):

    mlp_path = RESULTS / "mlp_demo_v2_metrics.json"

    if not mlp_path.is_file():
        print("MLP metrics not found; comparison skipped.")
        return

    with open(
        mlp_path,
        "r",
        encoding="utf-8"
    ) as file:
        mlp_metrics = json.load(file)

    comparison = [
        {
            "model": "MLP Demo V2",
            "accuracy": float(
                mlp_metrics["test_accuracy"]
            ),
            "best_val_accuracy": float(
                mlp_metrics["best_validation_accuracy"]
            ),
            "epochs": int(
                mlp_metrics["epochs_completed"]
            )
        },
        {
            "model": "CNN Demo V1",
            "accuracy": float(
                cnn_metrics["test_accuracy"]
            ),
            "best_val_accuracy": float(
                cnn_metrics["best_validation_accuracy"]
            ),
            "epochs": int(
                cnn_metrics["epochs_completed"]
            )
        }
    ]

    # Save comparison CSV
    csv_path = (
        RESULTS / "model_comparison_3class.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "accuracy",
                "best_val_accuracy",
                "epochs"
            ]
        )

        writer.writeheader()
        writer.writerows(comparison)

    # Save comparison chart
    names = [
        item["model"]
        for item in comparison
    ]

    accuracies = [
        item["accuracy"] * 100
        for item in comparison
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        names,
        accuracies
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("MLP vs CNN - 3 Class Demo")

    for bar, accuracy in zip(bars, accuracies):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 1,
            f"{accuracy:.2f}%",
            ha="center"
        )

    fig.tight_layout()

    chart_path = (
        RESULTS / "model_comparison_3class.png"
    )

    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print("Comparison CSV saved:", csv_path)
    print("Comparison chart saved:", chart_path)

    print("\nModel Comparison:")
    for item in comparison:
        print(
            item["model"],
            f"{item['accuracy'] * 100:.2f}%"
        )


# ==========================================
# 5. Main training workflow
# ==========================================

def main():

    print("DIT5411 - CNN Training Demo", flush=True)
    print("=" * 50, flush=True)

    # Load exactly the same split as MLP V2
    train_records = read_records(
        SPLITS / "demo_train.csv"
    )

    val_records = read_records(
        SPLITS / "demo_val.csv"
    )

    test_records = read_records(
        SPLITS / "demo_test.csv"
    )

    train_counts = Counter(
        label for _, label in train_records
    )

    val_counts = Counter(
        label for _, label in val_records
    )

    test_counts = Counter(
        label for _, label in test_records
    )

    print("Train:", dict(train_counts))
    print("Validation:", dict(val_counts))
    print("Test:", dict(test_counts))

    assert all(
        train_counts[i] == 32
        for i in range(3)
    )

    assert all(
        val_counts[i] == 8
        for i in range(3)
    )

    assert all(
        test_counts[i] == 11
        for i in range(3)
    )

    train_paths = {
        str(path) for path, _ in train_records
    }

    val_paths = {
        str(path) for path, _ in val_records
    }

    test_paths = {
        str(path) for path, _ in test_records
    }

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    print("Data Leakage Check: PASSED!")

    # Reuse the working MLP data generator.
    train_data = AugmentedTrainDataset(
        train_records,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=2
    )

    assert len(train_data.entries) == 600
    assert train_data.original_count == 96
    assert train_data.augmented_count == 504

    print(
        "Training samples per epoch:",
        len(train_data.entries)
    )

    print(
        "Augmented samples per epoch:",
        train_data.augmented_count
    )

    # Evaluation sets: no augmentation
    x_val, y_val = load_eval_data(
        val_records
    )

    x_test, y_test = load_eval_data(
        test_records
    )

    print("Validation shape:", x_val.shape)
    print("Test shape:", x_test.shape)

    # Create and compile CNN
    model = create_cnn()

    model.summary()

    print("Model parameters:", model.count_params())

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )

    ]

    print("\nStarting CNN training...", flush=True)

    start_time = time.perf_counter()

    history = model.fit(
        train_data,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=False,
        verbose=2
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    # Evaluate on the separate test CSV
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    probabilities = model.predict(
        x_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    correct = int(
        np.sum(predictions == y_test)
    )

    print("=" * 50)
    print("Final CNN Test Results")
    print("Test Loss:", round(float(test_loss), 4))
    print(
        "Test Accuracy:",
        round(float(test_accuracy) * 100, 2),
        "%"
    )

    print(
        "Correct Predictions:",
        f"{correct}/{len(y_test)}"
    )

    print(
        "Training seconds:",
        round(training_seconds, 2)
    )

    # Save trained CNN
    model_path = (
        MODELS / "cnn_demo_v1.keras"
    )

    model.save(model_path)

    print("CNN model saved:", model_path)

    # Save figures
    save_history(history)

    matrix = save_confusion_matrix(
        y_test,
        predictions
    )

    # Save misclassified image records
    errors_path = (
        RESULTS / "cnn_demo_v1_errors.csv"
    )

    with open(
        errors_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_path",
            "true_class_id",
            "predicted_class_id",
            "predicted_probability"
        ])

        for index, predicted in enumerate(predictions):

            actual = int(y_test[index])

            if int(predicted) != actual:

                path, _ = test_records[index]

                writer.writerow([
                    str(path),
                    CLASS_IDS[actual],
                    CLASS_IDS[int(predicted)],
                    float(
                        probabilities[index, int(predicted)]
                    )
                ])

    # Save metrics JSON
    metrics = {
        "model": MODEL_NAME,
        "class_ids": CLASS_IDS,
        "image_size": IMAGE_SIZE,
        "train_originals": 96,
        "train_samples_per_epoch": 600,
        "augmented_samples_per_epoch": 504,
        "validation_images": 24,
        "test_images": 33,
        "model_parameters": int(
            model.count_params()
        ),
        "epochs_completed": len(
            history.history["loss"]
        ),
        "training_seconds": float(
            training_seconds
        ),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "correct_test_predictions": correct,
        "confusion_matrix": matrix.tolist(),
        "history": {
            key: [
                float(value)
                for value in values
            ]
            for key, values in history.history.items()
        }
    }

    metrics_path = (
        RESULTS / "cnn_demo_v1_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Metrics saved:", metrics_path)
    print("Error records saved:", errors_path)

    # Compare with completed MLP experiment
    compare_with_mlp(metrics)

    print("=" * 50)
    print("CNN Demo Training PASSED!")


if __name__ == "__main__":
    main()