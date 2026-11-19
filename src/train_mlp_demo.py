from pathlib import Path
from collections import Counter
import csv
import json
import math

import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augmentation import augment_image


# ==========================================
# Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

CLASS_IDS = [1, 1000, 13065]

IMAGE_SIZE = 96
BATCH_SIZE = 16
TARGET_PER_CLASS = 200

EPOCHS = 20
SEED = 2026

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# Read CSV records
# ==========================================

def read_records(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    records = []

    with open(
        csv_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            class_id = int(row["class_id"])
            label = int(row["label_index"])

            if label not in range(len(CLASS_IDS)):
                raise ValueError("Invalid label index")

            if CLASS_IDS[label] != class_id:
                raise ValueError("Class ID mismatch")

            path = ROOT / row["image_path"]

            if not path.is_file():
                raise FileNotFoundError(path)

            records.append((path, label))

    return records


# ==========================================
# OpenCV image preprocessing
# ==========================================

def load_image(path, augment=False, seed=0):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(path)

    if augment:
        image = augment_image(
            image,
            seed=seed
        )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Dark strokes become high pixel values.
    image = 1.0 - (
        image.astype(np.float32) / 255.0
    )

    return image[..., np.newaxis]


# ==========================================
# Training data with augmentation
# ==========================================

class AugmentedTrainDataset(tf.keras.utils.PyDataset):

    def __init__(self, records, **kwargs):

        super().__init__(**kwargs)

        grouped = {
            label: []
            for label in range(len(CLASS_IDS))
        }

        for path, label in records:
            grouped[label].append(path)

        self.entries = []

        for label, paths in grouped.items():

            if not paths:
                raise ValueError(
                    f"No training images for label {label}"
                )

            if len(paths) > TARGET_PER_CLASS:
                raise ValueError(
                    "Original count exceeds target"
                )

            # Keep every original training image.
            for path in paths:
                self.entries.append(
                    (path, label, False)
                )

            # Add augmented copies to reach 200.
            additional = TARGET_PER_CLASS - len(paths)

            for index in range(additional):

                path = paths[index % len(paths)]

                self.entries.append(
                    (path, label, True)
                )

        self.order = np.arange(len(self.entries))
        self.rng = np.random.default_rng(SEED)
        self.epoch_number = 0

        self.rng.shuffle(self.order)

        self.original_count = len(records)
        self.augmented_count = (
            len(self.entries) - len(records)
        )

    def __len__(self):

        return math.ceil(
            len(self.entries) / BATCH_SIZE
        )

    def __getitem__(self, batch_index):

        start = batch_index * BATCH_SIZE
        end = min(
            start + BATCH_SIZE,
            len(self.entries)
        )

        selected = self.order[start:end]

        images = np.empty(
            (len(selected), IMAGE_SIZE, IMAGE_SIZE, 1),
            dtype=np.float32
        )

        labels = np.empty(
            len(selected),
            dtype=np.int32
        )

        for position, entry_index in enumerate(selected):

            entry_index = int(entry_index)

            path, label, augment = (
                self.entries[entry_index]
            )

            random_seed = (
                SEED
                + self.epoch_number * 1000000
                + entry_index
            )

            images[position] = load_image(
                path,
                augment=augment,
                seed=random_seed
            )

            labels[position] = label

        return images, labels

    def on_epoch_end(self):

        self.epoch_number += 1
        self.rng.shuffle(self.order)


# ==========================================
# Validation and test preprocessing
# ==========================================

def load_eval_data(records):

    images = np.stack([
        load_image(path)
        for path, label in records
    ])

    labels = np.array(
        [label for path, label in records],
        dtype=np.int32
    )

    return images, labels


# ==========================================
# Build MLP network
# ==========================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

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
# Save training charts
# ==========================================

def save_history_chart(history):

    data = history.history

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    axes[0].plot(
        data["accuracy"],
        label="Train"
    )

    axes[0].plot(
        data["val_accuracy"],
        label="Validation"
    )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        data["loss"],
        label="Train"
    )

    axes[1].plot(
        data["val_loss"],
        label="Validation"
    )

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    output = RESULTS / "mlp_demo_v2_history.png"

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("History chart saved:", output)


# ==========================================
# Save confusion matrix
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

    names = [
        str(class_id)
        for class_id in CLASS_IDS
    ]

    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("MLP Demo Confusion Matrix")

    for row in range(len(names)):
        for col in range(len(names)):

            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output = RESULTS / "mlp_demo_v2_confusion.png"

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Confusion matrix saved:", output)

    return matrix


# ==========================================
# Main
# ==========================================

def main():

    print("DIT5411 - Improved MLP Demo", flush=True)
    print("=" * 50, flush=True)

    train_records = read_records(
        SPLITS / "demo_train.csv"
    )

    val_records = read_records(
        SPLITS / "demo_val.csv"
    )

    test_records = read_records(
        SPLITS / "demo_test.csv"
    )

    # Check the expected demo dataset.
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
        "Original training images:",
        train_data.original_count
    )

    print(
        "Augmented samples per epoch:",
        train_data.augmented_count
    )

    print(
        "Total training samples per epoch:",
        len(train_data.entries)
    )

    x_val, y_val = load_eval_data(
        val_records
    )

    x_test, y_test = load_eval_data(
        test_records
    )

    print("Validation shape:", x_val.shape)
    print("Test shape:", x_test.shape)

    model = create_model()
    model.summary()

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

    print("\nStarting improved MLP training...", flush=True)

    history = model.fit(
        train_data,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=False,
        verbose=2
    )

    # Final evaluation on the separate test CSV.
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(
        x_test,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    correct = int(
        np.sum(predicted_labels == y_test)
    )

    print("=" * 50)
    print("Final Independent Test Results")
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

    # Save trained model.
    model_path = MODELS / "mlp_demo_v2.keras"
    model.save(model_path)

    print("Model saved:", model_path)

    # Save figures.
    save_history_chart(history)

    matrix = save_confusion_matrix(
        y_test,
        predicted_labels
    )

    # Save experiment metrics.
    metrics = {
        "model": "MLP Demo V2",
        "class_ids": CLASS_IDS,
        "original_training_images": 96,
        "augmented_samples_per_epoch": 504,
        "training_samples_per_epoch": 600,
        "validation_images": 24,
        "test_images": 33,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "correct_test_predictions": correct,
        "confusion_matrix": matrix.tolist(),
        "training_history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
    }

    output_json = RESULTS / "mlp_demo_v2_metrics.json"

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Metrics saved:", output_json)

    print("=" * 50)
    print("Improved MLP Demo PASSED!")


if __name__ == "__main__":
    main()