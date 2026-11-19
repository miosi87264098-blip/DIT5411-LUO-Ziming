from pathlib import Path
import csv
import json
import math
import os
import time

import numpy as np
import tensorflow as tf

from full_model_smoke_test import (
    ROOT, SPLITS, RESULTS, tf_process
)

MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 13065
PILOT_CLASSES = 128
SAMPLES_PER_CLASS = 40

BATCH_SIZE = 16
EPOCHS = 2
SEED = 2026

STATE_PATH = RESULTS / "full_pilot_state.json"

tf.keras.utils.set_random_seed(SEED)


def choose_classes():
    rng = np.random.default_rng(SEED)

    anchors = [0, 999, 13064]

    remaining = np.setdiff1d(
        np.arange(NUM_CLASSES),
        np.asarray(anchors)
    )

    additional = rng.choice(
        remaining,
        size=PILOT_CLASSES - len(anchors),
        replace=False
    )

    return sorted(
        anchors + additional.astype(int).tolist()
    )


SELECTED_CLASSES = choose_classes()

CONFIG = {
    "model_version": "full_pilot_conv_v1",
    "output_classes": NUM_CLASSES,
    "selected_classes": SELECTED_CLASSES,
    "samples_per_class": SAMPLES_PER_CLASS,
    "batch_size": BATCH_SIZE,
    "seed": SEED
}


def read_selected_records(csv_path):
    records = {
        label: [] for label in SELECTED_CLASSES
    }

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = int(row["label_index"])
            class_id = int(row["class_id"])

            if label != class_id - 1:
                raise ValueError("Incorrect label mapping")

            if label in records:
                records[label].append(
                    str(ROOT / row["image_path"])
                )

    for label, paths in records.items():
        if not paths:
            raise ValueError(
                f"No images for class {label + 1}"
            )

    return records


def make_dataset(records, epoch=None):
    paths = []
    labels = []
    flags = []
    seeds = []

    training = epoch is not None

    for label in SELECTED_CLASSES:
        originals = records[label]

        if training:
            total = SAMPLES_PER_CLASS
        else:
            total = min(8, len(originals))

        for slot in range(total):
            index = slot % len(originals)
            path = originals[index]

            should_augment = (
                training and slot >= len(originals)
            )

            seed = (
                SEED
                + (epoch or 0) * 20000000
                + label * 1000
                + slot
            )

            paths.append(path)
            labels.append(label)
            flags.append(should_augment)
            seeds.append(seed)

    if training:
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(len(paths))
    else:
        order = np.arange(len(paths))

    ordered_paths = [paths[i] for i in order]

    dataset = tf.data.Dataset.from_tensor_slices((
        ordered_paths,
        np.asarray(labels, np.int32)[order],
        np.asarray(flags, np.bool_)[order],
        np.asarray(seeds, np.int64)[order]
    ))

    dataset = dataset.map(
        tf_process,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset, len(paths)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(96, 96, 1)
        ),

        tf.keras.layers.Conv2D(
            16, 3, strides=2,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            32, 3, strides=2,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.Conv2D(
            48, 3, strides=2,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64, activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(NUM_CLASSES)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),

        metrics=["sparse_categorical_accuracy"]
    )

    return model


def save_state(state):
    temporary = STATE_PATH.with_suffix(".json.tmp")

    with temporary.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )

    os.replace(temporary, STATE_PATH)


def main():
    print("Stage 6D-1 - Pilot Training", flush=True)
    print("=" * 55)

    train_records = read_selected_records(
        SPLITS / "full_train.csv"
    )

    val_records = read_selected_records(
        SPLITS / "full_val.csv"
    )

    train_paths = {
        p for group in train_records.values() for p in group
    }

    val_paths = {
        p for group in val_records.values() for p in group
    }

    if not train_paths.isdisjoint(val_paths):
        raise ValueError("Train/validation leakage")

    validation, validation_count = make_dataset(
        val_records,
        epoch=None
    )

    total_per_epoch = (
        PILOT_CLASSES * SAMPLES_PER_CLASS
    )

    print("Output classes:", NUM_CLASSES)
    print("Pilot classes:", PILOT_CLASSES)
    print("Training images per epoch:", total_per_epoch)
    print("Validation images:", validation_count)
    print("Batch size:", BATCH_SIZE)

    if STATE_PATH.is_file():
        with STATE_PATH.open(
            encoding="utf-8"
        ) as file:
            state = json.load(file)

        if state["config"] != CONFIG:
            raise ValueError(
                "Saved pilot configuration differs."
            )

        completed = state["completed_epochs"]

        checkpoint = MODELS / state["checkpoint"]

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        model = tf.keras.models.load_model(checkpoint)

        print(
            "Resuming from completed epoch:",
            completed
        )
    else:
        model = create_model()

        completed = 0

        state = {
            "config": CONFIG,
            "completed_epochs": 0,
            "checkpoint": None,
            "history": []
        }

    assert model.output_shape[-1] == NUM_CLASSES

    print("Model parameters:", model.count_params())
    print("Output Layer Check: PASSED!")

    if completed >= EPOCHS:
        print("Pilot already completed.")
        print("Stage 6D-1 PASSED!")
        return

    for epoch in range(completed, EPOCHS):
        print("=" * 55)
        print(
            f"Starting Epoch {epoch + 1}/{EPOCHS}",
            flush=True
        )

        train_data, count = make_dataset(
            train_records,
            epoch=epoch
        )

        assert count == total_per_epoch

        start = time.perf_counter()

        history = model.fit(
            train_data,
            validation_data=validation,
            initial_epoch=epoch,
            epochs=epoch + 1,
            shuffle=False,
            verbose=2
        )

        elapsed = time.perf_counter() - start

        result = {
            "epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "loss": float(
                history.history["loss"][-1]
            ),
            "accuracy": float(
                history.history[
                    "sparse_categorical_accuracy"
                ][-1]
            ),
            "val_loss": float(
                history.history["val_loss"][-1]
            ),
            "val_accuracy": float(
                history.history[
                    "val_sparse_categorical_accuracy"
                ][-1]
            )
        }

        checkpoint_name = (
            f"full_pilot_epoch_{epoch + 1:03d}.keras"
        )

        checkpoint_path = MODELS / checkpoint_name

        # Includes model and optimizer state.
        model.save(checkpoint_path)

        state["history"].append(result)
        state["completed_epochs"] = epoch + 1
        state["checkpoint"] = checkpoint_name

        save_state(state)

        print(
            "Checkpoint saved:",
            checkpoint_path
        )

        print(
            "Epoch seconds:",
            round(elapsed, 2)
        )

    print("=" * 55)
    print("Completed epochs:", state["completed_epochs"])
    print("Training samples per epoch:", total_per_epoch)
    print("Pilot state:", STATE_PATH)
    print("This is NOT final test accuracy.")
    print("Stage 6D-1 PASSED!")


if __name__ == "__main__":
    main()