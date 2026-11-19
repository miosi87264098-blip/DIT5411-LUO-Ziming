from pathlib import Path
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augment_image


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 13065
IMAGE_SIZE = 96
BATCH_SIZE = 32
SEED = 2026

STATE_FILE = RESULTS / "full_13065_training_state.json"

CONFIG = {
    "architecture": "layernorm_cnn_13065_v1",
    "classes": NUM_CLASSES,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "training_manifest": "full_train.csv",
    "validation_manifest": "full_val.csv",
}

tf.keras.utils.set_random_seed(SEED)


def load_groups(filename, expected_total):
    groups = [[] for _ in range(NUM_CLASSES)]
    csv_path = SPLITS / filename

    total = 0

    with csv_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            label = int(row["label_index"])
            class_id = int(row["class_id"])

            if not 0 <= label < NUM_CLASSES:
                raise ValueError("Invalid label")

            if label != class_id - 1:
                raise ValueError("Incorrect class mapping")

            path = str(ROOT / row["image_path"])
            groups[label].append(path)
            total += 1

    if total != expected_total:
        raise ValueError(
            f"{filename}: expected {expected_total}, "
            f"found {total}"
        )

    missing = sum(not group for group in groups)

    if missing:
        raise ValueError(
            f"{missing} classes have no images"
        )

    print(
        f"{filename}: {total} images, "
        f"{NUM_CLASSES} classes"
    )

    return groups


def python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def load_image(path, label, augment_flag, seed):
    path = python_scalar(path)

    if isinstance(path, bytes):
        path = path.decode("utf-8")

    label = int(python_scalar(label))
    augment_flag = bool(python_scalar(augment_flag))
    seed = int(python_scalar(seed))

    image = cv2.imread(
        str(path), cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(path)

    if augment_flag:
        image = augment_image(image, seed=seed)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = 1.0 - (
        image.astype(np.float32) / 255.0
    )

    return image[..., np.newaxis], np.int32(label)


def tensorflow_process(path, label, flag, seed):
    image, target = tf.numpy_function(
        load_image,
        [path, label, flag, seed],
        [tf.float32, tf.int32]
    )

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    target.set_shape(())

    return image, target


SIGNATURE = (
    tf.TensorSpec((), tf.string),
    tf.TensorSpec((), tf.int32),
    tf.TensorSpec((), tf.bool),
    tf.TensorSpec((), tf.int64),
)


def train_generator(groups, samples_per_class, epoch):
    total = NUM_CLASSES * samples_per_class

    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(total)

    for value in order:
        slot, label = divmod(int(value), NUM_CLASSES)
        originals = groups[label]

        if samples_per_class == 8:
            original_limit = min(4, len(originals))
        else:
            original_limit = len(originals)

        if slot < original_limit:
            path = originals[slot]
            needs_augmentation = False
        else:
            source_index = (
                slot - original_limit
            ) % len(originals)

            path = originals[source_index]
            needs_augmentation = True

        seed = (
            SEED
            + epoch * 10000000
            + label * 200
            + slot
        )

        yield (
            path,
            np.int32(label),
            np.bool_(needs_augmentation),
            np.int64(seed),
        )


def validation_generator(groups, phase):
    for label, originals in enumerate(groups):

        if phase == "coverage":
            chosen = originals[:1]
        else:
            chosen = originals

        for path in chosen:
            yield (
                path,
                np.int32(label),
                np.bool_(False),
                np.int64(SEED),
            )


def build_dataset(generator_fn, count):
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=SIGNATURE,
    )

    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(count)
    )

    dataset = dataset.map(
        tensorflow_process,
        num_parallel_calls=2,
    )

    dataset = dataset.batch(
        BATCH_SIZE, drop_remainder=False
    )

    dataset = dataset.prefetch(1)

    return dataset


def build_model():
    layers = tf.keras.layers

    model = tf.keras.Sequential([
        layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        layers.Conv2D(
            16, 3, padding="same", use_bias=False
        ),
        layers.LayerNormalization(axis=-1),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(
            32, 3, padding="same", use_bias=False
        ),
        layers.LayerNormalization(axis=-1),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(
            64, 3, padding="same", use_bias=False
        ),
        layers.LayerNormalization(axis=-1),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(
            96, 3, padding="same", use_bias=False
        ),
        layers.LayerNormalization(axis=-1),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(NUM_CLASSES)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def save_json_atomic(path, data):
    temporary = path.with_suffix(".tmp")

    with temporary.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2)

    os.replace(temporary, path)


def read_or_create_state():
    if STATE_FILE.is_file():
        with STATE_FILE.open(
            encoding="utf-8"
        ) as file:
            state = json.load(file)

        if state["config"] != CONFIG:
            raise ValueError(
                "Checkpoint configuration mismatch"
            )

        checkpoint = state.get("checkpoint")

        if not checkpoint:
            raise ValueError(
                "Saved state has no checkpoint"
            )

        checkpoint_path = MODELS / checkpoint

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        model = tf.keras.models.load_model(
            checkpoint_path
        )

        print("Checkpoint restored:", checkpoint_path)

    else:
        state = {
            "config": CONFIG,
            "completed": {
                "coverage": 0,
                "full": 0,
            },
            "checkpoint": None,
            "history": [],
        }

        model = build_model()
        print("Created new 13065-class model")

    return model, state


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=["coverage", "full"],
        default="coverage",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Total epochs for the selected phase",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs must be positive")

    phase = args.phase

    print(
        "Stage 6D-2 - Full Class Training",
        flush=True
    )
    print("=" * 55)
    print("Phase:", phase)

    train_groups = load_groups(
        "full_train.csv", 417719
    )

    val_groups = load_groups(
        "full_val.csv", 104408
    )

    model, state = read_or_create_state()

    if phase == "full":
        if state["completed"]["coverage"] < 1:
            raise ValueError(
                "Run coverage phase before full training"
            )

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError("Invalid model output size")

    samples_per_class = (
        8 if phase == "coverage" else 200
    )

    train_total = NUM_CLASSES * samples_per_class

    val_total = (
        NUM_CLASSES
        if phase == "coverage"
        else 104408
    )

    print("Model parameters:", model.count_params())
    print("Output classes:", NUM_CLASSES)
    print("Training samples per epoch:", train_total)
    print("Validation images:", val_total)
    print("Batch size:", BATCH_SIZE)

    completed = state["completed"][phase]

    if completed >= args.epochs:
        print("Requested phase already completed.")
        print("Stage 6D-2 PASSED!")
        return

    validation = build_dataset(
        lambda: validation_generator(
            val_groups, phase
        ),
        val_total,
    )

    for epoch in range(completed, args.epochs):
        print("=" * 55)
        print(
            f"Phase {phase}: Epoch "
            f"{epoch + 1}/{args.epochs}",
            flush=True
        )

        global_epoch = len(state["history"]) + 1

        training = build_dataset(
            lambda e=global_epoch: train_generator(
                train_groups,
                samples_per_class,
                e,
            ),
            train_total,
        )

        start = time.perf_counter()

        history = model.fit(
            training,
            validation_data=validation,
            epochs=1,
            shuffle=False,
            verbose=2,
        )

        elapsed = time.perf_counter() - start

        result = {
            "phase": phase,
            "epoch": epoch + 1,
            "training_samples": train_total,
            "validation_samples": val_total,
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
            ),
        }

        if not np.isfinite(result["loss"]):
            raise ValueError("Training loss is not finite")

        checkpoint_name = (
            f"full_13065_{phase}_"
            f"epoch_{epoch + 1:03d}.keras"
        )

        checkpoint_path = MODELS / checkpoint_name

        temporary_model = MODELS / (
            f"full_13065_{phase}_"
            f"epoch_{epoch + 1:03d}.tmp.keras"
        )

        # Save complete model and optimizer state.
        model.save(temporary_model)
        os.replace(temporary_model, checkpoint_path)

        state["history"].append(result)
        state["completed"][phase] = epoch + 1
        state["checkpoint"] = checkpoint_name

        save_json_atomic(STATE_FILE, state)

        print(
            "Checkpoint saved:",
            checkpoint_path
        )
        print(
            "Epoch seconds:",
            round(elapsed, 2)
        )

    print("=" * 55)
    print("Completed phase:", phase)
    print("Completed epochs:", state["completed"][phase])
    print("State file:", STATE_FILE)
    print("This is not final test accuracy.")
    print("Stage 6D-2 PASSED!")


if __name__ == "__main__":
    main()