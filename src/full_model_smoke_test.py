from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augment_image

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

CLASSES = 13065
SIZE = 96
BATCH = 16
TRAIN_COUNT = 128
VAL_COUNT = 32
SEED = 2026

tf.keras.utils.set_random_seed(SEED)


def select_images(csv_path, selected_labels):
    wanted = set(map(int, selected_labels))
    found = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = int(row["label_index"])
            class_id = int(row["class_id"])

            if label != class_id - 1:
                raise ValueError("Incorrect class label")

            if label in wanted and label not in found:
                path = ROOT / row["image_path"]
                found[label] = str(path)

            if len(found) == len(wanted):
                break

    missing = wanted - set(found)

    if missing:
        raise ValueError(
            f"Missing classes: {sorted(missing)}"
        )

    for path in found.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    return [
        found[int(label)]
        for label in selected_labels
    ]


def python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def process_image(path, label, flag, seed):
    path = python_scalar(path)

    if isinstance(path, bytes):
        path = path.decode("utf-8")

    label = int(python_scalar(label))
    flag = bool(python_scalar(flag))
    seed = int(python_scalar(seed))

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(path)

    if flag:
        image = augment_image(image, seed=seed)

    image = cv2.resize(
        image,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = 1.0 - (
        image.astype(np.float32) / 255.0
    )

    image = image[..., np.newaxis]

    return image, np.int32(label)


def tf_process(path, label, flag, seed):
    image, target = tf.numpy_function(
        process_image,
        [path, label, flag, seed],
        [tf.float32, tf.int32]
    )

    image.set_shape((SIZE, SIZE, 1))
    target.set_shape(())

    return image, target


def make_dataset(paths, labels, augment):
    flags = np.array(
        [
            bool(augment and i % 2 == 0)
            for i in range(len(paths))
        ],
        dtype=np.bool_
    )

    seeds = np.arange(
        len(paths), dtype=np.int64
    ) + SEED

    dataset = tf.data.Dataset.from_tensor_slices((
        paths,
        np.asarray(labels, dtype=np.int32),
        flags,
        seeds
    ))

    dataset = dataset.map(
        tf_process,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH)
    dataset = dataset.prefetch(1)

    return dataset


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(SIZE, SIZE, 1)
        ),
        tf.keras.layers.Conv2D(
            8, 3, strides=2, activation="relu"
        ),
        tf.keras.layers.Conv2D(
            16, 3, strides=2, activation="relu"
        ),
        tf.keras.layers.Conv2D(
            32, 3, strides=2, activation="relu"
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(CLASSES)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=["sparse_categorical_accuracy"]
    )

    return model


def main():
    print(
        "Stage 6C - 13065 Class Smoke Test",
        flush=True
    )
    print("=" * 55)

    rng = np.random.default_rng(SEED)

    selected = rng.choice(
        CLASSES,
        size=TRAIN_COUNT + VAL_COUNT,
        replace=False
    )

    train_labels = selected[:TRAIN_COUNT]
    val_labels = selected[TRAIN_COUNT:]

    train_paths = select_images(
        SPLITS / "full_train.csv",
        train_labels
    )

    val_paths = select_images(
        SPLITS / "full_val.csv",
        val_labels
    )

    assert set(train_paths).isdisjoint(val_paths)

    train_ds = make_dataset(
        train_paths, train_labels, augment=True
    )

    val_ds = make_dataset(
        val_paths, val_labels, augment=False
    )

    model = create_model()

    assert model.output_shape[-1] == CLASSES

    print("Output classes:", CLASSES)
    print("Model parameters:", model.count_params())
    print("Training images:", TRAIN_COUNT)
    print("Validation images:", VAL_COUNT)
    print("Augmented training images:", TRAIN_COUNT // 2)
    print("Batch size:", BATCH)
    print("Output Layer Check: PASSED!")
    print("Starting training test...", flush=True)

    start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=2
    )

    elapsed = time.perf_counter() - start

    loss = float(history.history["loss"][0])
    val_loss = float(history.history["val_loss"][0])

    if not np.isfinite(loss) or not np.isfinite(val_loss):
        raise ValueError("Non-finite training loss")

    result = {
        "stage": "6C",
        "purpose": "Smoke test only",
        "output_classes": CLASSES,
        "training_images": TRAIN_COUNT,
        "validation_images": VAL_COUNT,
        "augmented_training_images": TRAIN_COUNT // 2,
        "batch_size": BATCH,
        "training_batches": TRAIN_COUNT // BATCH,
        "validation_batches": VAL_COUNT // BATCH,
        "model_parameters": model.count_params(),
        "training_loss": loss,
        "validation_loss": val_loss,
        "elapsed_seconds": elapsed,
        "final_accuracy_experiment": False
    }

    output = RESULTS / "full_model_smoke_test.json"

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    print("=" * 55)
    print("Training seconds:", round(elapsed, 2))
    print("Training loss:", round(loss, 4))
    print("Validation loss:", round(val_loss, 4))
    print("Results saved:", output)
    print("Full Model Smoke Test PASSED!")


if __name__ == "__main__":
    main()