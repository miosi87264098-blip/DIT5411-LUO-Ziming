"""Fit a pooled 13,065-class nearest-centroid baseline on development training images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import tensorflow as tf

from train_full_13065_v2 import IMAGE_SIZE, NUM_CLASSES, SEED, load_image
from train_full_baseline import TRAIN_ITEMS_PER_CLASS, TRAIN_MANIFEST, read_manifest


POOL_SIZE = 8
POOLED_SIZE = IMAGE_SIZE // POOL_SIZE
N_FEATURES = POOLED_SIZE**2
EXPECTED_TRAIN_ORIGINALS = 417719


def pool_image(image):
    return cv2.resize(
        image, (POOLED_SIZE, POOLED_SIZE), interpolation=cv2.INTER_AREA
    )


def augmentation_parameters(rng, number):
    angles = np.deg2rad(rng.uniform(-8, 8, number))
    scales = rng.uniform(0.90, 1.08, number)
    shears = rng.uniform(-0.12, 0.12, number)
    alpha = scales * np.cos(angles)
    beta = scales * np.sin(angles)
    horizontal = alpha - shears * beta
    vertical = beta + shears * alpha
    center = (POOLED_SIZE - 1) / 2

    transforms = np.empty((number, 2, 3), dtype=np.float32)
    transforms[:, 0, 0] = horizontal
    transforms[:, 0, 1] = vertical
    transforms[:, 0, 2] = center * (1 - horizontal - vertical)
    transforms[:, 1, 0] = -beta
    transforms[:, 1, 1] = alpha
    transforms[:, 1, 2] = center * (1 + beta - alpha)
    return transforms


def fit_class(record):
    label, paths = record
    originals = np.empty((len(paths), POOLED_SIZE, POOLED_SIZE), dtype=np.float32)
    for index, path in enumerate(paths):
        image, _ = load_image(path, label, False, SEED)
        originals[index] = pool_image(image)

    n_generated = TRAIN_ITEMS_PER_CLASS - len(paths)
    rng = np.random.default_rng([SEED, label])
    sources = rng.integers(len(paths), size=n_generated)
    transforms = augmentation_parameters(rng, n_generated)
    backgrounds = np.median(originals[:, [0, 0, -1, -1], [0, -1, 0, -1]], axis=1)
    total = np.sum(originals, axis=0, dtype=np.float64)
    for index, transform in zip(sources, transforms):
        total += cv2.warpAffine(
            originals[index],
            transform,
            (POOLED_SIZE, POOLED_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(backgrounds[index]),
        )
    return (total / TRAIN_ITEMS_PER_CLASS).reshape(N_FEATURES)


def build_model(centroids):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1)),
            tf.keras.layers.AveragePooling2D(pool_size=POOL_SIZE),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ],
        name="full_pooled_nearest_centroid",
    )
    squared_norms = np.einsum("ij,ij->i", centroids, centroids)
    model.layers[-1].set_weights([centroids.T, -0.5 * squared_norms])

    sample = np.random.default_rng(SEED).random((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    pooled = pool_image(sample).reshape(N_FEATURES)
    expected = -0.5 * np.sum((centroids - pooled) ** 2, axis=1)
    actual = model(sample[None, ...], training=False).numpy()[0]
    offset = 0.5 * np.dot(pooled, pooled)
    np.testing.assert_allclose(actual - offset, expected, atol=2e-5, rtol=2e-5)
    return model


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")

    args.output_dir.mkdir(parents=True)
    cv2.setNumThreads(1)
    manifest = args.dataset_root / TRAIN_MANIFEST
    manifest_hash = file_hash(manifest)
    groups, paths = read_manifest(manifest, args.dataset_root)
    if len(paths) != EXPECTED_TRAIN_ORIGINALS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ORIGINALS} unique development originals.")
    if not all(0 < len(group) <= TRAIN_ITEMS_PER_CLASS for group in groups):
        raise ValueError("Every class needs at least one original and at most 200 originals.")

    centroids = np.empty((NUM_CLASSES, N_FEATURES), dtype=np.float32)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for label, centroid in enumerate(executor.map(fit_class, enumerate(groups))):
            centroids[label] = centroid
            completed = label + 1
            if completed == 100 or completed % 500 == 0 or completed == NUM_CLASSES:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                print(
                    f"{completed:,}/{NUM_CLASSES:,} classes fitted | "
                    f"{rate:.1f} classes/s | ETA {(NUM_CLASSES - completed) / rate / 60:.1f} min",
                    flush=True,
                )

    model_path = args.output_dir / "full_pooled_nearest_centroid.keras"
    report_path = args.output_dir / "centroid_fit_summary.json"
    model = build_model(centroids)
    with TemporaryDirectory(dir=args.output_dir) as staging:
        staged_model = Path(staging) / model_path.name
        staged_report = Path(staging) / report_path.name
        model.save(staged_model)
        if manifest_hash != file_hash(manifest):
            raise RuntimeError("Training manifest changed during fitting.")
        report = fit_report(model, staged_model, manifest_hash, len(paths), started)
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged_model.replace(model_path)
        staged_report.replace(report_path)
    print(f"Saved {model_path} and {report_path}", flush=True)


def fit_report(model, model_path, manifest_hash, training_originals, started):
    fit_seconds = time.perf_counter() - started
    run = {
        "model": model_path.name,
        "model_sha256": file_hash(model_path),
        "estimator": "Analytic nearest centroid with squared Euclidean distance",
        "parameters": model.count_params(),
        "training_manifest": TRAIN_MANIFEST.as_posix(),
        "training_manifest_sha256": manifest_hash,
        "output_classes": NUM_CLASSES,
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 1],
        "feature_shape": [POOLED_SIZE, POOLED_SIZE],
        "training_items_per_class": TRAIN_ITEMS_PER_CLASS,
        "fitting_items_total": NUM_CLASSES * TRAIN_ITEMS_PER_CLASS,
        "original_training_items": training_originals,
        "augmented_training_items": NUM_CLASSES * TRAIN_ITEMS_PER_CLASS - training_originals,
        "augmentation": {
            "order": "96x96 canonical preprocessing, average pooling, then augmentation",
            "rotation_degrees": [-8, 8],
            "scale": [0.90, 1.08],
            "horizontal_shear": [-0.12, 0.12],
            "seed": SEED,
        },
        "fit_seconds": fit_seconds,
    }
    return run


if __name__ == "__main__":
    main()
