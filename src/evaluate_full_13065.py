"""Evaluate a 13,065-class checkpoint on original test or validation images."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from hashlib import file_digest, sha256
from io import BytesIO, TextIOWrapper
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import NamedTuple
from zipfile import ZipFile

import cv2
import numpy as np
import tensorflow as tf

from audit_image_overlap import FILENAME as ARCHIVE_IMAGE_PATTERN
from train_full_13065_v2 import IMAGE_SIZE, NUM_CLASSES, SEED, load_image, preprocess_image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "full_13065_v2_full_epoch_003.keras"
SPLITS = {"test": ("test.csv", 162550), "validation": ("full_val.csv", 104408)}
BATCH_SIZE = 32
REPORT_IMAGES = 8000


class EvalRow(NamedTuple):
    path: str
    label: int
    original_index: int | None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Project folder containing data/")
    parser.add_argument("--checkpoint", type=Path, help="Path to the .keras checkpoint")
    parser.add_argument("--manifest", type=Path, help="CSV containing rows from the selected original split")
    parser.add_argument("--split", choices=SPLITS, help="Evaluation split (default: test)")
    parser.add_argument("--output-dir", type=Path, help="Separate results folder for a custom evaluation")
    parser.add_argument("--expected-images", type=int, help="Require this many rows in a custom manifest")
    parser.add_argument("--archive", type=Path, help="Read original PNGs directly from this ZIP")
    parser.add_argument("--batch-size", type=int, help=f"Inference batch size (default: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, help="Threads to load images (default: 1)")
    args = parser.parse_args()
    args.legacy = not any(value is not None for value in vars(args).values())
    if not args.legacy and args.output_dir is None:
        parser.error("--output-dir is required for a custom evaluation")
    for flag in ("expected_images", "batch_size", "workers"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    return args


def manifest_rows(path, dataset_root, expected=None):
    content = path.read_bytes()
    rows = {}
    required = {"image_path", "label_index", "class_id"}
    with TextIOWrapper(BytesIO(content), encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing required manifest columns")
        for line, row in enumerate(reader, start=2):
            label = int(row["label_index"])
            if not 0 <= label < NUM_CLASSES or int(row["class_id"]) != label + 1:
                raise ValueError(f"{path}:{line}: invalid label or class ID")
            relative = Path(row["image_path"])
            if relative.anchor or ".." in relative.parts or not relative.name:
                raise ValueError(f"{path}:{line}: expected an image path relative to the dataset root")
            image = str(dataset_root / relative)
            key = os.path.normcase(image)
            if key in rows:
                raise ValueError(f"{path}:{line}: duplicate image path {relative}")
            value = row.get("original_sample_index")
            original_index = int(value) if value else None
            rows[key] = EvalRow(image, label, original_index)
    if not rows or (expected is not None and len(rows) != expected):
        expectation = "at least one row" if expected is None else f"{expected:,} rows"
        raise ValueError(f"{path}: expected {expectation}, found {len(rows):,}")
    return rows, sha256(content).hexdigest()


def select_rows(dataset_root, split, manifest, expected_images):
    filename, split_size = SPLITS[split]
    canonical = (dataset_root / "data" / "splits" / filename).resolve()
    path = manifest.resolve() if manifest else canonical
    source_rows, source_sha = manifest_rows(canonical, dataset_root, split_size)
    if path == canonical:
        rows, manifest_sha = source_rows, source_sha
    else:
        rows, manifest_sha = manifest_rows(path, dataset_root, expected_images)
        for key, row in rows.items():
            if key not in source_rows or source_rows[key].label != row.label:
                raise ValueError(f"{path}: image or label absent from canonical {split} manifest: {key}")
        rows = {key: source_rows[key] for key in rows}
    if expected_images is not None and len(rows) != expected_images:
        raise ValueError(f"{path}: expected {expected_images:,} rows, found {len(rows):,}")
    provenance = {
        "evaluation_manifest": path.name,
        "evaluation_manifest_sha256": manifest_sha,
        "canonical_manifest": canonical.name,
        "canonical_manifest_sha256": source_sha,
        "canonical_split_images": split_size,
    }
    return list(rows.values()), provenance


def source_sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_extracted(row):
    image, _ = load_image(row.path, row.label, False, SEED)
    return image


def archive_reader(stack, path, rows):
    if any(row.original_index is None or row.original_index < 0 for row in rows):
        raise ValueError("Archive evaluation requires numeric original_sample_index in the canonical split")
    keys = {(row.label + 1, row.original_index) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("The canonical split repeats an original archive sample ID")
    handle = stack.enter_context(path.open("rb"))
    digest = file_digest(handle, "sha256").hexdigest()
    handle.seek(0)
    archive = stack.enter_context(ZipFile(handle))
    entries = {}
    for info in archive.infolist():
        match = ARCHIVE_IMAGE_PATTERN.fullmatch(info.filename)
        if match is None:
            continue
        key = tuple(map(int, match.groups()))
        if key in keys:
            if key in entries:
                raise ValueError(f"Archive repeats original sample: {info.filename}")
            entries[key] = info
    if len(entries) != len(keys):
        missing = min(keys - entries.keys())
        raise ValueError(f"Archive is missing class/sample {missing}; found {len(entries):,} of {len(keys):,}")

    def read_image(row):
        info = entries[row.label + 1, row.original_index]
        raw = np.frombuffer(archive.read(info), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode archive image: {info.filename}")
        return preprocess_image(image)

    return read_image, {"image_source": "original_archive", "archive": path.name, "archive_sha256": digest}


def predict_batch(model, rows, fixed_batch, model_input, load_images):
    count = len(rows)
    inputs = np.zeros((fixed_batch or count, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    labels = np.asarray([row.label for row in rows], dtype=np.int32)
    for index, image in enumerate(load_images(rows)):
        inputs[index] = image
    supplied = tf.nest.map_structure(lambda _: inputs, model_input)
    (scores,) = tf.nest.flatten(model(supplied, training=False))
    scores = scores[:count]
    if not bool(tf.reduce_all(tf.math.is_finite(scores)).numpy()):
        raise ValueError("Model returned a non-finite prediction")
    # TensorFlow top_k ranks lower class indices first when scores are equal.
    top5 = tf.math.top_k(scores, k=5, sorted=True).indices.numpy()
    return labels, top5


def evaluate(model, rows, batch_size, fixed_batch, model_input, load_images):
    support = np.zeros(NUM_CLASSES, dtype=np.int64)
    correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    top5_correct = 0
    confusions = Counter()
    errors = []
    start = time.perf_counter()
    total = len(rows)
    for offset in range(0, total, batch_size):
        batch = rows[offset:offset + batch_size]
        labels, top5 = predict_batch(model, batch, fixed_batch, model_input, load_images)
        predicted = top5[:, 0]
        hits = predicted == labels
        support += np.bincount(labels, minlength=NUM_CLASSES)
        correct += np.bincount(labels[hits], minlength=NUM_CLASSES)
        top5_correct += int(np.any(top5 == labels[:, None], axis=1).sum())
        for index in np.flatnonzero(~hits):
            true_id = int(labels[index]) + 1
            predicted_id = int(predicted[index]) + 1
            confusions[true_id, predicted_id] += 1
            if len(errors) < 200:
                candidates = " ".join(str(int(label) + 1) for label in top5[index])
                errors.append((batch[index].path, true_id, predicted_id, candidates))
        done = offset + len(batch)
        if done // REPORT_IMAGES != offset // REPORT_IMAGES or done == total:
            elapsed = time.perf_counter() - start
            remaining = (total - done) * elapsed / done
            print(
                f"{done:,}/{total:,} ({done / total:.1%}) | "
                f"Top-1 {correct.sum() / done:.2%} | ETA {remaining / 60:.1f} min",
                flush=True,
            )
    present = support > 0
    top1_correct = int(correct.sum())
    metrics = {
        "output_classes": NUM_CLASSES,
        "evaluated_classes": int(present.sum()),
        "classes_without_images": int((~present).sum()),
        "evaluated_images": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
        "macro_class_accuracy": float(np.mean(correct[present] / support[present])),
        "evaluation_seconds": time.perf_counter() - start,
        "batch_size": batch_size,
        "random_top1_baseline": 1.0 / NUM_CLASSES,
        "top_k_tie_break": "ascending_class_id",
        "common_confusions": [
            {"true_class_id": true, "predicted_class_id": predicted, "count": count}
            for (true, predicted), count in confusions.most_common(30)
        ],
    }
    return metrics, list(zip(support.tolist(), correct.tolist())), errors


def write_outputs(filenames, directory, metrics, per_class, errors, split):
    metrics_path, class_path, error_path = (directory / name for name in filenames)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with class_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", f"{split}_images", "correct", "accuracy"])
        for class_id, (count, correct) in enumerate(per_class, start=1):
            writer.writerow([class_id, count, correct, correct / count if count else ""])
    with error_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class_id", "predicted_class_id", "top5_class_ids"])
        writer.writerows(errors)


def main():
    args = arguments()
    split = args.split or "test"
    dataset_root = (args.dataset_root or ROOT).resolve()
    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    output_dir = (args.output_dir or ROOT / "results").resolve()
    stem = "full_13065_v2_test" if args.legacy else f"{checkpoint.stem}_{split}"
    filenames = [f"{stem}_{suffix}" for suffix in ("metrics.json", "per_class.csv", "errors_sample.csv")]
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / f".{stem}.evaluation.lock"
    try:
        with lock.open("x", encoding="utf-8"):
            pass
    except FileExistsError as error:
        raise FileExistsError(f"Another evaluation is using {output_dir / stem}") from error

    try:
        if not args.legacy and any((output_dir / name).exists() for name in filenames):
            raise FileExistsError(f"Results for {stem} already exist in {output_dir}")
        rows, provenance = select_rows(dataset_root, split, args.manifest, args.expected_images)
        saved_hash = source_sha(checkpoint)
        model = tf.keras.models.load_model(checkpoint, compile=False)
        if not (
            len(model.inputs) == 1
            and model.inputs[0].shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, 1)
            and len(model.outputs) == 1
            and model.outputs[0].shape[1:] == (NUM_CLASSES,)
        ):
            raise ValueError(f"Expected one {IMAGE_SIZE}x{IMAGE_SIZE}x1 input and {NUM_CLASSES:,}-class output")
        if source_sha(checkpoint) != saved_hash:
            raise ValueError("Checkpoint changed while it was being loaded")
        fixed_batch = model.inputs[0].shape[0]
        batch_size = args.batch_size or fixed_batch or BATCH_SIZE
        if fixed_batch and fixed_batch != batch_size:
            raise ValueError(f"This checkpoint requires batch size {fixed_batch}")
        workers = args.workers or 1
        try:
            model_input = model.input
        except AttributeError:
            model_input = model.inputs[0]
        with ExitStack() as stack:
            if args.archive:
                read_image, image_source = archive_reader(stack, args.archive, rows)
            else:
                read_image = read_extracted
                image_source = {"image_source": "extracted_images"}
            if workers > 1:
                executor = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
                load_images = lambda batch: executor.map(read_image, batch)
            else:
                load_images = lambda batch: map(read_image, batch)
            print(
                f"Evaluating {checkpoint.name} on {len(rows):,} {split} rows "
                f"({image_source['image_source']}, batch size {batch_size})",
                flush=True,
            )
            result, per_class, errors = evaluate(
                model, rows, batch_size, fixed_batch, model_input, load_images
            )
        is_full_split = len(rows) == provenance["canonical_split_images"]
        evaluation_type = (
            "independent_test_set" if split == "test" else "validation_set"
        ) if is_full_split else f"{split}_manifest_subset"
        metrics = {
            "model": checkpoint.name,
            "checkpoint_sha256": saved_hash,
            "evaluation_type": evaluation_type,
            "split": split,
            **provenance,
            **image_source,
            **result,
            "image_workers": workers,
            f"{split}_images": result["evaluated_images"],
        }
        if split == "test":
            metrics["tested_classes"] = result["evaluated_classes"]
            metrics["classes_without_test_images"] = result["classes_without_images"]
        with TemporaryDirectory(dir=output_dir) as staging:
            write_outputs(filenames, Path(staging), metrics, per_class, errors, split)
            # Publish the summary last so it only appears once both CSVs are ready.
            for name in filenames[1:] + filenames[:1]:
                (Path(staging) / name).replace(output_dir / name)
        print(
            f"Top-1 {metrics['top1_accuracy']:.2%} | "
            f"Top-5 {metrics['top5_accuracy']:.2%} | "
            f"Macro class accuracy {metrics['macro_class_accuracy']:.2%}"
        )
        print(f"Results saved in {output_dir}")
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
