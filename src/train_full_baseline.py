"""Train an alternative 13,065-class model without using the test manifest."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import tempfile
import time
from zipfile import ZipFile

import cv2
import numpy as np
import tensorflow as tf

from train_full_13065_v2 import IMAGE_SIZE, NUM_CLASSES, SEED, load_image, train_generator


TRAIN_ITEMS_PER_CLASS = 200
TRAIN_ITEMS = NUM_CLASSES * TRAIN_ITEMS_PER_CLASS
TRAIN_MANIFEST = Path("data/splits/full_train.csv")
VALIDATION_MANIFEST = Path("data/splits/full_val.csv")
ARCHIVE = Path("data/raw/handwritting_data_all/all_data.zip")
ARCHIVE_MEMBER = re.compile(r"cleaned_data/(\d+)/.*_(\d+)\.png", re.IGNORECASE)
POOL_SIZE = 3
POOLED_SIZE = IMAGE_SIZE // POOL_SIZE
POOLED_PIXELS = POOLED_SIZE * POOLED_SIZE
POOLED_MAX = 255 * POOL_SIZE**2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("mlp", "cnn"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation-manifest", type=Path, help="Use an all-class labeled subset of the development validation CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for metrics and checkpoints.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0, help="TensorFlow intra-operation CPU threads. Zero uses the default.")
    parser.add_argument("--cache-dir", type=Path, help="Use a compact archive cache and augment after downsampling.")
    parser.add_argument("--benchmark-batches", type=int, default=0, help="Measure that many real batches without saving a model.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 1:
        parser.error("Epochs, batch size and workers must be positive.")
    if args.threads < 0:
        parser.error("Thread count cannot be negative.")
    if not 0 <= args.benchmark_batches <= math.ceil(TRAIN_ITEMS / args.batch_size):
        parser.error("Benchmark batches must fit within one training epoch.")
    return args


def read_manifest(path, dataset_root):
    groups = [[] for _ in range(NUM_CLASSES)]
    paths = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row["label_index"])
            if not 0 <= label < NUM_CLASSES or int(row["class_id"]) != label + 1:
                raise ValueError(f"{path}:{reader.line_num}: invalid class or label.")
            relative = Path(row["image_path"])
            if relative.anchor or not relative.parts or relative.parts[0] != "data" or ".." in relative.parts:
                raise ValueError(f"{path}:{reader.line_num}: expected a relative data path.")
            if relative in paths:
                raise ValueError(f"{path}:{reader.line_num}: duplicate image path: {relative}")
            paths.add(relative)
            groups[label].append(str(dataset_root / relative))
    if not paths:
        raise ValueError(f"{path} has no image records.")
    return groups, paths


def make_model(architecture, cached):
    image_size = POOLED_SIZE if cached else IMAGE_SIZE
    feature_layers = [] if cached else [tf.keras.layers.AveragePooling2D(pool_size=POOL_SIZE)]
    if architecture == "mlp":
        feature_layers.append(tf.keras.layers.Flatten())
    else:
        feature_layers.extend([
            tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(24, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
        ])
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 1)),
            *feature_layers,
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES),
        ],
        name=f"full_{architecture}_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="top1_accuracy")],
    )
    return model


def model_for_inference(model, cached):
    if not cached:
        return model
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    pooled = tf.keras.layers.AveragePooling2D(pool_size=POOL_SIZE)(inputs)
    return tf.keras.Model(inputs, model(pooled), name=f"{model.name}_inference")


def validation_records(groups):
    for label, paths in enumerate(groups):
        for path in paths:
            yield path, label, False, SEED


def batch_records(records, batch_size):
    for batch in itertools.batched(records, batch_size):
        yield tuple(zip(*batch))


def load_batch(paths, labels, flags, seeds):
    images = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    for index, record in enumerate(zip(paths, labels, flags, seeds)):
        images[index], _ = load_image(*record)
    return images, labels


def process_batch(paths, labels, flags, seeds):
    images, targets = tf.numpy_function(load_batch, (paths, labels, flags, seeds), (tf.float32, tf.int32))
    images.set_shape((None, IMAGE_SIZE, IMAGE_SIZE, 1))
    targets.set_shape((None,))
    return images, targets


def make_dataset(records, count, args, limit=None):
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_records(records(), args.batch_size),
        output_signature=tuple(
            tf.TensorSpec((None,), dtype) for dtype in (tf.string, tf.int32, tf.bool, tf.int64)
        ),
    )
    batches = math.ceil(count / args.batch_size)
    if limit:
        dataset = dataset.take(limit)
        batches = limit
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(batches))
    return dataset.map(process_batch, num_parallel_calls=args.workers).prefetch(2)


def archive_lookup(manifests_and_groups, dataset_root):
    lookup = {}
    offset = 0
    for manifest, groups in manifests_and_groups:
        index_by_path = {
            path: index
            for index, path in enumerate(itertools.chain.from_iterable(groups), start=offset)
        }
        with manifest.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                key = int(row["class_id"]), int(row["original_sample_index"])
                if key in lookup:
                    raise ValueError(f"Archive index appears multiple times in selected manifests: {key}")
                lookup[key] = index_by_path[str(dataset_root / row["image_path"])]
        offset += len(index_by_path)
    return lookup


def fill_archive_cache(cache, lookup, archive):
    copies = 0
    started = time.perf_counter()
    with ZipFile(archive) as source:
        entries = sorted(source.infolist(), key=lambda item: item.header_offset)
        for item in entries:
            match = ARCHIVE_MEMBER.fullmatch(item.filename)
            if match is None:
                continue
            key = tuple(map(int, match.groups()))
            position = lookup.pop(key, None)
            if position is None:
                continue

            image = cv2.imdecode(np.frombuffer(source.read(item), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Undecodable PNG in archive: {item.filename}")
            resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            pooled = cv2.resize(resized.astype(np.float32), (POOLED_SIZE, POOLED_SIZE), interpolation=cv2.INTER_AREA)
            cache[position, :-1] = POOLED_MAX - np.rint(pooled.reshape(-1) * POOL_SIZE**2)
            corners = np.concatenate((image[:8, :8].ravel(), image[:8, -8:].ravel(), image[-8:, :8].ravel(), image[-8:, -8:].ravel()))
            cache[position, -1] = (255 - int(np.median(corners))) * POOL_SIZE**2
            copies += 1
            if copies % 100_000 == 0:
                print(f"Prepared {copies:,} originals in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    if lookup:
        raise ValueError(f"{len(lookup):,} manifest originals missing from the source archive.")
    print(f"Prepared {copies:,} originals in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def load_archive_cache(cache_dir, dataset_root, manifests_and_groups, report):
    started = time.perf_counter()
    archive = dataset_root / ARCHIVE
    report["archive_sha256"] = file_hash(archive)
    inputs = [report[key] for key in ("archive_sha256", "training_manifest_sha256", "validation_manifest_sha256")]
    key = hashlib.sha256("".join(inputs).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pooled32-v1-{key}.npy"
    shape = (report["original_training_images"] + report["validation_images"], POOLED_PIXELS + 1)
    cache_reused = cache_path.exists()
    if not cache_reused:
        lookup = archive_lookup(manifests_and_groups, dataset_root)
        if len(lookup) != shape[0]:
            raise ValueError("Archive manifest lookup has an unexpected size.")
        with tempfile.TemporaryDirectory(dir=cache_dir) as partial_dir:
            partial = Path(partial_dir) / cache_path.name
            cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint16, shape=shape)
            fill_archive_cache(cache, lookup, archive)
            cache.flush()
            del cache
            partial.replace(cache_path)
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.shape != shape or cache.dtype != np.uint16:
        raise ValueError(f"Invalid archive cache shape or type: {cache_path}")
    report["archive_cache_reused"] = cache_reused
    report["archive_cache_seconds"] = time.perf_counter() - started
    return cache


def pooled_sample_images(cache, indices):
    packed = tf.numpy_function(lambda positions: cache[positions], [indices], tf.uint16)
    packed.set_shape((None, POOLED_PIXELS + 1))
    rows = tf.cast(packed, tf.float32) / POOLED_MAX
    return tf.reshape(rows[:, :-1], (-1, POOLED_SIZE, POOLED_SIZE, 1)), rows[:, -1, None, None, None]


def pooled_training_dataset(cache, groups, epoch, args):
    rng = np.random.default_rng(SEED + epoch)
    count = args.benchmark_batches * args.batch_size if args.benchmark_batches else TRAIN_ITEMS
    schedule = rng.permutation(TRAIN_ITEMS)[:count].astype(np.int32)
    slots, labels = np.divmod(schedule, NUM_CLASSES)
    sizes = np.asarray([len(group) for group in groups])
    starts = np.cumsum(sizes) - sizes
    size = sizes[labels]
    augment = slots >= size
    offsets = np.where(augment, slots - size, slots + epoch * 17) % size
    indices = (starts[labels] + offsets).astype(np.int32)

    # Use the canonical rotation, scaling and shear ranges. TensorFlow's
    # projective transform needs the inverse map from output to input pixels.
    angles = np.deg2rad((16 * rng.random(len(labels), dtype=np.float32) - 8) * augment)
    scales = 1 + (0.18 * rng.random(len(labels), dtype=np.float32) - 0.10) * augment
    shears = (0.24 * rng.random(len(labels), dtype=np.float32) - 0.12) * augment
    alpha, beta = np.cos(angles) / scales, np.sin(angles) / scales
    xx, xy, yx, yy = alpha, -beta - shears * alpha, beta, alpha - shears * beta
    center = (POOLED_SIZE - 1) / 2
    zeros = np.zeros_like(xx)
    matrix = np.column_stack((xx, xy, center * (1 - xx - xy), yx, yy, center * (1 - yx - yy), zeros, zeros))
    dataset = tf.data.Dataset.from_tensor_slices((indices, labels.astype(np.int32), matrix.astype(np.float32)))

    def process(indices, labels, matrix):
        images, background = pooled_sample_images(cache, indices)
        augmented = tf.raw_ops.ImageProjectiveTransformV3(
            images=images - background,
            transforms=matrix,
            output_shape=(POOLED_SIZE, POOLED_SIZE),
            fill_value=0.0,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
        )
        return augmented + background, labels

    return dataset.batch(args.batch_size).map(process, num_parallel_calls=args.workers).prefetch(2)


def pooled_validation_dataset(cache, groups, offset, args):
    labels = np.repeat(np.arange(NUM_CLASSES, dtype=np.int32), [len(group) for group in groups])
    indices = np.arange(offset, offset + len(labels), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((indices, labels))

    def process(indices, labels):
        images, _ = pooled_sample_images(cache, indices)
        return images, labels

    return dataset.batch(args.batch_size).map(process, num_parallel_calls=args.workers).prefetch(2)


class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches
        self.warmup = min(20, batches // 5)
        self.warmup_end = None
        self.training_seconds = None
        self.steady_batch_seconds = None

    def on_train_begin(self, logs=None):
        self.started = time.perf_counter()
        if not self.warmup:
            self.warmup_end = self.started

    def on_train_batch_end(self, batch, logs=None):
        completed = batch + 1
        now = time.perf_counter()
        if completed == self.warmup:
            self.warmup_end = now
        if completed in (1, 100, 500) or completed % 1000 == 0 or completed == self.batches:
            elapsed = now - self.started
            estimate = (self.batches - completed) * elapsed / completed
            print(f"Batches {completed:,}/{self.batches:,} | {elapsed / 60:.1f} min | ETA {estimate / 60:.1f} min", flush=True)
        if completed == self.batches:
            self.training_seconds = now - self.started
            self.steady_batch_seconds = (now - self.warmup_end) / (completed - self.warmup)


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_report(output, report):
    temporary = output / "training.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "training.json")


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    for protected in ("data", "models", "results"):
        protected_dir = dataset_root / protected
        if output.is_relative_to(protected_dir) or (cache_dir and cache_dir.is_relative_to(protected_dir)):
            raise ValueError("Choose work directories outside the input project's data, models and results.")
    train_path = dataset_root / TRAIN_MANIFEST
    canonical_validation_path = (dataset_root / VALIDATION_MANIFEST).resolve()
    validation_path = (args.validation_manifest or canonical_validation_path).resolve()
    train_groups, train_paths = read_manifest(train_path, dataset_root)
    val_groups, val_paths = read_manifest(validation_path, dataset_root)
    if any(not group or len(group) > TRAIN_ITEMS_PER_CLASS for group in train_groups):
        raise ValueError("Training must have one to 200 source images in each of the 13,065 classes.")
    if validation_path != canonical_validation_path:
        canonical_groups, _ = read_manifest(canonical_validation_path, dataset_root)
        if any(not set(selected).issubset(originals) for selected, originals in zip(val_groups, canonical_groups)):
            raise ValueError("An override must be a labeled subset of the canonical development validation manifest.")
    if any(not group for group in val_groups):
        raise ValueError("Validation must cover all 13,065 classes.")
    if not train_paths.isdisjoint(val_paths):
        raise ValueError("Development training and validation paths overlap.")

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    cv2.setNumThreads(1)
    model = make_model(args.architecture, cached=bool(cache_dir))
    output.mkdir(parents=True, exist_ok=False)
    full_batches = math.ceil(TRAIN_ITEMS / args.batch_size)
    report = {
        "architecture": args.architecture,
        "parameters": model.count_params(),
        "input_shape": [None, IMAGE_SIZE, IMAGE_SIZE, 1],
        "training_input_shape": model.input_shape,
        "output_classes": NUM_CLASSES,
        "seed": SEED,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "threads": args.threads,
        "planned_full_epochs": 0 if args.benchmark_batches else args.epochs,
        "augmentation_pipeline": (
            "96-to-32 pooled grayscale then batched seeded TensorFlow bilinear affine"
            if cache_dir else "OpenCV affine on original grayscale then canonical resize and pool"
        ),
        "training_manifest": train_path.name,
        "training_manifest_sha256": file_hash(train_path),
        "validation_manifest": validation_path.name,
        "validation_manifest_sha256": file_hash(validation_path),
        "original_training_images": len(train_paths),
        "dynamic_augmented_items_per_full_epoch": TRAIN_ITEMS - len(train_paths),
        "training_items_per_class_per_full_epoch": TRAIN_ITEMS_PER_CLASS,
        "training_items_per_full_epoch": TRAIN_ITEMS,
        "validation_images": len(val_paths),
        "validation_classes": sum(bool(group) for group in val_groups),
        "best_checkpoint": None,
        "history": [],
    }
    save_report(output, report)
    print(f"{args.architecture}: {model.count_params():,} parameters, {TRAIN_ITEMS:,} items / {full_batches:,} batches per full epoch.", flush=True)
    if cache_dir:
        cache = load_archive_cache(cache_dir, dataset_root, ((train_path, train_groups), (validation_path, val_groups)), report)
        save_report(output, report)
        validation = pooled_validation_dataset(cache, val_groups, len(train_paths), args)
    else:
        validation = make_dataset(lambda: validation_records(val_groups), len(val_paths), args)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        # V2's first full epoch used schedule seed offset 2 after its coverage run.
        if cache_dir:
            training = pooled_training_dataset(cache, train_groups, epoch + 1, args)
        else:
            records = lambda: train_generator(train_groups, TRAIN_ITEMS_PER_CLASS, epoch + 1)
            training = make_dataset(records, TRAIN_ITEMS, args, limit=args.benchmark_batches)
        progress = TrainingProgress(args.benchmark_batches or full_batches)
        if args.benchmark_batches:
            model.fit(training, callbacks=[progress], shuffle=False, verbose=0)
            report["benchmark"] = {
                "batches": args.benchmark_batches,
                "train_seconds": progress.training_seconds,
                "steady_seconds_per_batch": progress.steady_batch_seconds,
                "estimated_full_training_seconds_excluding_validation": full_batches * progress.steady_batch_seconds,
            }
            save_report(output, report)
            print(json.dumps(report["benchmark"], indent=2), flush=True)
            return

        print(f"Full epoch {epoch}/{args.epochs}", flush=True)
        started = time.perf_counter()
        history = model.fit(training, validation_data=validation, callbacks=[progress], shuffle=False, verbose=2)
        metrics = {name: float(values[-1]) for name, values in history.history.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite metrics in epoch {epoch}: {metrics}")
        report["history"].append({
            "epoch": epoch,
            "epoch_seconds_including_validation": time.perf_counter() - started,
            **metrics,
        })
        if metrics["val_top1_accuracy"] > best_accuracy:
            best_accuracy = metrics["val_top1_accuracy"]
            temporary = output / "best.tmp.keras"
            model_for_inference(model, cached=bool(cache_dir)).save(temporary)
            temporary.replace(output / "best.keras")
            report["best_checkpoint"] = "best.keras"
            report["best_epoch"] = epoch
            report["best_checkpoint_sha256"] = file_hash(output / "best.keras")
        save_report(output, report)


if __name__ == "__main__":
    main()
