"""Compare exact original PNG bytes and decoded grayscale pixels across split CSVs."""

import argparse
from collections import defaultdict
from csv import DictReader, DictWriter
from hashlib import file_digest, sha256
from io import BytesIO, TextIOWrapper
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import monotonic
from typing import NamedTuple
from zipfile import ZipFile

SPLITS = ('dev_train', 'validation', 'test')
CSV_NAMES = ('full_train', 'full_val', 'test')
FILENAME = re.compile(r'^cleaned_data/(\d+)/.*_(\d+)\.png$', re.I)
CSV_KEYS = ('image_path', 'class_id', 'label_index', 'sample_position', 'original_sample_index')
ARCHIVE = Path('data/raw/handwritting_data_all/all_data.zip')
NUM_CLASSES = 13065


class Sample(NamedTuple):
    split: int
    class_id: int
    position: int
    original_index: int

    @property
    def image_path(self):
        return f'data/extracted/cleaned_data/{self.class_id}/sample_{self.position:04d}.png'


def read_membership(root):
    lookup = {}
    paths = set()
    split_rows = [[] for _ in SPLITS]
    source_details = {}
    for group, name in enumerate(CSV_NAMES):
        path = root / f'data/splits/{name}.csv'
        contents = path.read_bytes()
        with TextIOWrapper(BytesIO(contents), encoding='utf-8-sig', newline='') as source:
            reader = DictReader(source)
            if not set(CSV_KEYS).issubset(reader.fieldnames or ()):
                raise ValueError(f'{path}: manifest is missing one or more required columns')
            for line, row in enumerate(reader, start=2):
                class_id = int(row['class_id'])
                index = int(row['original_sample_index'])
                position = int(row['sample_position'])
                member = Sample(group, class_id, position, index)
                if (
                    not 1 <= class_id <= NUM_CLASSES
                    or int(row['label_index']) != class_id - 1
                    or min(position, index) < 0
                    or member.image_path != row['image_path']
                ):
                    raise ValueError(f'{path}:{line}: invalid original image path or label')
                key = class_id, index
                if key in lookup or member.image_path in paths:
                    raise ValueError(f'{path}:{line}: repeated original sample or image path')
                paths.add(member.image_path)
                lookup[key] = member
                split_rows[group].append(member)
        if not split_rows[group]:
            raise ValueError(f'{path}: an original split cannot be empty')
        source_details[SPLITS[group]] = {
            'path': f'data/splits/{name}.csv',
            'sha256': sha256(contents).hexdigest(),
            'images': len(split_rows[group]),
        }
    return lookup, split_rows, source_details


def analyze_content(table, validation_rows, test_rows):
    matches = {pair: [] for pair in combinations(range(3), 2)}
    for values in table.values():
        if isinstance(values, Sample):
            continue
        source = defaultdict(set)
        for row in values:
            source[row.split].add(row)
        for a, b in matches:
            if a in source and b in source:
                matches[a, b].append((source[a], source[b]))
    pair_result = {}
    removed = set()
    validation_removed = set()
    for (a, b), groups in matches.items():
        left = set()
        right = set()
        right_same = set()
        right_diff = set()
        for aa, bb in groups:
            left.update(aa)
            right.update(bb)
            left_ids = {row.class_id for row in aa}
            right_same.update(row for row in bb if row.class_id in left_ids)
            right_diff.update(row for row in bb if left_ids - {row.class_id})
        if b == 2:
            removed.update(right)
        if (a, b) == (0, 1):
            validation_removed = right
        pair_result[f'{SPLITS[a]}_vs_{SPLITS[b]}'] = {
            'shared_hashes': len(groups),
            'left_rows': len(left),
            'right_rows': len(right),
            'right_rows_with_same_label_in_left': len(right_same),
            'right_rows_with_different_label_in_left': len(right_diff),
            'right_classes': len({row.class_id for row in right}),
        }
    kept = set(test_rows) - removed
    validation_kept = set(validation_rows) - validation_removed
    return dict(
        pairwise=pair_result,
        validation_rows=len(validation_rows),
        validation_overlaps_train=len(validation_removed),
        validation_remaining=len(validation_kept),
        validation_classes_remaining=len({row.class_id for row in validation_kept}),
        test_rows=len(test_rows),
        test_overlaps_train_or_validation=len(removed),
        test_overlap_classes=len({row.class_id for row in removed}),
        test_remaining=len(kept),
        test_classes_remaining=len({row.class_id for row in kept}),
    ), validation_removed, removed


def add_digest(table, digest, member):
    prior = table.get(digest)
    if prior is None:
        table[digest] = member
    elif isinstance(prior, Sample):
        table[digest] = [prior, member]
    else:
        prior.append(member)


def write_manifest(path, rows, selected):
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = DictWriter(target, fieldnames=CSV_KEYS)
        writer.writeheader()
        for member in rows:
            if member in selected:
                writer.writerow({
                    'image_path': member.image_path,
                    'class_id': member.class_id,
                    'label_index': member.class_id - 1,
                    'sample_position': member.position,
                    'original_sample_index': member.original_index,
                })


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', '--project', dest='project', type=Path, required=True)
    parser.add_argument('--output-dir', '--out', dest='out', type=Path, required=True)
    parser.add_argument('--source', choices=('archive', 'extracted'), default='archive')
    parser.add_argument('--archive', type=Path, help='Override the original ZIP location in archive mode')
    parser.add_argument(
        '--grayscale', action='store_true',
        help='Also compare identical decoded grayscale pixels using OpenCV',
    )
    args = parser.parse_args()
    if args.archive is not None and args.source != 'archive':
        parser.error('--archive applies only in archive mode')
    return args


def scan_images(args, members, add_image):
    start = monotonic()
    if args.source == 'archive':
        archive = args.archive or args.project / ARCHIVE
        with archive.open('rb') as handle:
            archive_sha = file_digest(handle, 'sha256').hexdigest()
            handle.seek(0)
            with ZipFile(handle) as source:
                entries = sorted(source.infolist(), key=lambda info: info.header_offset)
                images = (info for info in entries if not info.is_dir())
                for number, info in enumerate(images, 1):
                    match = FILENAME.fullmatch(info.filename)
                    if match is None:
                        raise ValueError(f'Unexpected archive member: {info.filename}')
                    member = members.pop(tuple(map(int, match.groups())), None)
                    if member is None:
                        raise ValueError(f'Archive image missing or repeated in manifests: {info.filename}')
                    add_image(member, source.read(info))
                    if number % 100000 == 0:
                        print(f'{number:,} images in {monotonic() - start:.1f}s', flush=True)
        if members:
            raise ValueError(f'{len(members):,} manifest rows absent from the archive')
        return {'archive': archive.name, 'archive_sha256': archive_sha}

    images = sorted(members.values(), key=lambda row: (row.class_id, row.position))
    for number, member in enumerate(images, 1):
        add_image(member, (args.project / member.image_path).read_bytes())
        if number % 100000 == 0:
            print(f'{number:,} images in {monotonic() - start:.1f}s', flush=True)
    return {}


def audit(args, staging):
    if args.grayscale:
        import cv2
        import numpy as np

    members, (_, validation, test), manifest_sources = read_membership(args.project)
    png_hashes = {}
    pixel_hashes = {}
    dimensions = set()

    def add_image(member, data):
        add_digest(png_hashes, sha256(data).digest(), member)
        if args.grayscale:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f'Undecodable image in {member}')
            dimensions.add(image.shape)
            shape_bytes = np.array(image.shape, dtype='<i4').tobytes()
            digest = sha256(shape_bytes)
            digest.update(image)
            add_digest(pixel_hashes, digest.digest(), member)

    source_details = scan_images(args, members, add_image)
    summary = {'source': args.source, 'splits': manifest_sources, **source_details}
    for name, table in [('png_bytes', png_hashes), ('grayscale_pixels', pixel_hashes)]:
        if not table:
            continue
        summary[name], validation_removed, test_removed = analyze_content(table, validation, test)
        for split, rows, removed in (
            ('validation', validation, validation_removed), ('test', test, test_removed),
        ):
            for cohort, selected in [('overlap', removed), ('nonoverlap', set(rows) - removed)]:
                path = staging / f'{args.source}_{name}_{cohort}_{split}.csv'
                write_manifest(path, rows, selected)
    if args.grayscale:
        summary['grayscale_dimensions'] = sorted(dimensions)
    report = staging / f'{args.source}_content_audit.json'
    report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return report.name


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / f'.{args.source}_content_audit.lock'
    try:
        with lock.open('x', encoding='utf-8'):
            pass
    except FileExistsError as error:
        raise FileExistsError(f'Another audit of this source is using {args.out}') from error
    try:
        if any(args.out.glob(f'{args.source}_*')):
            raise FileExistsError(f'Results for source {args.source} already exist in {args.out}')
        with TemporaryDirectory(dir=args.out) as staging:
            staging = Path(staging)
            report = audit(args, staging)
            for cohort in staging.glob('*.csv'):
                cohort.replace(args.out / cohort.name)
            (staging / report).replace(args.out / report)
        print(f'Content audit saved in {args.out / report}', flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
