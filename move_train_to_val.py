from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("dataset/output")
DEFAULT_RATIO = 0.10
DEFAULT_SEED = 42


def collect_files_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.is_file()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move paired image/label files from train to val."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset output root. Default: dataset/output",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Ratio of train pairs to move to val. Default: 0.10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible selection. Default: 42",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag, only prints what would be moved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.ratio < 1:
        raise ValueError("--ratio must be greater than 0 and less than 1.")

    images_train = args.output_root / "images" / "train"
    images_val = args.output_root / "images" / "val"
    labels_train = args.output_root / "labels" / "train"
    labels_val = args.output_root / "labels" / "val"

    for directory in (images_train, labels_train):
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

    if args.apply:
        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

    image_files = collect_files_by_stem(images_train)
    label_files = collect_files_by_stem(labels_train)
    paired_stems = sorted(set(image_files) & set(label_files))

    missing_labels = sorted(set(image_files) - set(label_files))
    missing_images = sorted(set(label_files) - set(image_files))

    if not paired_stems:
        raise RuntimeError("No paired image/label files found in train.")

    move_count = round(len(paired_stems) * args.ratio)
    move_count = max(1, move_count)

    rng = random.Random(args.seed)
    selected_stems = sorted(rng.sample(paired_stems, move_count))

    collisions: list[Path] = []
    for stem in selected_stems:
        image_dest = images_val / image_files[stem].name
        label_dest = labels_val / label_files[stem].name
        if image_dest.exists():
            collisions.append(image_dest)
        if label_dest.exists():
            collisions.append(label_dest)

    if collisions:
        joined = "\n".join(f"  {path}" for path in collisions[:20])
        extra = "" if len(collisions) <= 20 else f"\n  ... and {len(collisions) - 20} more"
        raise FileExistsError(f"Destination files already exist:\n{joined}{extra}")

    mode = "MOVED" if args.apply else "DRY-RUN"
    print(f"[{mode}] Paired train samples: {len(paired_stems)}")
    print(f"[{mode}] Ratio: {args.ratio:.2%}")
    print(f"[{mode}] Samples to move: {len(selected_stems)}")

    if missing_labels:
        print(f"[WARN] Images without labels: {len(missing_labels)}")
    if missing_images:
        print(f"[WARN] Labels without images: {len(missing_images)}")

    for stem in selected_stems:
        image_src = image_files[stem]
        label_src = label_files[stem]
        image_dest = images_val / image_src.name
        label_dest = labels_val / label_src.name

        print(f"  {image_src} -> {image_dest}")
        print(f"  {label_src} -> {label_dest}")

        if args.apply:
            shutil.move(str(image_src), str(image_dest))
            shutil.move(str(label_src), str(label_dest))

    if not args.apply:
        print("\nRun again with --apply to actually move these files.")


if __name__ == "__main__":
    main()
