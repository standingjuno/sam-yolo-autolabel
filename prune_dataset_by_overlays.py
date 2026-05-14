from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("dataset/output")


def collect_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.iterdir() if path.is_file()}


def prune_directory(directory: Path, valid_stems: set[str], do_apply: bool) -> list[Path]:
    removed: list[Path] = []

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.stem in valid_stems:
            continue

        removed.append(path)
        if do_apply:
            path.unlink()

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete image and label files whose stems do not exist in the "
            "corresponding overlays directory."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset output root. Default: dataset/output",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to prune. Default: train",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only prints what would be deleted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overlays_dir = args.output_root / "overlays" / args.split
    images_dir = args.output_root / "images" / args.split
    labels_dir = args.output_root / "labels" / args.split

    for directory in (overlays_dir, images_dir, labels_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

    valid_stems = collect_stems(overlays_dir)
    removed_images = prune_directory(images_dir, valid_stems, args.apply)
    removed_labels = prune_directory(labels_dir, valid_stems, args.apply)

    mode = "DELETED" if args.apply else "DRY-RUN"
    print(f"[{mode}] Overlay files kept: {len(valid_stems)}")
    print(f"[{mode}] Image files to delete: {len(removed_images)}")
    for path in removed_images:
        print(f"  image: {path}")
    print(f"[{mode}] Label files to delete: {len(removed_labels)}")
    for path in removed_labels:
        print(f"  label: {path}")

    if not args.apply:
        print("\nRun again with --apply to actually delete these files.")


if __name__ == "__main__":
    main()
