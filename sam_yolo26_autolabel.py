from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULTS = {
    "val_ratio": 0.2,
    "seed": 42,
    "shuffle_split": False,
    "device": "cuda",
    "amp_dtype": "bfloat16",
    "sam_version": "sam3.1",
    "min_score": 0.85,
    "min_area": 100,
    "simplify": 0.003,
    "max_points": 300,
    "checkpoint_path": None,
    "copy_mode": "copy",
    "overwrite": False,
    "no_save_previews": False,
}

PALETTE = np.array(
    [
        [255, 64, 64],
        [64, 180, 255],
        [80, 220, 120],
        [255, 190, 64],
        [190, 90, 255],
        [255, 90, 180],
        [80, 230, 230],
        [220, 220, 80],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ClassPrompt:
    class_id: int
    name: str
    prompt: str


@dataclass(frozen=True)
class MaskAnnotation:
    class_id: int
    class_name: str
    score: float
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use SAM 3 image text prompts to auto-label images as an "
            "Ultralytics YOLO26 segmentation dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON config file containing run parameters and classes. "
            "CLI args override config values when both are provided."
        ),
    )
    parser.add_argument("--images", type=str, default=None, help="Input image folder.")
    parser.add_argument("--out", type=str, default=None, help="Output YOLO dataset folder.")
    parser.add_argument(
        "--classes",
        default=None,
        type=str,
        help=(
            "Class prompts source. Either a JSON file path, or omit it when config JSON "
            "contains a 'classes' field."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for train/val split.")
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        default=None,
        help="Shuffle images before train/val split. By default, files are processed in sorted order.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--amp-dtype",
        default=None,
        choices=["none", "float16", "bfloat16"],
        help="CUDA autocast dtype for SAM inference. Use 'none' for full FP32.",
    )
    parser.add_argument(
        "--sam-version",
        default=None,
        choices=["sam3.1", "sam3"],
        help="SAM checkpoint family to download from Hugging Face when --checkpoint-path is omitted.",
    )
    parser.add_argument("--min-score", type=float, default=None, help="SAM score threshold.")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum mask contour area in pixels.")
    parser.add_argument(
        "--simplify",
        type=float,
        default=None,
        help=(
            "Polygon simplification ratio. This is multiplied by contour perimeter. "
            "Use 0 for no simplification."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum polygon points per object after simplification/resampling.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Optional local SAM checkpoint. If omitted, the SAM package downloads "
            "the default image checkpoint from Hugging Face after hf auth login."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "link"],
        default=None,
        help="Copy images or create hard links in the YOLO dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--no-save-previews",
        action="store_true",
        default=None,
        help="Do not save overlay and mask preview images.",
    )
    return parser.parse_args()


def parse_class_prompts(data: object) -> list[ClassPrompt]:
    prompts: list[ClassPrompt] = []

    if isinstance(data, dict):
        for idx, (name, prompt) in enumerate(data.items()):
            prompts.append(ClassPrompt(idx, str(name), str(prompt)))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("List classes must contain objects with at least a 'name' field.")
            name = str(item["name"])
            prompt = str(item.get("prompt", name))
            class_id = int(item.get("id", idx))
            prompts.append(ClassPrompt(class_id, name, prompt))
    else:
        raise ValueError("--classes must be a JSON object or list.")

    ids = [item.class_id for item in prompts]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Class ids must be contiguous and start at 0.")
    return sorted(prompts, key=lambda item: item.class_id)


def load_class_prompts(path: Path) -> list[ClassPrompt]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_class_prompts(data)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def merge_args_with_config(args: argparse.Namespace) -> tuple[argparse.Namespace, Path]:
    config: dict = {}
    config_base = Path.cwd()

    config_path: Path | None = None
    if args.config:
        config_path = args.config.resolve()
    else:
        default_candidates = [
            Path.cwd() / "autolabel_config.json",
            Path(__file__).resolve().parent / "autolabel_config.json",
        ]
        config_path = next((candidate for candidate in default_candidates if candidate.exists()), None)

    if config_path:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("--config must point to a JSON object.")
        config_base = config_path.parent

    def pick(name: str):
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        if name in config:
            return config[name]
        return DEFAULTS[name]

    images_value = args.images if args.images is not None else config.get("images")
    out_value = args.out if args.out is not None else config.get("out")
    classes_value = args.classes if args.classes is not None else config.get("classes")

    if images_value is None:
        raise ValueError("Missing images path. Set --images or provide 'images' in config JSON.")
    if out_value is None:
        raise ValueError("Missing output path. Set --out or provide 'out' in config JSON.")
    if classes_value is None:
        raise ValueError("Missing classes. Set --classes or provide 'classes' in config JSON.")

    merged = argparse.Namespace(
        images=resolve_path(images_value, config_base),
        out=resolve_path(out_value, config_base),
        classes=classes_value,
        val_ratio=float(pick("val_ratio")),
        seed=int(pick("seed")),
        shuffle_split=bool(pick("shuffle_split")),
        device=str(pick("device")),
        amp_dtype=str(pick("amp_dtype")),
        sam_version=str(pick("sam_version")),
        min_score=float(pick("min_score")),
        min_area=int(pick("min_area")),
        simplify=float(pick("simplify")),
        max_points=int(pick("max_points")),
        checkpoint_path=pick("checkpoint_path"),
        copy_mode=str(pick("copy_mode")),
        overwrite=bool(pick("overwrite")),
        no_save_previews=bool(pick("no_save_previews")),
    )

    if merged.checkpoint_path is not None:
        merged.checkpoint_path = resolve_path(str(merged.checkpoint_path), config_base)

    return merged, config_base


def class_prompts_from_source(source: object, base_dir: Path) -> list[ClassPrompt]:
    if isinstance(source, (dict, list)):
        return parse_class_prompts(source)
    if isinstance(source, str):
        return load_class_prompts(resolve_path(source, base_dir))
    if isinstance(source, Path):
        return load_class_prompts(resolve_path(source, base_dir))
    raise ValueError("Classes source must be a JSON object/list or a JSON file path.")


def iter_images(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def natural_sort_key(path: Path) -> list[int | str]:
    relative = path.as_posix().lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", relative)]


def split_images(
    images: list[Path], val_ratio: float, seed: int, shuffle: bool
) -> tuple[list[Path], list[Path]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be >= 0 and < 1.")
    ordered = images[:]
    if shuffle:
        random.Random(seed).shuffle(ordered)
    val_count = int(round(len(ordered) * val_ratio))
    if val_count == 0:
        return ordered, []
    train = ordered[:-val_count]
    val = ordered[-val_count:]
    if not train and val:
        train, val = val[:1], val[1:]
    return train, val


def prepare_output(root: Path, overwrite: bool, save_previews: bool) -> None:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} already exists and is not empty. Pass --overwrite to use it.")
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if save_previews:
            (root / "overlays" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)


def copy_or_link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def dataset_stem(image_path: Path, image_root: Path) -> str:
    relative = image_path.relative_to(image_root)
    safe_parts = [part.replace(" ", "_") for part in relative.with_suffix("").parts]
    return "__".join(safe_parts)


def ensure_local_sam3_repo_on_path() -> None:
    """Allow running with a bundled ./sam3 repository layout."""
    script_dir = Path(__file__).resolve().parent
    local_sam3_repo = script_dir / "sam3"
    local_sam3_package = local_sam3_repo / "sam3"
    if local_sam3_package.exists():
        repo_path = str(local_sam3_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)


def build_sam_processor(device: str, sam_version: str, checkpoint_path: Path | None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    ensure_local_sam3_repo_on_path()
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3.model_builder as sam3_model_builder

    resolved_checkpoint = (
        str(checkpoint_path)
        if checkpoint_path
        else sam3_model_builder.download_ckpt_from_hf(version=sam_version)
    )
    sam3_root = Path(sam3_model_builder.__file__).resolve().parent
    bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        raise FileNotFoundError(f"SAM3 tokenizer file not found: {bpe_path}")

    model = sam3_model_builder.build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        checkpoint_path=resolved_checkpoint,
        load_from_HF=False,
    )
    return Sam3Processor(model)


def autocast_context(device: str, amp_dtype: str):
    if device != "cuda" or amp_dtype == "none":
        return torch.amp.autocast(device_type="cuda", enabled=False)
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def as_numpy_masks(masks) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks > 0


def as_numpy_scores(scores, count: int) -> np.ndarray:
    if scores is None:
        return np.ones(count, dtype=np.float32)
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def resample_polygon(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices]


def mask_to_yolo_polygons(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: int,
    simplify: float,
    max_points: int,
) -> list[list[float]]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        if simplify > 0:
            epsilon = simplify * cv2.arcLength(contour, closed=True)
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        points = contour.reshape(-1, 2)
        points = resample_polygon(points, max_points)
        if len(points) < 3:
            continue

        normalized: list[float] = []
        for x, y in points:
            normalized.extend(
                [
                    min(max(float(x) / image_width, 0.0), 1.0),
                    min(max(float(y) / image_height, 0.0), 1.0),
                ]
            )
        polygons.append(normalized)

    return polygons


def save_preview_images(
    image_path: Path,
    annotations: list[MaskAnnotation],
    overlay_path: Path,
    mask_path: Path,
    alpha: float = 0.45,
) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    overlay = image.astype(np.float32)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for annotation in annotations:
        mask = annotation.mask
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color = PALETTE[annotation.class_id % len(PALETTE)]
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color.astype(np.float32) * alpha
        color_mask[mask] = color

    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    for annotation in annotations:
        mask_u8 = annotation.mask.astype(np.uint8) * 255
        if mask_u8.shape != (height, width):
            mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_tuple = tuple(int(value) for value in PALETTE[annotation.class_id % len(PALETTE)])
        cv2.drawContours(overlay_u8, contours, -1, color_tuple, thickness=2)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label = f"{annotation.class_name} {annotation.score:.2f}"
            cv2.putText(
                overlay_u8,
                label,
                (x, max(y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_tuple,
                1,
                cv2.LINE_AA,
            )

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_u8).save(overlay_path)
    Image.fromarray(color_mask).save(mask_path)


@torch.inference_mode()
def label_image(
    processor,
    image_path: Path,
    class_prompts: Iterable[ClassPrompt],
    device: str,
    amp_dtype: str,
    min_score: float,
    min_area: int,
    simplify: float,
    max_points: int,
) -> tuple[list[str], list[MaskAnnotation]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    lines: list[str] = []
    annotations: list[MaskAnnotation] = []
    with autocast_context(device, amp_dtype):
        inference_state = processor.set_image(image)

        for class_prompt in class_prompts:
            output = processor.set_text_prompt(state=inference_state, prompt=class_prompt.prompt)
            masks = as_numpy_masks(output.get("masks"))
            scores = as_numpy_scores(output.get("scores"), len(masks))

            for mask, score in zip(masks, scores):
                if float(score) < min_score:
                    continue
                polygons = mask_to_yolo_polygons(
                    mask=mask,
                    image_width=width,
                    image_height=height,
                    min_area=min_area,
                    simplify=simplify,
                    max_points=max_points,
                )
                for polygon in polygons:
                    coords = " ".join(f"{value:.6f}" for value in polygon)
                    lines.append(f"{class_prompt.class_id} {coords}")
                if polygons:
                    annotations.append(
                        MaskAnnotation(
                            class_id=class_prompt.class_id,
                            class_name=class_prompt.name,
                            score=float(score),
                            mask=mask,
                        )
                    )

    return lines, annotations


def write_data_yaml(root: Path, class_prompts: list[ClassPrompt]) -> None:
    names = {item.class_id: item.name for item in class_prompts}
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def process_split(
    processor,
    images: list[Path],
    input_root: Path,
    split: str,
    out_root: Path,
    class_prompts: list[ClassPrompt],
    args: argparse.Namespace,
) -> None:
    for src in tqdm(images, desc=f"label {split}"):
        stem = dataset_stem(src, input_root)
        dst_image = out_root / "images" / split / f"{stem}{src.suffix.lower()}"
        dst_label = out_root / "labels" / split / f"{stem}.txt"
        dst_overlay = out_root / "overlays" / split / f"{stem}.jpg"
        dst_mask = out_root / "masks" / split / f"{stem}.png"
        copy_or_link_image(src, dst_image, args.copy_mode)
        lines, annotations = label_image(
            processor=processor,
            image_path=src,
            class_prompts=class_prompts,
            device=args.device,
            amp_dtype=args.amp_dtype,
            min_score=args.min_score,
            min_area=args.min_area,
            simplify=args.simplify,
            max_points=args.max_points,
        )
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not args.no_save_previews:
            save_preview_images(src, annotations, dst_overlay, dst_mask)


def main() -> None:
    raw_args = parse_args()
    args, config_base = merge_args_with_config(raw_args)

    images = iter_images(args.images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.images}.")

    class_prompts = class_prompts_from_source(args.classes, config_base)
    train_images, val_images = split_images(images, args.val_ratio, args.seed, args.shuffle_split)
    prepare_output(args.out, args.overwrite, not args.no_save_previews)

    processor = build_sam_processor(args.device, args.sam_version, args.checkpoint_path)
    process_split(processor, train_images, args.images, "train", args.out, class_prompts, args)
    process_split(processor, val_images, args.images, "val", args.out, class_prompts, args)
    write_data_yaml(args.out, class_prompts)

    print(f"Done: {args.out.resolve()}")
    print(f"Train images: {len(train_images)}, val images: {len(val_images)}")
    print(f"YOLO data file: {(args.out / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
