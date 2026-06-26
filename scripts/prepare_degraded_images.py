#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add an images_degraded column for RA-VAD degrade control.")
    parser.add_argument("--input", default="data/train.parquet", help="Input parquet path.")
    parser.add_argument("--output", default="data/train_degraded.parquet", help="Output parquet path.")
    parser.add_argument("--image-column", default="images", help="Source image column.")
    parser.add_argument("--output-dir", default="data/images_degraded", help="Directory for degraded images.")
    parser.add_argument(
        "--low-tokens",
        type=int,
        default=256,
        help="Visual-token budget for the low-resolution (downsampled) pass.",
    )
    parser.add_argument(
        "--up-tokens",
        type=int,
        default=1536,
        help="Visual-token budget for the high-resolution (upsample) pass; "
        "kept for parity with unsup-opsd/ra_vad (the image is upsampled back to its "
        "ORIGINAL exact dimensions, not to this area, so the processor grid is identical).",
    )
    return parser.parse_args()


def image_entry_path(entry) -> str:
    if isinstance(entry, dict):
        if entry.get("path"):
            return entry["path"]
        if entry.get("image"):
            return image_entry_path(entry["image"])
    raise ValueError(f"Unsupported image entry for offline degradation: {type(entry)}")


def _target_pixel_area(visual_tokens: int) -> int:
    """Qwen3-VL: one visual token covers a 28x28 patch, so pixel area = tokens*28*28."""
    return int(visual_tokens) * 28 * 28


def degrade_image(src_path: str, dst_path: Path, low_tokens: int) -> None:
    """Faithful port of unsup-opsd/ra_vad/models/image_preprocess.py:degrade_image.

    downsample to ~low-token pixel area (preserving aspect ratio), then upsample
    back to the ORIGINAL exact (W, H) so the processor yields the identical
    ``image_grid_thw`` as T_hi. Only fine detail is destroyed.
    """
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            image.save(dst_path)
            return
        low_area = _target_pixel_area(low_tokens)
        scale = (low_area / (orig_w * orig_h)) ** 0.5
        low_w = max(1, round(orig_w * scale))
        low_h = max(1, round(orig_h * scale))
        low = image.resize((low_w, low_h), Image.Resampling.BICUBIC)
        degraded = low.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        degraded.save(dst_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = Path(args.output_dir)

    df = pd.read_parquet(input_path)
    if args.image_column not in df.columns:
        raise KeyError(f"Image column {args.image_column!r} not found in {input_path}")

    degraded_column = []
    for idx, images in enumerate(df[args.image_column]):
        if isinstance(images, np.ndarray):
            image_list = images.tolist()
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            image_list = [images]

        degraded_entries = []
        for image_idx, entry in enumerate(image_list):
            src_path = image_entry_path(entry)
            digest = hashlib.sha1(f"{idx}:{image_idx}:{src_path}".encode("utf-8")).hexdigest()[:16]
            suffix = Path(src_path).suffix or ".png"
            dst_path = output_dir / f"{idx:06d}_{digest}{suffix}"
            if not dst_path.exists():
                degrade_image(src_path, dst_path, args.low_tokens)
            degraded_entries.append({"path": str(dst_path)})
        degraded_column.append(np.array(degraded_entries, dtype=object))

    df["images_degraded"] = degraded_column
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} with images_degraded for {len(df)} rows.")


if __name__ == "__main__":
    main()
