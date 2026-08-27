#!/usr/bin/env python3
"""Convert the VisDrone2019-DET train/val splits to YOLO format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image


SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}


def convert_annotation(raw: str, width: int, height: int) -> str | None:
    """Convert one VisDrone annotation row to a normalized YOLO label."""
    fields = raw.strip().split(",")
    if len(fields) < 6 or width <= 0 or height <= 0:
        return None
    try:
        x, y, box_w, box_h = (float(value) for value in fields[:4])
        score = int(fields[4])
        category = int(fields[5])
    except ValueError:
        return None
    if score == 0 or category < 1 or category > 10 or box_w <= 0 or box_h <= 0:
        return None
    center_x = (x + box_w / 2) / width
    center_y = (y + box_h / 2) / height
    normalized_w = box_w / width
    normalized_h = box_h / height
    if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, normalized_w, normalized_h)):
        return None
    return (
        f"{category - 1} {center_x:.6f} {center_y:.6f} "
        f"{normalized_w:.6f} {normalized_h:.6f}"
    )


def validate_output_target(source: Path, output: Path) -> Path:
    """Reject broad or source-overlapping destinations before replacement."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == Path(output.anchor):
        raise ValueError("The output directory cannot be a filesystem root.")
    if output == Path.home().resolve():
        raise ValueError("The output directory cannot be the home directory.")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("The output directory must be separate from the raw dataset.")
    return output


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def convert_split(raw_root: Path, output_root: Path, split: str, copy_files: bool) -> tuple[int, int]:
    split_root = raw_root / SPLITS[split]
    image_root = split_root / "images"
    annotation_root = split_root / "annotations"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"Expected {image_root} and {annotation_root}")

    image_output = output_root / "images" / split
    label_output = output_root / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    images = 0
    labels = 0
    for image_path in sorted(image_root.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        annotation_path = annotation_root / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        with Image.open(image_path) as image:
            width, height = image.size
        converted = [
            label
            for raw in annotation_path.read_text(encoding="utf-8").splitlines()
            if (label := convert_annotation(raw, width, height)) is not None
        ]
        link_or_copy(image_path, image_output / image_path.name, copy_files)
        (label_output / f"{image_path.stem}.txt").write_text(
            "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
        )
        images += 1
        labels += len(converted)
    return images, labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the official train and val splits")
    parser.add_argument("--output", type=Path, default=Path("datasets/VisDrone2019-DET"))
    parser.add_argument("--yaml", type=Path, default=Path("configs/visdrone.local.yaml"))
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symbolic links")
    parser.add_argument("--overwrite", action="store_true", help="Replace the exact output directory if it exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = validate_output_target(source, args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    totals = {split: convert_split(source, output, split, args.copy) for split in SPLITS}
    dataset = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        },
    }
    args.yaml.parent.mkdir(parents=True, exist_ok=True)
    args.yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    for split, (image_count, label_count) in totals.items():
        print(f"{split}: {image_count} images, {label_count} objects")
    print(f"Dataset configuration: {args.yaml}")


if __name__ == "__main__":
    main()
