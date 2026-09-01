"""VisDrone YOLO-to-COCO conversion and extended metric extraction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence


def coco_bbox_from_yolo_line(line: str, *, width: int, height: int) -> dict[str, object]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected five YOLO fields, got {len(fields)}")
    category, center_x, center_y, box_width, box_height = fields
    category_id = int(category) + 1
    normalized = [float(center_x), float(center_y), float(box_width), float(box_height)]
    if width <= 0 or height <= 0 or any(not math.isfinite(value) for value in normalized):
        raise ValueError("invalid image dimensions or box values")
    if any(value < 0 or value > 1 for value in normalized):
        raise ValueError("YOLO box values must be normalized to [0,1]")
    cx, cy, bw, bh = normalized
    absolute_width = bw * width
    absolute_height = bh * height
    x = (cx - bw / 2.0) * width
    y = (cy - bh / 2.0) * height
    return {
        "category_id": category_id,
        "bbox": [x, y, absolute_width, absolute_height],
        "area": absolute_width * absolute_height,
    }


def build_coco_ground_truth(
    image_sizes: Mapping[str, tuple[int, int]],
    labels_dir: Path,
    class_names: Sequence[str],
    expected_images: int,
    expected_instances: int,
) -> dict[str, object]:
    if len(image_sizes) != expected_images:
        raise ValueError(f"expected {expected_images} images, found {len(image_sizes)}")
    if not class_names:
        raise ValueError("class_names must be non-empty")
    images = []
    annotations = []
    annotation_id = 1
    for image_id, filename in enumerate(sorted(image_sizes), start=1):
        width, height = image_sizes[filename]
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label for {filename}: {label_path}")
        images.append({"id": image_id, "file_name": filename, "width": width, "height": height})
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            converted = coco_bbox_from_yolo_line(line, width=width, height=height)
            if converted["category_id"] > len(class_names):
                raise ValueError(f"category id exceeds class count in {label_path}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    **converted,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    if len(annotations) != expected_instances:
        raise ValueError(f"expected {expected_instances} instances, found {len(annotations)}")
    categories = [{"id": index, "name": name} for index, name in enumerate(class_names, start=1)]
    return {
        "info": {"description": "VisDrone2019-DET validation converted from the frozen YOLO overlay"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _flatten_valid(values: object) -> list[float]:
    if isinstance(values, (list, tuple)):
        flattened = []
        for value in values:
            flattened.extend(_flatten_valid(value))
        return flattened
    numeric = float(values)
    return [numeric] if math.isfinite(numeric) and numeric >= 0 else []


def extract_extended_metrics(
    stats: Sequence[float],
    per_class_precision: Sequence[object],
    class_names: Sequence[str],
    max_dets: Sequence[int],
) -> dict[str, object]:
    if len(stats) < 13 or any(not math.isfinite(float(value)) for value in stats[:13]):
        raise ValueError("COCO statistics must contain thirteen finite values")
    if list(max_dets) != [1, 10, 100, 500]:
        raise ValueError("max_dets must be [1, 10, 100, 500]")
    if len(per_class_precision) != len(class_names):
        raise ValueError("per-class precision and class count differ")
    per_class = {}
    for name, precision in zip(class_names, per_class_precision):
        valid = _flatten_valid(precision)
        if not valid:
            raise ValueError(f"class {name} has no valid precision values")
        per_class[name] = sum(valid) / len(valid)
    return {
        "AP50_95": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APS": float(stats[3]),
        "APM": float(stats[4]),
        "APL": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "ARS": float(stats[9]),
        "ARM": float(stats[10]),
        "ARL": float(stats[11]),
        "AR500": float(stats[12]),
        "per_class_AP": per_class,
    }

