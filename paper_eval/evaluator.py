"""Standalone inference and pycocotools evaluation used for the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from .visdrone import build_coco_ground_truth, extract_extended_metrics


PAPER_PROTOCOL = {
    "evaluator": "pycocotools.COCOeval",
    "images": 548,
    "instances": 38759,
    "confidence": 0.001,
    "nms_iou": 0.7,
    "max_detections": [1, 10, 100, 500],
    "image_size": 640,
}


def repair_legacy_shell_metadata(detector: object) -> list[str]:
    """Restore metadata absent from frozen checkpoints without changing tensors."""
    repaired = []
    for name, module in detector.model.named_modules():
        if module.__class__.__name__ != "DMSSelectiveShell" or hasattr(module, "expert_pool"):
            continue
        experts = getattr(module, "experts", None)
        module.expert_pool = "single" if experts is not None and len(experts) == 1 else "specialized"
        repaired.append(name)
    return repaired


def _mean_valid(values: object) -> float:
    array = np.asarray(values)
    valid = array[array > -1]
    return float(valid.mean()) if valid.size else -1.0


def coco_statistics(evaluation: object) -> tuple[list[float], list[object]]:
    """Extract the paper's AP/AR vector from a completed COCOeval object."""
    precision = evaluation.eval["precision"]
    recall = evaluation.eval["recall"]
    params = evaluation.params
    area_indices = {label: index for index, label in enumerate(params.areaRngLbl)}
    max_det_indices = {value: index for index, value in enumerate(params.maxDets)}

    def ap(iou: float | None, area: str, max_det: int) -> float:
        values = precision
        if iou is not None:
            indices = np.where(np.isclose(params.iouThrs, iou))[0]
            values = values[indices]
        return _mean_valid(values[:, :, :, area_indices[area], max_det_indices[max_det]])

    def ar(area: str, max_det: int) -> float:
        return _mean_valid(recall[:, :, area_indices[area], max_det_indices[max_det]])

    stats = [
        ap(None, "all", 500),
        ap(0.50, "all", 500),
        ap(0.75, "all", 500),
        ap(None, "small", 500),
        ap(None, "medium", 500),
        ap(None, "large", 500),
        ar("all", 1),
        ar("all", 10),
        ar("all", 100),
        ar("small", 500),
        ar("medium", 500),
        ar("large", 500),
        ar("all", 500),
    ]
    max_index = max_det_indices[500]
    all_area = area_indices["all"]
    per_class = [precision[:, :, category, all_area, max_index].tolist() for category in range(precision.shape[2])]
    return stats, per_class


def evaluate_detections(
    *, ground_truth_path: Path, detections: Sequence[Mapping[str, object]], class_names: Sequence[str]
) -> dict[str, object]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(ground_truth_path))
    detections_api = ground_truth.loadRes(list(detections))
    evaluation = COCOeval(ground_truth, detections_api, "bbox")
    evaluation.params.maxDets = list(PAPER_PROTOCOL["max_detections"])
    evaluation.evaluate()
    evaluation.accumulate()
    stats, per_class = coco_statistics(evaluation)
    return extract_extended_metrics(
        stats=stats,
        per_class_precision=per_class,
        class_names=class_names,
        max_dets=PAPER_PROTOCOL["max_detections"],
    )


def _resolve_dataset(data_path: Path) -> tuple[Path, Path, list[str]]:
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(payload["path"]).expanduser()
    if not root.is_absolute():
        working_candidate = (Path.cwd() / root).resolve()
        config_candidate = (data_path.parent / root).resolve()
        root = working_candidate if working_candidate.exists() else config_candidate
    images_dir = (root / payload["val"]).resolve()
    val_parts = list(Path(payload["val"]).parts)
    if "images" not in val_parts:
        raise ValueError("validation path must contain an images directory")
    val_parts[val_parts.index("images")] = "labels"
    labels_dir = (root / Path(*val_parts)).resolve()
    names = payload["names"]
    if isinstance(names, Mapping):
        class_names = [str(names[index]) for index in sorted(names, key=int)]
    else:
        class_names = [str(name) for name in names]
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"expected validation images at {images_dir} and labels at {labels_dir}")
    return images_dir, labels_dir, class_names


def _build_ground_truth(images_dir: Path, labels_dir: Path, class_names: Sequence[str]) -> dict[str, object]:
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    image_sizes = {}
    for path in image_paths:
        with Image.open(path) as image:
            image_sizes[path.name] = image.size
    return build_coco_ground_truth(
        image_sizes=image_sizes,
        labels_dir=labels_dir,
        class_names=class_names,
        expected_images=int(PAPER_PROTOCOL["images"]),
        expected_instances=int(PAPER_PROTOCOL["instances"]),
    )


def run_evaluation(
    *, weights: str, data: Path, output: Path, batch: int, device: str
) -> dict[str, object]:
    """Run frozen inference, convert detections to COCO, and write auditable artifacts."""
    from ultralytics import YOLO

    images_dir, labels_dir, class_names = _resolve_dataset(data)
    ground_truth = _build_ground_truth(images_dir, labels_dir, class_names)
    output.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output / "ground_truth.coco.json"
    ground_truth_path.write_text(json.dumps(ground_truth, ensure_ascii=False), encoding="utf-8")
    image_id_by_name = {image["file_name"]: image["id"] for image in ground_truth["images"]}
    image_paths = sorted(images_dir / name for name in image_id_by_name)

    detector = YOLO(weights, task="detect")
    legacy_metadata_repairs = repair_legacy_shell_metadata(detector)
    detector.fuse()
    detections = []
    for offset in range(0, len(image_paths), batch):
        chunk = image_paths[offset : offset + batch]
        results = detector.predict(
            source=[str(path) for path in chunk],
            imgsz=int(PAPER_PROTOCOL["image_size"]),
            batch=batch,
            device=device,
            half=str(device).lower() != "cpu",
            conf=float(PAPER_PROTOCOL["confidence"]),
            iou=float(PAPER_PROTOCOL["nms_iou"]),
            max_det=max(PAPER_PROTOCOL["max_detections"]),
            verbose=False,
            save=False,
            stream=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"prediction returned {len(results)} results for {len(chunk)} inputs")
        for input_path, result in zip(chunk, results):
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            for coordinates, score, category in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            ):
                x1, y1, x2, y2 = [float(value) for value in coordinates]
                detections.append(
                    {
                        "image_id": image_id_by_name[input_path.name],
                        "category_id": int(category) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    detections_path = output / "detections.coco.json"
    detections_path.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    metrics = evaluate_detections(
        ground_truth_path=ground_truth_path,
        detections=detections,
        class_names=class_names,
    )
    result = {
        "protocol": PAPER_PROTOCOL,
        "weights": weights,
        "data": str(data),
        "images": len(image_paths),
        "detections": len(detections),
        "metrics": metrics,
        "legacy_metadata_repairs": legacy_metadata_repairs,
        "ground_truth": str(ground_truth_path),
        "detections_file": str(detections_path),
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
