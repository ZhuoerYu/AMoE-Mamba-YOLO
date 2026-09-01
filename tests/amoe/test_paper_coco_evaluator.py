import csv
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_evaluator_dry_run_works_from_a_fresh_clone_without_editable_install():
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--weights", "weights/model.pt", "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"evaluator": "pycocotools.COCOeval"' in completed.stdout


def test_evaluation_defaults_match_the_frozen_paper_protocol():
    evaluate = importlib.import_module("scripts.evaluate")
    args = evaluate.parse_args(["--weights", "weights/model.pt", "--dry-run"])

    protocol = evaluate.build_evaluation_config(args)

    assert protocol["evaluator"] == "pycocotools.COCOeval"
    assert protocol["images"] == 548
    assert protocol["instances"] == 38759
    assert protocol["confidence"] == 0.001
    assert protocol["nms_iou"] == 0.7
    assert protocol["max_detections"] == [1, 10, 100, 500]
    assert protocol["image_size"] == 640


def test_yolo_annotations_are_converted_to_coco_xywh():
    visdrone = importlib.import_module("paper_eval.visdrone")

    annotation = visdrone.coco_bbox_from_yolo_line(
        "2 0.5 0.5 0.2 0.4", width=100, height=50
    )

    assert annotation["category_id"] == 3
    assert annotation["bbox"] == [40.0, 15.0, 20.0, 20.0]
    assert annotation["area"] == 400.0


def test_coco_statistics_use_the_500_detection_slot():
    evaluator = importlib.import_module("paper_eval.evaluator")
    precision = np.full((2, 2, 1, 4, 4), -1.0)
    recall = np.full((2, 1, 4, 4), -1.0)
    precision[:, :, :, 0, 3] = 0.25
    precision[0, :, :, 0, 3] = 0.50
    precision[1, :, :, 0, 3] = 0.75
    precision[:, :, :, 1, 3] = 0.10
    precision[:, :, :, 2, 3] = 0.20
    precision[:, :, :, 3, 3] = 0.30
    recall[:, :, 0, 0] = 0.01
    recall[:, :, 0, 1] = 0.10
    recall[:, :, 0, 2] = 0.20
    recall[:, :, 0, 3] = 0.40
    recall[:, :, 1, 3] = 0.30
    recall[:, :, 2, 3] = 0.50
    recall[:, :, 3, 3] = 0.60
    fake = type(
        "FakeEvaluation",
        (),
        {
            "eval": {"precision": precision, "recall": recall},
            "params": type(
                "Params",
                (),
                {
                    "areaRngLbl": ["all", "small", "medium", "large"],
                    "maxDets": [1, 10, 100, 500],
                    "iouThrs": np.array([0.50, 0.75]),
                },
            )(),
        },
    )()

    stats, per_class = evaluator.coco_statistics(fake)

    assert stats == [0.625, 0.5, 0.75, 0.1, 0.2, 0.3, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.4]
    assert per_class == [[[0.5, 0.5], [0.75, 0.75]]]


def test_legacy_checkpoint_metadata_repair_does_not_replace_experts():
    evaluator = importlib.import_module("paper_eval.evaluator")
    LegacyShell = type("DMSSelectiveShell", (), {})
    shell = LegacyShell()
    shell.experts = [object(), object(), object(), object()]
    identities = [id(expert) for expert in shell.experts]

    model = type("Model", (), {"named_modules": lambda self: [("model.13.ss2d", shell)]})()
    detector = type("Detector", (), {"model": model})()

    repaired = evaluator.repair_legacy_shell_metadata(detector)

    assert repaired == ["model.13.ss2d"]
    assert shell.expert_pool == "specialized"
    assert [id(expert) for expert in shell.experts] == identities


def test_published_coco_results_match_the_manuscript_contract():
    path = ROOT / "results" / "paper_coco_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["ID"]: row for row in csv.DictReader(handle)}

    top2 = rows["s0-m01"]
    top4 = rows["ablation-m01-axis-top4-s0"]

    assert int(top2["Images"]) == 548
    assert float(top2["AP50_95_%"]) == 14.176
    assert float(top2["AP50_%"]) == 25.977
    assert float(top2["AP75_%"]) == 13.516
    assert float(top2["AR500_%"]) == 29.820
    assert float(top4["AP50_95_%"]) == 14.228
