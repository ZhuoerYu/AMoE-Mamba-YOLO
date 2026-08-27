from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_MODEL_ROOT = ROOT / "ultralytics/cfg/models/amoe"
FROZEN_MODEL_ROOT = ROOT / "ultralytics/cfg/models/drmamba"

MODEL_PAIRS = {
    "baseline-conv-yolo-n.yaml": "drm-yolo-m00.yaml",
    "amoe-mamba-yolo-n.yaml": "drm-yolo-m01.yaml",
    "backbone-moe-yolo-n.yaml": "drm-yolo-m10.yaml",
    "backbone-moe-amoe-yolo-n.yaml": "drm-yolo-m11.yaml",
}

ABLATION_PAIRS = {
    "single-ss2d.yaml": "drm-yolo-m01-single-ss2d.yaml",
    "homogeneous4-axis-top2.yaml": "drm-yolo-m01-homogeneous4-axis-top2.yaml",
    "global-top2.yaml": "drm-yolo-m01-global-top2.yaml",
    "gap-top2.yaml": "drm-yolo-m01-gap-top2.yaml",
    "axis-top1.yaml": "drm-yolo-m01-axis-top1.yaml",
    "axis-top4.yaml": "drm-yolo-m01-axis-top4.yaml",
}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_public_model_graphs_equal_frozen_training_graphs():
    for public_name, frozen_name in MODEL_PAIRS.items():
        assert load_yaml(PUBLIC_MODEL_ROOT / public_name) == load_yaml(FROZEN_MODEL_ROOT / frozen_name)


def test_public_ablation_graphs_equal_frozen_training_graphs():
    for public_name, frozen_name in ABLATION_PAIRS.items():
        assert load_yaml(PUBLIC_MODEL_ROOT / "ablations" / public_name) == load_yaml(FROZEN_MODEL_ROOT / frozen_name)


def test_proposed_config_has_four_amoe_blocks():
    config = load_yaml(PUBLIC_MODEL_ROOT / "amoe-mamba-yolo-n.yaml")

    assert sum(layer[2] == "DMSXSSBlock" for layer in config["head"]) == 4


def test_public_manifests_pin_the_paper_protocol_and_three_seeds():
    main = load_yaml(ROOT / "configs/main_experiments.yaml")
    ablations = load_yaml(ROOT / "configs/ablations.yaml")

    assert main["fixed"] == {
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "optimizer": "AdamW",
        "lr0": 0.000714,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nbs": 64,
        "amp": True,
        "deterministic": True,
        "pretrained": False,
        "patience": 0,
        "workers": 8,
    }
    assert main["seeds"] == [20260821, 20260822, 20260823]
    assert len(main["models"]) == 4
    assert len(ablations["models"]) == 6


def test_public_configs_have_no_machine_specific_paths():
    paths = [
        *PUBLIC_MODEL_ROOT.rglob("*.yaml"),
        *(ROOT / "configs").glob("*.yaml"),
    ]
    prohibited = (
        "/" + "data/",
        "/" + "Users/",
        "5090" + "ubuntu",
        "5090" + "-ubuntu",
        "system" + "ctl",
    )

    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in prohibited), path
