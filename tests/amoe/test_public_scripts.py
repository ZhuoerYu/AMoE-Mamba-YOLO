import importlib
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    return importlib.import_module(f"scripts.{name}")


def test_visdrone_annotation_is_converted_to_normalized_yolo():
    prepare = load_script("prepare_visdrone")

    line = prepare.convert_annotation("10,20,30,40,1,4,0,0", width=100, height=200)

    assert line == "3 0.250000 0.200000 0.300000 0.200000"


def test_visdrone_ignored_or_invalid_annotations_are_skipped():
    prepare = load_script("prepare_visdrone")

    assert prepare.convert_annotation("10,20,30,40,0,4,0,0", width=100, height=200) is None
    assert prepare.convert_annotation("10,20", width=100, height=200) is None


def test_dataset_output_must_not_equal_source_or_filesystem_root(tmp_path):
    prepare = load_script("prepare_visdrone")
    source = tmp_path / "raw"
    source.mkdir()

    with pytest.raises(ValueError):
        prepare.validate_output_target(source, source)
    with pytest.raises(ValueError):
        prepare.validate_output_target(source, Path("/"))
    with pytest.raises(ValueError):
        prepare.validate_output_target(source, tmp_path)


def test_train_defaults_match_the_frozen_protocol():
    train = load_script("train")
    args = train.parse_args(["--dry-run"])

    kwargs = train.build_train_kwargs(args)

    assert kwargs == {
        "data": "configs/visdrone.yaml",
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
        "seed": 20260821,
        "device": "0",
        "project": "runs/train",
        "name": "amoe-mamba-yolo-n-seed20260821",
        "exist_ok": False,
    }


def test_reproduction_manifest_expands_models_and_seeds():
    reproduce = load_script("reproduce_visdrone")

    main_tasks = reproduce.load_tasks(ROOT / "configs/main_experiments.yaml")
    ablation_tasks = reproduce.load_tasks(ROOT / "configs/ablations.yaml")

    assert len(main_tasks) == 12
    assert len(ablation_tasks) == 6
    assert main_tasks[0]["seed"] == 20260821
    assert main_tasks[-1]["seed"] == 20260823
    assert all(not Path(task["model"]).is_absolute() for task in main_tasks + ablation_tasks)


@pytest.mark.parametrize("name", ["train", "evaluate", "predict", "benchmark", "reproduce_visdrone"])
def test_public_command_parsers_accept_dry_run(name):
    module = load_script(name)
    argv = ["--dry-run"]
    if name in {"evaluate", "predict", "benchmark"}:
        argv += ["--weights", "weights/model.pt"]
    if name == "predict":
        argv += ["--source", "assets/example.jpg"]

    args = module.parse_args(argv)

    assert isinstance(args, Namespace)
    assert args.dry_run is True
