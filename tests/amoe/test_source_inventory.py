from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_runtime_files_are_present():
    required = [
        "ultralytics/nn/modules/drmamba_block.py",
        "ultralytics/nn/modules/drmamba_router.py",
        "ultralytics/nn/modules/drmamba_experts.py",
        "ultralytics/nn/modules/drmamba_scan.py",
        "ultralytics/nn/modules/drmamba_cuda.py",
        "selective_scan/setup.py",
    ]

    missing = [path for path in required if not (ROOT / path).is_file()]

    assert missing == []


def test_public_runtime_contains_upstream_license_files():
    required = [
        "LICENSE",
        "LICENSES/Mamba-YOLO.LICENSE",
        "LICENSES/YOLO-Master.LICENSE",
    ]

    missing = [path for path in required if not (ROOT / path).is_file()]

    assert missing == []
