import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_manifest_has_three_seed_assets():
    manifest = json.loads((ROOT / "results/checkpoints.json").read_text(encoding="utf-8"))
    checkpoints = manifest["checkpoints"]

    assert [entry["seed"] for entry in checkpoints] == [20260821, 20260822, 20260823]
    assert all(entry["model"] == "AMoE-Mamba-YOLO-N" for entry in checkpoints)
    assert all(entry["validation_images"] == 548 for entry in checkpoints)
    assert all(len(entry["sha256"]) == 64 for entry in checkpoints)


def test_local_release_assets_match_the_manifest_when_present():
    manifest = json.loads((ROOT / "results/checkpoints.json").read_text(encoding="utf-8"))
    asset_root = ROOT / "release-assets/v0.1.0"
    if not asset_root.is_dir():
        return
    for entry in manifest["checkpoints"]:
        asset = asset_root / entry["filename"]
        assert asset.stat().st_size == entry["bytes"]
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == entry["sha256"]
