import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "ultralytics/nn/modules/drmamba_scan.py"
SPEC = importlib.util.spec_from_file_location("drmamba_scan_under_test", MODULE_PATH)
drmamba_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drmamba_scan)


def test_cross_scan_direction_order():
    x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    seq = drmamba_scan.cross_scan_2d(x)[0, :, 0]

    assert seq[drmamba_scan.LR].tolist() == [1, 2, 3, 4, 5, 6]
    assert seq[drmamba_scan.TB].tolist() == [1, 4, 2, 5, 3, 6]
    assert seq[drmamba_scan.RL].tolist() == [6, 5, 4, 3, 2, 1]
    assert seq[drmamba_scan.BT].tolist() == [6, 3, 5, 2, 4, 1]


def test_identity_direction_outputs_merge_to_original():
    x = torch.arange(1.0, 13.0).view(1, 2, 2, 3)
    sequences = drmamba_scan.cross_scan_2d(x)

    for directions in (
        drmamba_scan.ALL_DIRECTIONS,
        (drmamba_scan.LR, drmamba_scan.RL),
        (drmamba_scan.TB, drmamba_scan.BT),
    ):
        selected = sequences[:, list(directions)]
        restored = drmamba_scan.merge_directions(selected, directions, h=2, w=3)
        torch.testing.assert_close(restored, x)


def test_rectangular_non_contiguous_input_has_finite_gradient():
    base = torch.randn(2, 3, 4, 7, requires_grad=True)
    x = base.transpose(2, 3)
    assert not x.is_contiguous()

    sequences = drmamba_scan.cross_scan_2d(x)
    restored = drmamba_scan.merge_directions(
        sequences, drmamba_scan.ALL_DIRECTIONS, h=x.shape[2], w=x.shape[3]
    )
    restored.square().mean().backward()

    torch.testing.assert_close(restored, x)
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()
