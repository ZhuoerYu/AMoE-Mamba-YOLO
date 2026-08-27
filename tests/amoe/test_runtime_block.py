import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

MODULE_ROOT = Path(__file__).resolve().parents[2] / "ultralytics/nn/modules"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("ultralytics", "ultralytics.nn", "ultralytics.nn.modules"):
    sys.modules.setdefault(package_name, types.ModuleType(package_name))
_load_module("ultralytics.nn.modules.drmamba_scan", "drmamba_scan.py")
_load_module("ultralytics.nn.modules.drmamba_experts", "drmamba_experts.py")
_load_module("ultralytics.nn.modules.drmamba_router", "drmamba_router.py")
_load_module("ultralytics.nn.modules.drmamba_cuda", "drmamba_cuda.py")
block_module = _load_module("ultralytics.nn.modules.drmamba_block", "drmamba_block.py")


class IdentityScan(torch.nn.Module):
    real_selective_scan = False
    backend_name = "identity_test_scan"

    def forward(self, u, delta, A, B, C, D):
        return u


@pytest.mark.parametrize(
    ("batch", "channels_in", "channels_out", "height", "width"),
    ((1, 8, 8, 5, 5), (2, 8, 12, 4, 7), (1, 12, 16, 3, 5)),
)
def test_block_preserves_requested_bchw_shape(batch, channels_in, channels_out, height, width):
    block = block_module.DMSXSSBlock(channels_in, channels_out, scan_backend=IdentityScan())
    x = torch.randn(batch, channels_in, height, width)

    output = block(x)

    assert output.shape == (batch, channels_out, height, width)
    assert torch.isfinite(output).all()


def test_hard_top2_executes_only_two_experts_for_one_sample():
    block = block_module.DMSXSSBlock(8, 8, router="gap", scan_backend=IdentityScan())
    with torch.no_grad():
        for parameter in block.ss2d.router.parameters():
            parameter.zero_()
        block.ss2d.router.classifier[-1].bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    calls = [0, 0, 0, 0]
    handles = []
    for index, expert in enumerate(block.ss2d.experts):
        handles.append(
            expert.register_forward_hook(lambda module, args, output, i=index: calls.__setitem__(i, calls[i] + 1))
        )

    try:
        block(torch.randn(1, 8, 4, 5))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [1, 1, 0, 0]
    assert block.last_routing["expert_calls"] == (0, 1)
    assert block.last_routing["real_selective_scan"] is False


def test_reference_block_backward_is_finite():
    block = block_module.DMSXSSBlock(4, 4, ssm_ratio=1.0)
    x = torch.randn(1, 4, 2, 3, requires_grad=True)

    output = block(x)
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    selected = set(block.last_routing["selected_indices"].flatten().tolist())
    for index, expert in enumerate(block.ss2d.experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if index in selected:
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)


def test_legacy_specialized_checkpoint_without_expert_pool_metadata_still_runs():
    """The paper-v1-r5 checkpoint predates the ablation metadata field."""
    block = block_module.DMSXSSBlock(4, 4, ssm_ratio=1.0, scan_backend=IdentityScan())
    del block.ss2d.expert_pool

    output = block(torch.randn(1, 4, 2, 3))

    assert torch.isfinite(output).all()
    assert block.last_routing["expert_pool"] == "specialized"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression requires a GPU")
@pytest.mark.parametrize(
    ("router", "expert_pool", "top_k"),
    (
        ("none", "single", 1),
        ("axis", "homogeneous", 2),
        ("global_route", "specialized", 2),
        ("gap", "specialized", 2),
        ("axis", "specialized", 1),
        ("axis", "specialized", 4),
    ),
)
def test_all_ablation_blocks_backward_are_finite_under_cuda_amp(router, expert_pool, top_k):
    """Every control route must honor the expert-mixture AMP dtype contract."""
    block = block_module.DMSXSSBlock(
        4,
        4,
        top_k=top_k,
        router=router,
        expert_pool=expert_pool,
        ssm_ratio=1.0,
        scan_backend=IdentityScan(),
    ).cuda()
    x = torch.randn(2, 4, 2, 3, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = block(x)
        loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
