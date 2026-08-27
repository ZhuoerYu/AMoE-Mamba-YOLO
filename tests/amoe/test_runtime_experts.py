import importlib.util
import sys
import types
from pathlib import Path

import torch


MODULE_ROOT = Path(__file__).resolve().parents[2] / "ultralytics/nn/modules"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("ultralytics", "ultralytics.nn", "ultralytics.nn.modules"):
    sys.modules.setdefault(package_name, types.ModuleType(package_name))
_load_module("ultralytics.nn.modules.drmamba_scan", MODULE_ROOT / "drmamba_scan.py")
experts_module = _load_module(
    "ultralytics.nn.modules.drmamba_experts", MODULE_ROOT / "drmamba_experts.py"
)


def test_default_expert_specs_and_parameters_are_independent():
    experts = experts_module.build_default_experts(dim=8, dt_rank=1)

    assert [expert.spec.name for expert in experts] == ["horizontal", "vertical", "large", "lite"]
    assert [expert.spec.directions for expert in experts] == [(0, 2), (1, 3), (0, 1, 2, 3), (0, 1, 2, 3)]
    assert [expert.spec.d_state for expert in experts] == [16, 16, 32, 8]
    assert all(expert.dim == 8 for expert in experts)

    parameter_ids = [{id(parameter) for parameter in expert.parameters()} for expert in experts]
    for left in range(len(experts)):
        for right in range(left + 1, len(experts)):
            assert parameter_ids[left].isdisjoint(parameter_ids[right])


class IdentityScan(torch.nn.Module):
    def forward(self, u, delta, A, B, C, D):
        return u


def test_direction_count_is_normalized_for_every_expert():
    x = torch.randn(2, 8, 2, 3)

    for expert in experts_module.build_default_experts(dim=8, dt_rank=1):
        output = expert(x, IdentityScan())
        torch.testing.assert_close(output, x)


def test_state_capacity_order_and_finite_independent_gradients():
    torch.manual_seed(31)
    experts = experts_module.build_default_experts(dim=4, dt_rank=1)
    backend = experts_module.TorchReferenceDirectionalScan()
    x = torch.randn(1, 4, 2, 3)

    parameter_counts = [sum(parameter.numel() for parameter in expert.parameters()) for expert in experts]
    assert parameter_counts[2] > parameter_counts[0]
    assert parameter_counts[3] < parameter_counts[2]

    loss = sum(expert(x, backend).square().mean() for expert in experts)
    loss.backward()
    for expert in experts:
        gradients = [parameter.grad for parameter in expert.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
