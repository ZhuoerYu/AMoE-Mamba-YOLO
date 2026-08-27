"""Direction/capacity-specialized SS2D expert cores.

Parameter layout and initialization follow the pinned Mamba-YOLO ``SS2D`` core.
The four-expert organization is the frozen pilot design; the input/output shell
is deliberately kept outside this module so it can be shared once per neck block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .drmamba_scan import ALL_DIRECTIONS, BT, LR, RL, TB, cross_scan_2d, merge_directions


@dataclass(frozen=True)
class DMSExpertSpec:
    name: str
    directions: tuple[int, ...]
    d_state: int


DEFAULT_EXPERT_SPECS = (
    DMSExpertSpec("horizontal", (LR, RL), 16),
    DMSExpertSpec("vertical", (TB, BT), 16),
    DMSExpertSpec("large", ALL_DIRECTIONS, 32),
    DMSExpertSpec("lite", ALL_DIRECTIONS, 8),
)
HOMOGENEOUS_EXPERT_SPECS = tuple(DMSExpertSpec(f"homogeneous_{index}", ALL_DIRECTIONS, 16) for index in range(4))
SINGLE_SS2D_EXPERT_SPECS = (DMSExpertSpec("single_ss2d", ALL_DIRECTIONS, 16),)


class TorchReferenceDirectionalScan(nn.Module):
    """Slow differentiable CPU recurrence used only for semantic unit tests."""

    backend_name = "torch_reference_directional_scan"
    real_selective_scan = False

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        state_c: torch.Tensor,
        skip_d: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (u, delta, state_a, state_b, state_c, skip_d)
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise RuntimeError("the reference recurrence is CPU-only; CUDA must use the verified extension")
        batch, directions, channels, length = u.shape
        state_size = state_a.shape[-1]
        if delta.shape != u.shape:
            raise ValueError("delta must match u")
        if state_a.shape != (directions, channels, state_size):
            raise ValueError("state_a has an incompatible shape")
        if state_b.shape != (batch, directions, state_size, length):
            raise ValueError("state_b has an incompatible shape")
        if state_c.shape != (batch, directions, state_size, length):
            raise ValueError("state_c has an incompatible shape")
        if skip_d.shape != (directions, channels):
            raise ValueError("skip_d has an incompatible shape")

        state = u.new_zeros((batch, directions, channels, state_size))
        outputs = []
        positive_delta = F.softplus(delta)
        for step in range(length):
            dt = positive_delta[..., step]
            transition = torch.exp(dt[..., None] * state_a[None])
            drive = dt[..., None] * state_b[:, :, None, :, step] * u[..., step, None]
            state = transition * state + drive
            output = (state * state_c[:, :, None, :, step]).sum(dim=-1)
            outputs.append(output + skip_d[None] * u[..., step])
        return torch.stack(outputs, dim=-1)


class DirectionalSS2DExpert(nn.Module):
    """Independent official-layout recurrence parameters for selected 2D directions."""

    def __init__(self, dim: int, spec: DMSExpertSpec, dt_rank: int | str = "auto") -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if not spec.directions or any(direction not in ALL_DIRECTIONS for direction in spec.directions):
            raise ValueError(f"invalid directions for {spec.name}: {spec.directions}")
        self.dim = dim
        self.spec = spec
        self.d_state = spec.d_state
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else int(dt_rank)
        self.k = len(spec.directions)
        projection_width = self.dt_rank + 2 * self.d_state

        self.x_proj_weight = nn.Parameter(torch.empty(self.k, projection_width, dim))
        self.dt_projs_weight = nn.Parameter(torch.empty(self.k, dim, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(self.k, dim))
        self.A_logs = nn.Parameter(torch.empty(self.k * dim, self.d_state))
        self.Ds = nn.Parameter(torch.ones(self.k * dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        nn.init.uniform_(self.dt_projs_bias, -0.1, 0.1)
        base = torch.arange(1, self.d_state + 1, dtype=self.A_logs.dtype).log()
        with torch.no_grad():
            self.A_logs.copy_(base.repeat(self.k * self.dim, 1))

    def forward(self, x: torch.Tensor, scan_backend: nn.Module) -> torch.Tensor:
        """Run this expert through an explicitly supplied selective-scan backend."""
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"{self.spec.name} expected Bx{self.dim}xHxW, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        sequences = cross_scan_2d(x)[:, list(self.spec.directions)]
        projected = torch.einsum("bkcl,kqc->bkql", sequences, self.x_proj_weight)
        delta_low_rank, state_b, state_c = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        delta = torch.einsum("bkrl,kcr->bkcl", delta_low_rank, self.dt_projs_weight)
        delta = delta + self.dt_projs_bias[None, :, :, None]
        state_a = -self.A_logs.float().exp().reshape(self.k, self.dim, self.d_state)
        skip_d = self.Ds.float().reshape(self.k, self.dim)
        scanned = scan_backend(sequences, delta, state_a, state_b, state_c, skip_d)
        return merge_directions(scanned, self.spec.directions, height, width)


def build_default_experts(dim: int, **official_ss2d_args) -> nn.ModuleList:
    """Instantiate the frozen horizontal, vertical, large-state, and lite pool."""
    return build_expert_pool("specialized", dim, **official_ss2d_args)


def build_expert_pool(pool: str, dim: int, **official_ss2d_args) -> nn.ModuleList:
    """Build one of the paper's controlled SS2D expert pools with an unchanged core implementation."""
    supported = {"dt_rank"}
    unknown = set(official_ss2d_args) - supported
    if unknown:
        raise TypeError(f"unsupported SS2D expert arguments: {sorted(unknown)}")
    specs = {
        "specialized": DEFAULT_EXPERT_SPECS,
        "homogeneous": HOMOGENEOUS_EXPERT_SPECS,
        "single": SINGLE_SS2D_EXPERT_SPECS,
    }.get(pool)
    if specs is None:
        raise ValueError("expert pool must be 'specialized', 'homogeneous', or 'single'")
    return nn.ModuleList(DirectionalSS2DExpert(dim=dim, spec=spec, **official_ss2d_args) for spec in specs)


__all__ = (
    "DEFAULT_EXPERT_SPECS",
    "HOMOGENEOUS_EXPERT_SPECS",
    "SINGLE_SS2D_EXPERT_SPECS",
    "DMSExpertSpec",
    "DirectionalSS2DExpert",
    "TorchReferenceDirectionalScan",
    "build_default_experts",
    "build_expert_pool",
)
