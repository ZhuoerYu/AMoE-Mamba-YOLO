"""Shared-shell DMS-XSS block for the DualRoute-Mamba neck.

The LSBlock → SS2D residual → RGBlock residual ordering and SS2D input/depthwise/
gate/output shell are adapted from the pinned Mamba-YOLO XSSBlock (AGPL-3.0,
commit b26cbda230dfa217f96faee8dc7020db3962f3df). Only the SS2D recurrence pool
and sparse per-image router are replaced as specified by the pilot design.
"""

from __future__ import annotations

import torch
from torch import nn

from .drmamba_cuda import DirectionalSelectiveScanBackend
from .drmamba_experts import build_expert_pool
from .drmamba_router import AxisAwareTopKRouter, GAPTopKRouter, GlobalTopKRouter, RouterOutput


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LSBlock(nn.Module):
    """Local spatial block retained from Mamba-YOLO."""

    def __init__(self, channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.fc2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.fc3 = nn.Conv2d(channels, channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc3(self.act(self.fc2(self.norm(self.fc1(x))))))


class RGBlock(nn.Module):
    """Residual gated MLP retained from Mamba-YOLO."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(2 * channels * mlp_ratio / 3))
        self.fc1 = nn.Conv2d(channels, 2 * hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, value = self.fc1(x).chunk(2, dim=1)
        content = self.act(self.dwconv(content) + content) * value
        return self.drop(self.fc2(self.drop(content)))


class DMSSelectiveShell(nn.Module):
    """One official SS2D shell around four independent, truly sparse expert cores."""

    def __init__(
        self,
        channels: int,
        *,
        top_k: int = 2,
        router: str = "axis",
        detach_router: bool = True,
        expert_pool: str = "specialized",
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        scan_backend: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.inner = int(channels * ssm_ratio)
        self.in_proj = nn.Conv2d(channels, 2 * self.inner, 1, bias=False)
        self.conv2d = nn.Conv2d(self.inner, self.inner, 3, padding=1, groups=self.inner)
        self.act = nn.GELU()
        self.expert_pool = expert_pool
        self.experts = build_expert_pool(expert_pool, self.inner, dt_rank=dt_rank)
        expert_count = len(self.experts)
        if expert_pool == "single":
            if router != "none" or top_k != 1:
                raise ValueError("single expert pool requires router='none' and top_k=1")
            self.router = None
        elif router == "axis":
            self.router = AxisAwareTopKRouter(
                channels, experts=expert_count, top_k=top_k, detach_descriptor=detach_router
            )
        elif router == "gap":
            self.router = GAPTopKRouter(channels, experts=expert_count, top_k=top_k)
        elif router == "global_route":
            self.router = GlobalTopKRouter(channels, experts=expert_count, top_k=top_k)
        else:
            raise ValueError("router must be 'axis', 'gap', or 'global_route'")
        self.scan_backend = scan_backend if scan_backend is not None else DirectionalSelectiveScanBackend()
        self.out_norm = nn.LayerNorm(self.inner)
        self.out_proj = nn.Conv2d(self.inner, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_routing: dict[str, object] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.in_proj(x).chunk(2, dim=1)
        content = self.act(self.conv2d(content))
        gate = self.act(gate)
        if self.router is None:
            batch = x.shape[0]
            route = RouterOutput(
                logits=content.new_zeros((batch, 1)),
                weights=content.new_ones((batch, 1)),
                selected_indices=torch.zeros((batch, 1), dtype=torch.long, device=content.device),
                entropy=content.new_zeros((batch,)),
            )
        else:
            route = self.router(x)
        mixed = torch.zeros_like(content)
        expert_calls = []
        for expert_index, expert in enumerate(self.experts):
            row_indices = torch.nonzero((route.selected_indices == expert_index).any(dim=1), as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            expert_calls.append(expert_index)
            expert_output = expert(content.index_select(0, row_indices), self.scan_backend).to(mixed.dtype)
            coefficient = route.weights.index_select(0, row_indices)[:, expert_index, None, None, None].to(mixed.dtype)
            mixed.index_add_(0, row_indices, coefficient * expert_output)

        normalized = self.out_norm(mixed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        output = self.dropout(self.out_proj(normalized * gate))
        self.last_routing = {
            "weights": route.weights.detach(),
            "selected_indices": route.selected_indices.detach(),
            "entropy": route.entropy.detach(),
            "expert_calls": tuple(expert_calls),
            "real_selective_scan": bool(getattr(self.scan_backend, "real_selective_scan", False)),
            "scan_backend": getattr(self.scan_backend, "backend_name", type(self.scan_backend).__name__),
            # paper-v1-r5 checkpoints predate the ablation metadata field; their
            # serialized expert modules are the original specialized pool.
            "expert_pool": getattr(self, "expert_pool", "specialized"),
            "router": "none" if self.router is None else type(self.router).__name__,
        }
        return output


class DMSXSSBlock(nn.Module):
    """Mamba-YOLO XSS shell with a four-expert DMS selective-scan core."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: tuple[int, int, int, int] = (16, 16, 32, 8),
        top_k: int = 2,
        router: str = "axis",
        detach_router: bool = True,
        expert_pool: str = "specialized",
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        scan_backend: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if tuple(d_state) != (16, 16, 32, 8):
            raise ValueError("pilot d_state must remain (16, 16, 32, 8)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.lsblock = LSBlock(out_channels, dropout)
        self.norm = LayerNorm2d(out_channels)
        self.ss2d = DMSSelectiveShell(
            out_channels,
            top_k=top_k,
            router=router,
            detach_router=detach_router,
            expert_pool=expert_pool,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            scan_backend=scan_backend,
            dropout=dropout,
        )
        self.norm2 = LayerNorm2d(out_channels)
        self.mlp = RGBlock(out_channels, mlp_ratio, dropout)

    @property
    def last_routing(self) -> dict[str, object]:
        return self.ss2d.last_routing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        local = self.lsblock(x)
        x = x + self.ss2d(self.norm(local))
        return x + self.mlp(self.norm2(x))


__all__ = ("DMSSelectiveShell", "DMSXSSBlock", "LSBlock", "LayerNorm2d", "RGBlock")
