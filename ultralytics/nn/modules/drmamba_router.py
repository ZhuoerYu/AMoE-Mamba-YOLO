"""Per-image sparse routers for the DMS-Mamba neck expert pool."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class RouterOutput(NamedTuple):
    logits: torch.Tensor
    weights: torch.Tensor
    selected_indices: torch.Tensor
    entropy: torch.Tensor


class _TopKRouterBase(nn.Module):
    def __init__(self, dim: int, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if not 1 <= top_k <= experts:
            raise ValueError(f"top_k must be in [1, {experts}]")
        self.dim = dim
        self.num_experts = experts
        self.top_k = top_k

    def route_logits(self, logits: torch.Tensor) -> RouterOutput:
        if logits.ndim != 2 or logits.shape[1] != self.num_experts:
            raise ValueError(f"expected [B,{self.num_experts}] logits, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("router logits contain NaN or Inf")
        selected_logits, selected_indices = torch.topk(logits.float(), self.top_k, dim=1)
        selected_weights = torch.softmax(selected_logits, dim=1).to(logits.dtype)
        weights = torch.zeros_like(logits).scatter(1, selected_indices, selected_weights)
        entropy = -(selected_weights.float() * selected_weights.float().clamp_min(1e-12).log()).sum(dim=1)
        return RouterOutput(logits, weights, selected_indices, entropy)


class GAPTopKRouter(_TopKRouterBase):
    """YOLO-Master-style global-average-pooling gate with exact sparse Top-K."""

    def __init__(self, dim: int, experts: int = 4, top_k: int = 2, reduction: int = 8) -> None:
        super().__init__(dim, experts, top_k)
        hidden = max(dim // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, experts))

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"expected Bx{self.dim}xHxW, got {tuple(x.shape)}")
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        return self.route_logits(self.classifier(self.describe(x)))


class AxisAwareTopKRouter(_TopKRouterBase):
    """Coordinate-pooling descriptor used only for routing, never feature modulation."""

    def __init__(
        self,
        dim: int,
        experts: int = 4,
        top_k: int = 2,
        reduction: int = 8,
        detach_descriptor: bool = True,
    ) -> None:
        super().__init__(dim, experts, top_k)
        hidden = max(dim // reduction, 8)
        self.detach_descriptor = detach_descriptor
        self.classifier = nn.Sequential(nn.Linear(4 * dim, hidden), nn.SiLU(), nn.Linear(hidden, experts))

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"expected Bx{self.dim}xHxW, got {tuple(x.shape)}")
        source = x.detach() if self.detach_descriptor else x
        pooled_h = source.mean(dim=3)
        pooled_w = source.mean(dim=2)
        return torch.cat(
            (
                pooled_h.mean(dim=2),
                pooled_h.amax(dim=2),
                pooled_w.mean(dim=2),
                pooled_w.amax(dim=2),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> RouterOutput:
        return self.route_logits(self.classifier(self.describe(x)))


class GlobalTopKRouter(_TopKRouterBase):
    """Input-independent learned Top-K gate used as the non-adaptive routing control."""

    def __init__(self, dim: int, experts: int = 4, top_k: int = 2) -> None:
        super().__init__(dim, experts, top_k)
        self.global_logits = nn.Parameter(torch.empty(experts))
        nn.init.normal_(self.global_logits, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"expected Bx{self.dim}xHxW, got {tuple(x.shape)}")
        return self.route_logits(self.global_logits.unsqueeze(0).expand(x.shape[0], -1))


__all__ = ("AxisAwareTopKRouter", "GAPTopKRouter", "GlobalTopKRouter", "RouterOutput")
