"""Canonical four-direction 2D sequence ordering for DMS-Mamba.

The ordering is adapted from ``CrossScan``/``CrossMerge`` in the pinned
HZAI-ZJNU/Mamba-YOLO implementation (AGPL-3.0, commit b26cbda230dfa217f96faee8dc7020db3962f3df).
This module only makes the four official directions individually addressable;
it does not introduce a new scan kernel.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


LR = 0
TB = 1
RL = 2
BT = 3
ALL_DIRECTIONS = (LR, TB, RL, BT)


def cross_scan_2d(x: torch.Tensor) -> torch.Tensor:
    """Return left/right and top/bottom sequence views as ``[B, 4, C, H*W]``."""
    if x.ndim != 4:
        raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
    b, c, h, w = x.shape
    sequences = x.new_empty((b, 4, c, h * w))
    sequences[:, LR] = x.flatten(2, 3)
    sequences[:, TB] = x.transpose(2, 3).flatten(2, 3)
    sequences[:, RL : BT + 1] = torch.flip(sequences[:, LR : TB + 1], dims=[-1])
    return sequences


def merge_directions(
    y: torch.Tensor,
    directions: Iterable[int],
    h: int,
    w: int,
) -> torch.Tensor:
    """Align selected directional outputs, average them, and restore BCHW shape."""
    directions = tuple(directions)
    if not directions:
        raise ValueError("at least one direction is required")
    if y.ndim != 4 or y.shape[1] != len(directions) or y.shape[-1] != h * w:
        raise ValueError(
            f"expected [B,{len(directions)},C,{h * w}] directional output, got {tuple(y.shape)}"
        )

    aligned = []
    for index, direction in enumerate(directions):
        sequence = y[:, index]
        if direction in (RL, BT):
            sequence = sequence.flip(dims=[-1])
        if direction in (TB, BT):
            sequence = sequence.view(y.shape[0], y.shape[2], w, h).transpose(2, 3).contiguous()
        else:
            sequence = sequence.view(y.shape[0], y.shape[2], h, w)
        aligned.append(sequence)
    return torch.stack(aligned, dim=0).mean(dim=0)


__all__ = ("LR", "TB", "RL", "BT", "ALL_DIRECTIONS", "cross_scan_2d", "merge_directions")
