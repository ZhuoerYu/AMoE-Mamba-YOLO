"""Fail-closed adapter for the official Mamba-YOLO selective-scan CUDA core."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .drmamba_experts import TorchReferenceDirectionalScan


class CUDABackendUnavailable(RuntimeError):
    """The pinned official CUDA extension cannot be loaded."""


class CUDAVerificationError(RuntimeError):
    """The CUDA implementation has not passed the current-process gate."""


class _OfficialSelectiveScan(torch.autograd.Function):
    extension: Any = None

    @staticmethod
    def forward(ctx, u, delta, state_a, state_b, state_c, skip_d):
        if _OfficialSelectiveScan.extension is None:
            raise CUDABackendUnavailable("selective_scan_cuda_core is not loaded")
        u = u.contiguous()
        delta = delta.contiguous()
        state_b = state_b.contiguous()
        state_c = state_c.contiguous()
        skip_d = skip_d.contiguous()
        result = _OfficialSelectiveScan.extension.fwd(
            u, delta, state_a, state_b, state_c, skip_d, None, True, 1
        )
        output, scan_state, *_ = result
        ctx.save_for_backward(u, delta, state_a, state_b, state_c, skip_d, scan_state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u, delta, state_a, state_b, state_c, skip_d, scan_state = ctx.saved_tensors
        gradients = _OfficialSelectiveScan.extension.bwd(
            u,
            delta,
            state_a,
            state_b,
            state_c,
            skip_d,
            None,
            grad_output.contiguous(),
            scan_state,
            True,
            1,
        )
        grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d, *_ = gradients
        return grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d


class OfficialDirectionalSelectiveScanCUDA(nn.Module):
    """Directional wrapper around Mamba-YOLO's official selective_scan_cuda_core."""

    backend_name = "official_mamba_yolo_selective_scan_cuda_core"
    real_selective_scan = True

    def __init__(self, import_module: Callable[[str], Any] = importlib.import_module) -> None:
        super().__init__()
        self._import_module = import_module
        self._extension = None
        self.formal_cuda_verified = False
        self.extension_sha256: str | None = None
        self.call_count = 0
        self.direction_sequence_count = 0
        self._self_test_active = False

    def load(self) -> "OfficialDirectionalSelectiveScanCUDA":
        if self._extension is not None:
            return self
        try:
            extension = self._import_module("selective_scan_cuda_core")
        except ImportError as error:
            raise CUDABackendUnavailable("selective_scan_cuda_core is unavailable") from error
        extension_path = Path(getattr(extension, "__file__", ""))
        if not extension_path.is_file():
            raise CUDABackendUnavailable("selective_scan_cuda_core has no loadable shared object")
        digest = hashlib.sha256()
        with extension_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.extension_sha256 = digest.hexdigest()
        self._extension = extension
        _OfficialSelectiveScan.extension = extension
        return self

    def __getstate__(self) -> dict[str, object]:
        """Exclude the imported extension module from EMA/checkpoint deep copies."""
        state = super().__getstate__()
        state["_extension"] = None
        state["_self_test_active"] = False
        return state

    @property
    def extension_path(self) -> Path:
        self.load()
        return Path(self._extension.__file__).resolve(strict=True)

    def forward(self, u, delta, state_a, state_b, state_c, skip_d):
        if u.device.type != "cuda":
            raise CUDAVerificationError("official selective scan requires CUDA tensors")
        if not self.formal_cuda_verified and not self._self_test_active:
            raise CUDAVerificationError("official CUDA backend has not passed verify_cuda()")
        self.load()
        self.call_count += 1
        self.direction_sequence_count += int(u.shape[1])
        original_dtype = u.dtype
        if self.training:
            u, delta, state_b, state_c = (tensor.float() for tensor in (u, delta, state_b, state_c))
        batch, directions, channels, length = u.shape
        output = _OfficialSelectiveScan.apply(
            u.reshape(batch, directions * channels, length),
            delta.reshape(batch, directions * channels, length),
            state_a.reshape(directions * channels, state_a.shape[-1]).float(),
            state_b,
            state_c,
            skip_d.reshape(directions * channels).float(),
        )
        return output.reshape(batch, directions, channels, length).to(original_dtype)

    def verify_cuda(self, device: torch.device | str = "cuda:0") -> dict[str, object]:
        """Run FP32 parity plus FP16 finite-gradient gates, then enable this instance."""
        if not torch.cuda.is_available():
            raise CUDAVerificationError("CUDA is unavailable")
        self.load()
        device = torch.device(device)
        self._self_test_active = True
        try:
            # Ultralytics validation runs under torch.inference_mode(), which cannot be
            # overridden by enable_grad alone. Disable it before creating self-test tensors.
            with torch.inference_mode(False), torch.enable_grad():
                return self._verify_cuda_with_grad(device)
        finally:
            self._self_test_active = False

    def _verify_cuda_with_grad(self, device: torch.device) -> dict[str, object]:
        """Execute the formal self-test with ordinary autograd-capable tensors."""
        checks: dict[str, object] = {}
        torch.manual_seed(20260821)
        cpu_reference = TorchReferenceDirectionalScan()
        base = (
            torch.randn(1, 2, 3, 5),
            torch.randn(1, 2, 3, 5),
            -torch.rand(2, 3, 2),
            torch.randn(1, 2, 2, 5),
            torch.randn(1, 2, 2, 5),
            torch.randn(2, 3),
        )
        expected = cpu_reference(*(tensor.clone() for tensor in base))
        cuda_args = [tensor.to(device).requires_grad_(True) for tensor in base]
        self.train()
        actual = self(*cuda_args).float()
        max_error = float((actual.detach().cpu() - expected).abs().max())
        actual.square().mean().backward()
        fp32_gradients = all(arg.grad is not None and torch.isfinite(arg.grad).all() for arg in cuda_args)
        checks["fp32_max_abs_error"] = max_error
        checks["fp32_finite_gradients"] = bool(fp32_gradients)

        half_args = []
        for index, tensor in enumerate(base):
            dtype = torch.float32 if index in (2, 5) else torch.float16
            half_args.append(tensor.to(device=device, dtype=dtype).requires_grad_(True))
        half_output = self(*half_args)
        half_output.float().square().mean().backward()
        checks["fp16_finite"] = bool(
            torch.isfinite(half_output).all()
            and all(arg.grad is not None and torch.isfinite(arg.grad).all() for arg in half_args)
        )
        if max_error > 5e-4 or not fp32_gradients or checks["fp16_finite"] is not True:
            raise CUDAVerificationError(f"CUDA selective-scan self-test failed: {checks}")
        self.formal_cuda_verified = True
        checks["extension_path"] = str(self.extension_path)
        checks["extension_sha256"] = self.extension_sha256
        checks["formal_cuda_verified"] = True
        return checks


class DirectionalSelectiveScanBackend(nn.Module):
    """CPU semantic reference plus verified official CUDA execution, selected by device."""

    backend_name = "device_dispatch_directional_selective_scan"

    def __init__(self) -> None:
        super().__init__()
        self.reference = TorchReferenceDirectionalScan()
        self.cuda_backend = OfficialDirectionalSelectiveScanCUDA()

    @property
    def real_selective_scan(self) -> bool:
        return bool(self.cuda_backend.formal_cuda_verified)

    @property
    def formal_cuda_verified(self) -> bool:
        return bool(self.cuda_backend.formal_cuda_verified)

    def verify_cuda(self, device: torch.device | str = "cuda:0") -> dict[str, object]:
        return self.cuda_backend.verify_cuda(device)

    def forward(self, *args):
        if args[0].device.type == "cuda":
            if not self.cuda_backend.formal_cuda_verified:
                self.cuda_backend.verify_cuda(args[0].device)
            return self.cuda_backend(*args)
        return self.reference(*args)


__all__ = (
    "CUDABackendUnavailable",
    "CUDAVerificationError",
    "OfficialDirectionalSelectiveScanCUDA",
    "DirectionalSelectiveScanBackend",
)
