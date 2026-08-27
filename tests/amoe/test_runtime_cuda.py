import copy
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
cuda_module = _load_module("ultralytics.nn.modules.drmamba_cuda", "drmamba_cuda.py")


def test_missing_official_extension_fails_closed():
    def missing_import(name):
        raise ImportError(name)

    backend = cuda_module.OfficialDirectionalSelectiveScanCUDA(import_module=missing_import)

    with pytest.raises(cuda_module.CUDABackendUnavailable, match="selective_scan_cuda_core"):
        backend.load()


def test_auto_backend_uses_only_cpu_reference_on_cpu():
    backend = cuda_module.DirectionalSelectiveScanBackend()
    reference = cuda_module.TorchReferenceDirectionalScan()
    args = (
        torch.randn(1, 2, 3, 4),
        torch.randn(1, 2, 3, 4),
        -torch.rand(2, 3, 2),
        torch.randn(1, 2, 2, 4),
        torch.randn(1, 2, 2, 4),
        torch.randn(2, 3),
    )

    torch.testing.assert_close(backend(*args), reference(*args))
    assert backend.real_selective_scan is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the formal CUDA backend")
def test_cuda_verification_enables_grad_inside_validation_context():
    backend = cuda_module.DirectionalSelectiveScanBackend().cuda()

    with torch.inference_mode():
        report = backend.verify_cuda()

    assert report["formal_cuda_verified"] is True
    assert report["fp32_finite_gradients"] is True
    assert report["fp16_finite"] is True

    cloned = copy.deepcopy(backend)
    assert cloned.cuda_backend.formal_cuda_verified is True
    assert cloned.cuda_backend._extension is None
