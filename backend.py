"""
Backend helpers for the ROCm-oriented autoresearch fork.
"""

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: torch.device
    device_type: str
    device_name: str
    runtime_version: str
    precision: str
    autocast_dtype: torch.dtype
    use_grad_scaler: bool
    supports_compile: bool
    theoretical_peak_flops: float | None = None


def resolve_backend():
    if platform.system() != "Linux":
        raise SystemExit("This fork only supports native Linux ROCm environments.")
    if not torch.cuda.is_available():
        raise SystemExit("No ROCm GPU is visible to PyTorch. Install ROCm and a ROCm PyTorch build.")
    runtime_version = getattr(torch.version, "hip", None)
    if runtime_version is None:
        raise SystemExit("This fork targets AMD ROCm. Install the ROCm PyTorch wheel, not CUDA.")
    return BackendInfo(
        name="rocm",
        device=torch.device("cuda"),
        device_type="cuda",
        device_name=torch.cuda.get_device_name(0),
        runtime_version=runtime_version,
        precision="fp16",
        autocast_dtype=torch.float16,
        use_grad_scaler=True,
        supports_compile=False,
        theoretical_peak_flops=None,
    )


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def synchronize():
    torch.cuda.synchronize()


def max_memory_allocated_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024
