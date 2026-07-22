from __future__ import annotations

from typing import Any


def select_flagembedding_device(
    requested_device: str,
    *,
    allow_cpu_fallback: bool,
    metadata: dict[str, Any],
) -> str | None:
    """Return a FlagEmbedding `devices` value when auto selection needs overriding."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device not in {"auto", "cuda", "cuda:0"}:
        return requested_device
    if not allow_cpu_fallback:
        return None if requested_device == "auto" else requested_device

    try:
        import torch
    except Exception as exc:
        metadata["device_probe_error"] = f"{type(exc).__name__}: {exc}"
        return None if requested_device == "auto" else requested_device

    if not torch.cuda.is_available():
        return None if requested_device == "auto" else requested_device

    try:
        major, minor = torch.cuda.get_device_capability()
        device_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
    except Exception as exc:
        metadata["device_probe_error"] = f"{type(exc).__name__}: {exc}"
        return None if requested_device == "auto" else requested_device

    if supported_arches and device_arch not in supported_arches:
        metadata["warning"] = (
            f"CUDA device {device_arch} is unsupported by installed torch; using CPU"
        )
        metadata["detected_cuda_arch"] = device_arch
        metadata["supported_cuda_arches"] = sorted(supported_arches)
        return "cpu"

    return None if requested_device == "auto" else requested_device
