from __future__ import annotations

import functools
from typing import Any


@functools.lru_cache(maxsize=1)
def get_hardware_info() -> dict[str, Any]:
    """
    Probe available compute hardware. Returns a dict with:
      device_type: "cuda" | "cpu"
      devices: list of {index, name, vram_mb}
      primary_name: human-readable name of the first / best device
      primary_vram_mb: usable VRAM in MiB (0 for CPU)
    """
    try:
        import torch

        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "vram_mb": props.total_memory // (1024 * 1024),
                    }
                )
            if devices:
                return {
                    "device_type": "cuda",
                    "devices": devices,
                    "primary_name": devices[0]["name"],
                    "primary_vram_mb": devices[0]["vram_mb"],
                }
    except Exception:
        pass

    # CPU fallback
    return {
        "device_type": "cpu",
        "devices": [],
        "primary_name": "CPU (no CUDA)",
        "primary_vram_mb": 0,
    }
