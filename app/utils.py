import os
import psutil
import torch
import io

try:
    from py3nvml import py3nvml
    py3nvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False



def get_ram_usage_mb() -> float:
    """Returns current RAM usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


def get_cpu_utilization() -> float:
    """Returns system-wide CPU utilization percentage."""
    return psutil.cpu_percent(interval=0.1)


def get_vram_usage_mb() -> float:
    """Returns GPU memory usage in MB for the first GPU."""
    if not _nvml_available:
        return 0.0
    try:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / (1024 ** 2)
    except Exception:
        return 0.0



def get_gpu_utilization() -> float:
    """Returns GPU utilization percentage for the first GPU."""
    if not _nvml_available:
        return 0.0
    try:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        util = py3nvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimates model size in MB by serializing state_dict."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)
    return size_mb
