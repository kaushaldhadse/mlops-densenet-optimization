import os
import psutil
import torch
import io

try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
except Exception:
    _pynvml_available = False


def get_ram_usage_mb() -> float:
    """Returns current RAM usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


def get_cpu_utilization() -> float:
    """Returns system-wide CPU utilization percentage."""
    return psutil.cpu_percent(interval=0.1)


def get_vram_usage_mb() -> float:
    """Returns VRAM (GPU memory) usage in MB if GPU is available, else 0."""
    if not _pynvml_available:
        return 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / (1024 ** 2)
    except Exception:
        return 0.0


def get_gpu_utilization() -> float:
    """Returns GPU utilization percentage if available, else 0."""
    if not _pynvml_available:
        return 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimates model size in MB by serializing state_dict."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)
    return size_mb
