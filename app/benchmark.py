import os
import time
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import psutil
import utils  # our helper functions (in app/utils.py)


# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZES = [1, 4, 8, 16, 32]
INPUT_SHAPE = (3, 224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_CSV_DIR = os.path.join(PROJECT_ROOT, "results\\benchmarking_results.csv")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs\\tensorboard")
os.makedirs(LOG_DIR, exist_ok=True)



def benchmark_densenet():
    os.makedirs(LOG_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)
    results = []

    print(f"Running benchmarks on device: {DEVICE}")

    # -----------------------------
    # Load model
    # -----------------------------
    start_time = time.time()
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(DEVICE)  # no pretrained weights
    model.eval()
    model_load_time = (time.time() - start_time) * 1000  # ms

    # -----------------------------
    # Run for each batch size
    # -----------------------------
    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking batch size = {batch_size}")
        inputs = torch.randn(batch_size, *INPUT_SHAPE).to(DEVICE)

        # warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(inputs)

        # profiling
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if DEVICE == "cuda" else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                torch.cuda.synchronize() if DEVICE == "cuda" else None
                start = time.time()
                with torch.no_grad():
                    outputs = model(inputs)
                torch.cuda.synchronize() if DEVICE == "cuda" else None
                end = time.time()

        # latency and throughput
        latency_ms = (end - start) * 1000
        throughput = batch_size / (end - start)

        # system stats
        ram_usage = utils.get_ram_usage_mb()
        vram_usage = utils.get_vram_usage_mb() if DEVICE == "cuda" else 0
        cpu_util = utils.get_cpu_utilization()
        gpu_util = utils.get_gpu_utilization() if DEVICE == "cuda" else 0

        # log to tensorboard
        writer.add_scalar(f"Latency/batch_{batch_size}", latency_ms)
        writer.add_scalar(f"Throughput/batch_{batch_size}", throughput)
        writer.add_scalar(f"RAM_Usage_MB/batch_{batch_size}", ram_usage)
        if DEVICE == "cuda":
            writer.add_scalar(f"VRAM_Usage_MB/batch_{batch_size}", vram_usage)
            writer.add_scalar(f"GPU_Utilization/batch_{batch_size}", gpu_util)

        # append to results
        results.append({
            "model_variant": "densenet121_baseline",
            "batch_size": batch_size,
            "device": DEVICE,
            "ram_usage_mb": ram_usage,
            "vram_usage_mb": vram_usage,
            "cpu_utilization_pct": cpu_util,
            "gpu_utilization_pct": gpu_util,
            "latency_ms": latency_ms,
            "throughput_samples_sec": throughput,
            "accuracy_top1": "NA",  # will be added in Part 2
            "accuracy_top5": "NA",
            "model_size_mb": utils.get_model_size_mb(model),
            "optimization_technique": "baseline",
            "model_load_time_ms": model_load_time
        })

    writer.close()

    # save CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULT_CSV_DIR, index=False)
    print(f"\n✅ Results saved to {RESULT_CSV_DIR}")


if __name__ == "__main__":
    benchmark_densenet()
