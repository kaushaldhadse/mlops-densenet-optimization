import os
import time
import torch
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import utils
import optimisations  # NEW

# Which optimization techniques to benchmark
OPTIMIZATIONS = ["baseline", "amp"]  # add "jit", "quantization" later if needed
BATCH_SIZES = [1, 4, 8, 16, 32]

def benchmark_densenet(batch_sizes=BATCH_SIZES, device=None):
    results = []

    # -----------------------------
    # Setup device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running benchmarks on device: {device}")

    # Setup TensorBoard writer
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        # fallback for Jupyter notebooks
        project_root = os.getcwd()
        
    log_dir = os.path.join(project_root, "logs", "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Sample input for optimisations (needed by JIT / AMP)
    sample_input = torch.randn(1, 3, 224, 224).to(device)

    # -----------------------------
    # Loop over optimizations
    # -----------------------------
    for opt in OPTIMIZATIONS:
        print(f"\n[INFO] === Optimization: {opt} ===")

        # Load model once per optimization
        start_time = time.time()
        
        model_full = optimisations.get_model("baseline", device, sample_input)  # get plain model first
        if opt == "quantization":
            model = optimisations.apply_quantization(model_full.cpu())  # quantize on CPU
            run_device = "cpu"
        else:
            model = optimisations.get_model(opt, device, sample_input)
            run_device = device

        model.eval()
        model_load_time = (time.time() - start_time) * 1000  # ms

        for batch_size in batch_sizes:
            print(f"[INFO] Benchmarking batch size = {batch_size}")
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(run_device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    if opt == "amp" and device == "cuda":
                        with torch.amp.autocast("cuda"):
                            _ = model(input_tensor)
                    else:
                        _ = model(input_tensor)

            # Profiling + inference
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    torch.cuda.synchronize() if device == "cuda" else None
                    start = time.time()
                    with torch.no_grad():
                        if opt == "amp" and device == "cuda":
                            with torch.amp.autocast("cuda"):
                                outputs = model(input_tensor)
                        else:
                            outputs = model(input_tensor)
                    torch.cuda.synchronize() if device == "cuda" else None
                    end = time.time()

            # Latency & throughput
            latency_ms = (end - start) * 1000
            throughput = batch_size / (end - start)

            # Collect metrics
            ram_usage = utils.get_ram_usage_mb()
            vram_usage = utils.get_vram_usage_mb() if device == "cuda" else None
            cpu_util = utils.get_cpu_utilization()
            gpu_util = utils.get_gpu_utilization() if device == "cuda" else None
            model_size_mb = utils.get_model_size_mb(model)

            # Log to TensorBoard
            writer.add_scalar(f"{opt}/Latency_batch_{batch_size}", latency_ms)
            writer.add_scalar(f"{opt}/Throughput_batch_{batch_size}", throughput)
            writer.add_scalar(f"{opt}/RAM_MB_batch_{batch_size}", ram_usage)
            if device == "cuda":
                writer.add_scalar(f"{opt}/VRAM_MB_batch_{batch_size}", vram_usage)
                writer.add_scalar(f"{opt}/GPU_util_batch_{batch_size}", gpu_util)

            # Append to results
            results.append({
                "model_variant": "densenet121",
                "batch_size": batch_size,
                "device": device,
                "ram_usage_mb": ram_usage,
                "vram_usage_mb": vram_usage,
                "cpu_utilization_pct": cpu_util,
                "gpu_utilization_pct": gpu_util,
                "latency_ms": latency_ms,
                "throughput_samples_sec": throughput,
                "accuracy_top1": None,   # will be added in Part 2
                "accuracy_top5": None,
                "model_size_mb": model_size_mb,
                "optimization_technique": opt,
                "model_load_time_ms": model_load_time,
            })

            print(f"[DONE] {opt} | batch={batch_size} | "
                  f"latency={latency_ms:.2f}ms | throughput={throughput:.2f}/s")

    # -----------------------------
    # Save results
    # -----------------------------
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_csv = os.path.join(results_dir, "benchmark_results.csv")

    df = pd.DataFrame(results)
    if os.path.exists(result_csv):
        os.remove(result_csv)
    df.to_csv(result_csv, index=False)

    writer.close()
    print(f"\n✅ Benchmarking complete. Results saved to {result_csv}")


if __name__ == "__main__":
    benchmark_densenet()






# import os
# import time
# import torch
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity
# from torch.utils.tensorboard import SummaryWriter
# import pandas as pd
# import psutil
# import utils  # our helper functions (in app/utils.py)


# # -----------------------------
# # CONFIG
# # -----------------------------
# BATCH_SIZES = [1, 4, 8, 16, 32]
# INPUT_SHAPE = (3, 224, 224)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# RESULT_CSV_DIR = os.path.join(PROJECT_ROOT, "results\\benchmarking_results.csv")
# LOG_DIR = os.path.join(PROJECT_ROOT, "logs\\tensorboard")
# os.makedirs(LOG_DIR, exist_ok=True)



# def benchmark_densenet():
#     os.makedirs(LOG_DIR, exist_ok=True)

#     writer = SummaryWriter(LOG_DIR)
#     results = []

#     print(f"Running benchmarks on device: {DEVICE}")

#     # -----------------------------
#     # Load model
#     # -----------------------------
#     start_time = time.time()
#     model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(DEVICE)  # no pretrained weights
#     model.eval()
#     model_load_time = (time.time() - start_time) * 1000  # ms

#     # -----------------------------
#     # Run for each batch size
#     # -----------------------------
#     for batch_size in BATCH_SIZES:
#         print(f"\nBenchmarking batch size = {batch_size}")
#         inputs = torch.randn(batch_size, *INPUT_SHAPE).to(DEVICE)

#         # warmup
#         with torch.no_grad():
#             for _ in range(3):
#                 _ = model(inputs)

#         # profiling
#         with profile(
#             activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if DEVICE == "cuda" else [ProfilerActivity.CPU],
#             record_shapes=True,
#             profile_memory=True,
#             with_stack=True
#         ) as prof:
#             with record_function("model_inference"):
#                 torch.cuda.synchronize() if DEVICE == "cuda" else None
#                 start = time.time()
#                 with torch.no_grad():
#                     outputs = model(inputs)
#                 torch.cuda.synchronize() if DEVICE == "cuda" else None
#                 end = time.time()

#         # latency and throughput
#         latency_ms = (end - start) * 1000
#         throughput = batch_size / (end - start)

#         # system stats
#         ram_usage = utils.get_ram_usage_mb()
#         vram_usage = utils.get_vram_usage_mb() if DEVICE == "cuda" else 0
#         cpu_util = utils.get_cpu_utilization()
#         gpu_util = utils.get_gpu_utilization() if DEVICE == "cuda" else 0

#         # log to tensorboard
#         writer.add_scalar(f"Latency/batch_{batch_size}", latency_ms)
#         writer.add_scalar(f"Throughput/batch_{batch_size}", throughput)
#         writer.add_scalar(f"RAM_Usage_MB/batch_{batch_size}", ram_usage)
#         if DEVICE == "cuda":
#             writer.add_scalar(f"VRAM_Usage_MB/batch_{batch_size}", vram_usage)
#             writer.add_scalar(f"GPU_Utilization/batch_{batch_size}", gpu_util)

#         # append to results
#         results.append({
#             "model_variant": "densenet121_baseline",
#             "batch_size": batch_size,
#             "device": DEVICE,
#             "ram_usage_mb": ram_usage,
#             "vram_usage_mb": vram_usage,
#             "cpu_utilization_pct": cpu_util,
#             "gpu_utilization_pct": gpu_util,
#             "latency_ms": latency_ms,
#             "throughput_samples_sec": throughput,
#             "accuracy_top1": "NA",  # will be added in Part 2
#             "accuracy_top5": "NA",
#             "model_size_mb": utils.get_model_size_mb(model),
#             "optimization_technique": "baseline",
#             "model_load_time_ms": model_load_time
#         })

#     writer.close()

#     # save CSV
#     df = pd.DataFrame(results)
#     df.to_csv(RESULT_CSV_DIR, index=False)
#     print(f"\n✅ Results saved to {RESULT_CSV_DIR}")


# if __name__ == "__main__":
#     benchmark_densenet()
