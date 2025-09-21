import os
import time
import torch
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import utils
import optimisations  # NEW
import evaluate_accuracy  # Import accuracy evaluation functions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Which optimization techniques to benchmark
OPTIMIZATIONS = ["baseline", "amp", "quantization"]  # add "jit", "quantization" later if needed
BATCH_SIZES = [1, 4, 8, 16, 32]


def extract_profiler_metrics(prof, device):
    """
    Extract key metrics from PyTorch profiler data.
    
    Args:
        prof: PyTorch profiler object
        device: Device type ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary containing profiler metrics
    """
    try:
        # Get profiler events
        events = prof.events()
        
        # Initialize metrics
        metrics = {
            'cpu_time_total_ms': 0.0,
            'cuda_time_total_ms': 0.0,
            'cpu_memory_usage_mb': 0.0,
            'cuda_memory_usage_mb': 0.0,
            'num_cpu_ops': 0,
            'num_cuda_ops': 0,
            'top_cpu_ops': [],
            'top_cuda_ops': []
        }
        
        # Collect CPU and CUDA times
        cpu_events = []
        cuda_events = []
        
        for event in events:
            if event.device_type == torch.profiler.DeviceType.CPU:
                metrics['cpu_time_total_ms'] += event.cpu_time_total / 1000.0  # Convert to ms
                metrics['num_cpu_ops'] += 1
                cpu_events.append((event.name, event.cpu_time_total / 1000.0))
                
                # Memory usage (if available)
                if hasattr(event, 'cpu_memory_usage') and event.cpu_memory_usage:
                    metrics['cpu_memory_usage_mb'] += event.cpu_memory_usage / (1024 * 1024)
                    
            elif event.device_type == torch.profiler.DeviceType.CUDA and device == "cuda":
                metrics['cuda_time_total_ms'] += event.cuda_time_total / 1000.0  # Convert to ms
                metrics['num_cuda_ops'] += 1
                cuda_events.append((event.name, event.cuda_time_total / 1000.0))
                
                # CUDA memory usage (if available)
                if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage:
                    metrics['cuda_memory_usage_mb'] += event.cuda_memory_usage / (1024 * 1024)
        
        # Get top 3 CPU operations by time
        cpu_events.sort(key=lambda x: x[1], reverse=True)
        metrics['top_cpu_ops'] = [f"{name}:{time_ms:.2f}ms" for name, time_ms in cpu_events[:3]]
        
        # Get top 3 CUDA operations by time (if CUDA)
        if device == "cuda":
            cuda_events.sort(key=lambda x: x[1], reverse=True)
            metrics['top_cuda_ops'] = [f"{name}:{time_ms:.2f}ms" for name, time_ms in cuda_events[:3]]
        
        return metrics
        
    except Exception as e:
        logging.error(f"Failed to extract profiler metrics: {e}")
        return {
            'cpu_time_total_ms': 0.0,
            'cuda_time_total_ms': 0.0,
            'cpu_memory_usage_mb': 0.0,
            'cuda_memory_usage_mb': 0.0,
            'num_cpu_ops': 0,
            'num_cuda_ops': 0,
            'top_cpu_ops': [],
            'top_cuda_ops': []
        }

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

            # Profiling + inference with TensorBoard integration
            profiler_log_dir = os.path.join(log_dir, f"profiler_{opt}_batch_{batch_size}")
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                # TensorBoard integration
                on_trace_ready=tensorboard_trace_handler(profiler_log_dir),
                schedule=schedule(wait=1, warmup=1, active=3, repeat=1)
            ) as prof:
                with record_function(f"model_inference_{opt}_batch_{batch_size}"):
                    torch.cuda.synchronize() if device == "cuda" else None
                    start = time.time()
                    with torch.no_grad():
                        # Run multiple steps for the profiler schedule
                        for step in range(5):  # wait=1, warmup=1, active=3
                            if opt == "amp" and device == "cuda":
                                with torch.amp.autocast("cuda"):
                                    outputs = model(input_tensor)
                            else:
                                outputs = model(input_tensor)
                            prof.step()  # Signal profiler to move to next step
                    torch.cuda.synchronize() if device == "cuda" else None
                    end = time.time()

            # Latency & throughput
            latency_ms = (end - start) * 1000
            throughput = batch_size / (end - start)

            # Extract profiler data
            profiler_data = extract_profiler_metrics(prof, device)
            
            # Log profiler trace location
            print(f"[INFO] Profiler trace saved to: {profiler_log_dir}")

            # Collect metrics
            ram_usage = utils.get_ram_usage_mb()
            vram_usage = utils.get_vram_usage_mb() if device == "cuda" else None
            cpu_util = utils.get_cpu_utilization()
            gpu_util = utils.get_gpu_utilization() if device == "cuda" else None
            model_size_mb = utils.get_model_size_mb(model)

            # Calculate accuracy using a subset of test data
            print(f"[INFO] Calculating accuracy for {opt} with batch_size={batch_size}...")
            use_amp = (opt == "amp" and device == "cuda")
            accuracy_device = run_device  # Use the same device as the model
            
            try:
                acc_top1, acc_top5 = evaluate_accuracy.evaluate_model_subset(
                    model, 
                    device=accuracy_device, 
                    use_amp=use_amp, 
                    num_batches=5,  # Use 5 batches for accuracy evaluation to balance speed vs accuracy
                    batch_size=min(batch_size, 32)  # Cap evaluation batch size at 32 for memory efficiency
                )
                print(f"[INFO] Accuracy - Top1: {acc_top1:.2f}%, Top5: {acc_top5:.2f}%")
            except Exception as e:
                logging.error(f"Failed to calculate accuracy for {opt}: {e}")
                acc_top1, acc_top5 = 0.0, 0.0

            # Log to TensorBoard with global_step for proper X-axis visualization
            writer.add_scalar(f"{opt}/Latency", latency_ms, global_step=batch_size)
            writer.add_scalar(f"{opt}/Throughput", throughput, global_step=batch_size)
            writer.add_scalar(f"{opt}/RAM_MB", ram_usage, global_step=batch_size)
            writer.add_scalar(f"{opt}/Accuracy_Top1", acc_top1, global_step=batch_size)
            writer.add_scalar(f"{opt}/Accuracy_Top5", acc_top5, global_step=batch_size)
            
            # Log profiler data to TensorBoard
            writer.add_scalar(f"{opt}/Profiler_CPU_Time_ms", profiler_data.get('cpu_time_total_ms', 0.0), global_step=batch_size)
            writer.add_scalar(f"{opt}/Profiler_Num_CPU_Ops", profiler_data.get('num_cpu_ops', 0), global_step=batch_size)
            
            if device == "cuda":
                writer.add_scalar(f"{opt}/VRAM_MB", vram_usage, global_step=batch_size)
                writer.add_scalar(f"{opt}/GPU_util", gpu_util, global_step=batch_size)
                writer.add_scalar(f"{opt}/Profiler_CUDA_Time_ms", profiler_data.get('cuda_time_total_ms', 0.0), global_step=batch_size)
                writer.add_scalar(f"{opt}/Profiler_Num_CUDA_Ops", profiler_data.get('num_cuda_ops', 0), global_step=batch_size)

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
                "accuracy_top1": acc_top1,
                "accuracy_top5": acc_top5,
                "model_size_mb": model_size_mb,
                "optimization_technique": opt,
                "model_load_time_ms": model_load_time,
                # Profiler data
                "profiler_cpu_time_ms": profiler_data.get('cpu_time_total_ms', 0.0),
                "profiler_cuda_time_ms": profiler_data.get('cuda_time_total_ms', 0.0),
                "profiler_cpu_memory_mb": profiler_data.get('cpu_memory_usage_mb', 0.0),
                "profiler_cuda_memory_mb": profiler_data.get('cuda_memory_usage_mb', 0.0),
                "profiler_num_cpu_ops": profiler_data.get('num_cpu_ops', 0),
                "profiler_num_cuda_ops": profiler_data.get('num_cuda_ops', 0),
                "profiler_top_cpu_ops": "; ".join(profiler_data.get('top_cpu_ops', [])),
                "profiler_top_cuda_ops": "; ".join(profiler_data.get('top_cuda_ops', [])),
            })

            print(f"[DONE] {opt} | batch={batch_size} | "
                  f"latency={latency_ms:.2f}ms | throughput={throughput:.2f}/s | "
                  f"acc_top1={acc_top1:.2f}% | acc_top5={acc_top5:.2f}% | "
                  f"cpu_ops={profiler_data.get('num_cpu_ops', 0)} | "
                  f"profiler_cpu_time={profiler_data.get('cpu_time_total_ms', 0.0):.2f}ms")

    # -----------------------------
    # Save results with required structure
    # -----------------------------
    # results_dir = os.path.join(project_root, "results")
    results_dir = os.environ.get("RESULTS_DIR", os.path.join(project_root, "results"))
    os.makedirs(results_dir, exist_ok=True)

    profiles_dir = os.path.join(results_dir, "profiles")
    models_dir = os.path.join(results_dir, "models")
    
    # Create required directories
    # os.makedirs(results_dir, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save main CSV with exact required format
    result_csv = os.path.join(results_dir, "benchmark_results.csv")
    df = pd.DataFrame(results)
    
    # Ensure CSV has exact required columns in correct order
    required_columns = [
        'model_variant', 'batch_size', 'device', 'ram_usage_mb', 'vram_usage_mb',
        'cpu_utilization_pct', 'gpu_utilization_pct', 'latency_ms', 'throughput_samples_sec',
        'accuracy_top1', 'accuracy_top5', 'model_size_mb', 'optimization_technique'
    ]
    
    # Reorder columns and add any missing ones
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Save with required columns only
    df_output = df[required_columns]
    if os.path.exists(result_csv):
        os.remove(result_csv)
    df_output.to_csv(result_csv, index=False)
    
    # Save detailed results with all profiler data
    detailed_csv = os.path.join(results_dir, "benchmark_results_detailed.csv")
    df.to_csv(detailed_csv, index=False)
    print(f"[INFO] ✅ Main results CSV written to {result_csv}")

    
    # Export profiler traces to profiles directory
    print(f"[INFO] Exporting profiler traces to {profiles_dir}")
    try:
        import shutil
        profiler_source = os.path.join(log_dir, "profiler_*")
        import glob
        for profiler_dir in glob.glob(os.path.join(log_dir, "profiler_*")):
            if os.path.isdir(profiler_dir):
                dest_name = os.path.basename(profiler_dir)
                dest_path = os.path.join(profiles_dir, dest_name)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(profiler_dir, dest_path)
                print(f"[INFO] Copied profiler trace: {dest_name}")
    except Exception as e:
        print(f"[WARN] Failed to copy profiler traces: {e}")
    
    # Save model information to models directory
    model_info = {
        'model_architecture': 'DenseNet-121',
        'optimization_techniques': list(set(df['optimization_technique'])),
        'total_parameters': 'TBD',  # Could be calculated
        'model_size_mb': df['model_size_mb'].iloc[0] if len(df) > 0 else 0,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    import json
    model_info_file = os.path.join(models_dir, "model_info.json")
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"[INFO] Model information saved to {model_info_file}")

    writer.close()
    print(f"\n✅ Benchmarking complete!")
    print(f"\n📁 Output Structure:")
    print(f"   📄 Main Results: {result_csv}")
    print(f"   📊 Detailed Results: {detailed_csv}")
    print(f"   🔍 Profiling Reports: {profiles_dir}/")
    print(f"   🤖 Model Checkpoints: {models_dir}/")
    print(f"\n📊 TensorBoard Visualization:")
    print(f"   Metrics: tensorboard --logdir {log_dir}")
    print(f"   Profiler traces: tensorboard --logdir {log_dir} --port 6007")
    print(f"\n💡 In TensorBoard, go to the 'PROFILE' tab to view detailed profiler traces")


if __name__ == "__main__":
    benchmark_densenet()

