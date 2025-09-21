# MLOps DenseNet Optimization

## Project Overview

This repository provides a fully automated framework to benchmark **DenseNet-121** with various optimization techniques using **PyTorch**.
It collects detailed latency, throughput, memory usage, accuracy, and profiler data, and supports TensorBoard visualization.

### Key Features

- Benchmarking with **Baseline FP32**, **Automatic Mixed Precision (AMP)**, **Quantization (INT8 / dynamic)**.
- Support for **CPU and GPU (CUDA)**.
- Logging metrics to **TensorBoard** for real-time visualization.
- Generating **CSV reports** with summary and detailed profiler data.
- Saving **model info** and **profiler traces** for reproducibility.
- Handling multiple **batch sizes**.
- **Docker-based** reproducible environments.
- **GPU Tested**: Nvidia GeForce GTX 1650 Ti with MaxQ Design (4 GB).

---

## Setup Instructions

### Expected Project Structure

```
mlops-densenet-optimization
├── app
│   ├── benchmark.py
│   ├── optimisations.py
│   ├── evaluate_accuracy.py
│   └── utils.py
├── data
│   └── imagenet
├── logs
│   └── tensorboard
├── results
│   ├── models
│   └── profiles
├── requirements.txt
├── docker-entrypoint.sh
├── build_and_run.sh
├── README.md
└── docker-compose.yml
```

### Dataset Setup

The project expects the dataset (**ImageNet**) to be stored in the `data` folder.

```
data/
└── imagenet/
    ├── train/
    └── val/
```

### Prerequisites

- Docker Engine **20.10+**
- Docker Compose **2.0+**
- NVIDIA Docker Runtime (for GPU support)
- At least **8GB RAM**
- **10GB free disk space**
- WSL2 setup if running on Windows (and run the script within WSL)

---

## Usage Guide

### 1. Basic Run (Default Settings)

```bash
./build_and_run.sh
```

- Results → `./results/benchmark_results.csv`
- TensorBoard → [http://localhost:6006](http://localhost:6006)

### 2. Custom Ports

```bash
./build_and_run.sh --tensorboard-port 8006 --profiler-port 8007
```

### 3. Build Only

```bash
./build_and_run.sh --build-only
docker-compose up
```

### 4. Background Mode

```bash
./build_and_run.sh --detached
docker-compose logs -f mlops-densenet
docker-compose down
```

### 5. CPU-Only Mode

```bash
./build_and_run.sh --gpu-enabled false
```

---

## Expected Output Structure

```
./results/
├── benchmark_results.csv
├── benchmark_results_detailed.csv
├── profiles/
│   ├── profiler_baseline_batch_1/
│   ├── profiler_amp_batch_1/
│   └── ...
└── models/
    └── model_info.json

./logs/tensorboard/
├── events.out.tfevents.*
└── profiler_*/
```

---

## Optimization Approaches

### Baseline (FP32)

- Standard DenseNet-121 in **full precision**.
- Serves as a reference for accuracy and performance.

### Automatic Mixed Precision (AMP)

- Uses FP16 where safe, FP32 otherwise.
- Typically reduces latency and memory usage.
- On GTX 1650, **\~20% speedup observed**.

### Quantization (INT8 / Dynamic)

- Reduces model size and memory usage.
- Trades slight accuracy drop for lower latency.
- Best suited for **CPU inference**.

---

## Results Summary

### Key Metrics (GTX 1650)

- **VRAM Usage**: \~4GB limit, automatically adjusted.
- **Throughput**: Increases with AMP, constrained by VRAM at higher batch sizes.
- **Latency**: AMP significantly reduces latency at batch sizes 1–8.
- **Accuracy**: Remains stable across optimizations (AMP nearly identical to FP32).

### Example: Optimal Settings for GTX 1650

```bash
./build_and_run.sh
```

- Batch size **1–8** → Good performance
- Batch size **16+** → VRAM-limited

---

## Performance Analysis

### TensorBoard Insights

- **Scalars Tab**: Compare throughput, latency, and memory usage.
- **Profile Tab**: Identify GPU utilization bottlenecks.
- **Batch Size Axis**: Performance measured across 1, 4, 8, 16, 32.

### CSV Format

`benchmark_results.csv` columns:

```
model_variant,batch_size,device,ram_usage_mb,vram_usage_mb,
cpu_utilization_pct,gpu_utilization_pct,latency_ms,
throughput_samples_sec,accuracy_top1,accuracy_top5,
model_size_mb,optimization_technique
```

---

## Possible Issues and Solutions

### 1. Docker Credential Error

**Error:** error getting credentials - err: exit status 1

**Cause:**  
Docker Desktop was using `credsStore: "desktop.exe"` in `~/.docker/config.json`.

**Solution:**  
Edit `~/.docker/config.json` and remove or update the `credsStore` entry:

```json
{
  "auths": {
    "https://index.docker.io/v1/": {}
  }
}
```

### 2. NVIDIA Docker GPU Support in WSL2

**Error:**
RuntimeError: CUDA driver not found,
torch.cuda.is_available() -> False

**Cause:**

- NVIDIA GPU driver not properly installed for WSL2.
- `nvidia-docker2` or `nvidia-container-toolkit` missing.
- Docker Desktop not configured to expose GPU to WSL.

**Solution:**

- Enable GPU Support in Docker Desktop

  Open Docker Desktop → Settings → Resources → WSL Integration.

  Enable integration with your WSL2 distro.

  Under Settings → GPU, check "Use the WSL2 based engine" and enable GPU support.

- Install NVIDIA Container Toolkit inside WSL2

  ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker

  ```

- Test GPU Access in Docker

  ```bash
    docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04 nvidia-smi

  ```

## Trade-offs Discussion

- **AMP**: Faster and memory-efficient but requires hardware support.
- **Quantization**: Excellent for CPU inference, but accuracy may drop slightly.
- **Baseline FP32**: Most accurate but slowest and most memory-hungry.
- **Batch Size Scaling**: Higher batch sizes improve throughput until VRAM is saturated.

---

## Known Limitations

- GTX 1650 (4GB VRAM) restricts larger batch sizes.
- Quantization accuracy may degrade compared to FP32.
- Dataset must be pre-downloaded (ImageNet not included).
- Profiling overhead may slightly affect performance readings.

---

## Future Improvements

- Support for additional models (**ResNet, EfficientNet**).
- Integration with **ONNX Runtime** and **TensorRT** for inference.
- Advanced quantization (**PTQ, QAT**).
- Automated hyperparameter tuning for optimal batch sizes.
- Cloud-native deployment (**Kubernetes integration**).
