#!/bin/bash

# MLOps DenseNet Optimization - Docker Entrypoint Script
# This script orchestrates the complete benchmarking pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              MLOps DenseNet Optimization Suite              ║"
    echo "║                  Docker Containerized Benchmark             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print section header
print_section() {
    echo -e "\n${YELLOW}==== $1 ====${NC}\n"
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Print info message
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check GPU availability
check_gpu() {
    print_section "GPU Availability Check"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        print_success "GPU detected and available"
        export CUDA_AVAILABLE=true
    else
        print_info "No GPU detected, running on CPU only"
        export CUDA_AVAILABLE=false
    fi
}

# Verify Python environment
check_environment() {
    print_section "Environment Verification"
    
    echo "Python version: $(python --version)"
    echo "Python executable: $(which python)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "TorchVision version: $(python -c 'import torchvision; print(torchvision.__version__)')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
        echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    fi
    
    print_success "Environment verification completed"
}

# Setup directories
setup_directories() {
    print_section "Directory Setup"
    
    mkdir -p /densenet_optimization/results
    mkdir -p /densenet_optimization/logs/tensorboard
    mkdir -p /densenet_optimization/data
    
    # Set permissions
    chmod 755 /densenet_optimization/results
    chmod 755 /densenet_optimization/logs
    
    print_success "Directories created and configured"
}

# Run comprehensive benchmarking
run_benchmarking() {
    print_section "Starting Comprehensive DenseNet Benchmarking"
    
    cd /densenet_optimization
    
    print_info "Benchmark Configuration:"
    echo "  - Model: DenseNet-121"
    echo "  - Optimizations: Baseline, AMP, Quantization"
    echo "  - Batch Sizes: [1, 4, 8, 16, 32]"
    echo "  - Dataset: ImageNet Mini Validation (or fallback)"
    echo "  - Profiler: Enabled with TensorBoard integration"
    echo ""
    
    # Run the main benchmark script
    print_info "Executing benchmark script..."
    python app/benchmark.py 2>&1 | tee /densenet_optimization/logs/benchmark_execution.log
    
    if [ $? -eq 0 ]; then
        print_success "Benchmarking completed successfully"
    else
        print_error "Benchmarking failed - check logs for details"
        exit 1
    fi
}

# Generate summary report
generate_summary() {
    print_section "Generating Summary Report"
    
    if [ -f "/densenet_optimization/results/benchmark_results.csv" ]; then
        python3 << 'EOF'
import pandas as pd
import os

# Read results
try:
    df = pd.read_csv('/densenet_optimization/results/benchmark_results.csv')
    
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total experiments: {len(df)}")
    print(f"Optimization techniques: {df['optimization_technique'].nunique()}")
    print(f"Batch sizes tested: {sorted(df['batch_size'].unique())}")
    print(f"Device used: {df['device'].iloc[0]}")
    
    print("\n🏆 PERFORMANCE HIGHLIGHTS:")
    print("-" * 30)
    
    # Best throughput
    best_throughput = df.loc[df['throughput_samples_sec'].idxmax()]
    print(f"Best Throughput: {best_throughput['throughput_samples_sec']:.2f} samples/sec")
    print(f"  └─ {best_throughput['optimization_technique']} @ batch_size={best_throughput['batch_size']}")
    
    # Best accuracy
    if df['accuracy_top1'].notna().any():
        best_accuracy = df.loc[df['accuracy_top1'].idxmax()]
        print(f"Best Top-1 Accuracy: {best_accuracy['accuracy_top1']:.2f}%")
        print(f"  └─ {best_accuracy['optimization_technique']} @ batch_size={best_accuracy['batch_size']}")
    
    # Lowest latency
    best_latency = df.loc[df['latency_ms'].idxmin()]
    print(f"Lowest Latency: {best_latency['latency_ms']:.2f}ms")
    print(f"  └─ {best_latency['optimization_technique']} @ batch_size={best_latency['batch_size']}")
    
    print("\n📈 OPTIMIZATION COMPARISON:")
    print("-" * 30)
    
    # Group by optimization technique
    opt_summary = df.groupby('optimization_technique').agg({
        'latency_ms': 'mean',
        'throughput_samples_sec': 'mean',
        'accuracy_top1': 'mean',
        'ram_usage_mb': 'mean'
    }).round(2)
    
    for opt, row in opt_summary.iterrows():
        print(f"{opt.upper()}:")
        print(f"  Avg Latency: {row['latency_ms']:.2f}ms")
        print(f"  Avg Throughput: {row['throughput_samples_sec']:.2f} samples/sec")
        if not pd.isna(row['accuracy_top1']):
            print(f"  Avg Accuracy: {row['accuracy_top1']:.2f}%")
        print(f"  Avg RAM Usage: {row['ram_usage_mb']:.2f}MB")
        print()
    
    print("📁 OUTPUT FILES:")
    print("-" * 15)
    print("  📄 CSV Results: /densenet_optimization/results/benchmark_results.csv")
    print("  📊 TensorBoard Logs: /densenet_optimization/logs/tensorboard/")
    print("  🔍 Profiler Traces: /densenet_optimization/logs/tensorboard/profiler_*/")
    
except Exception as e:
    print(f"Error generating summary: {e}")
EOF
        
        print_success "Summary report generated"
    else
        print_error "Results file not found - benchmark may have failed"
    fi
}

# Print TensorBoard access information
print_tensorboard_info() {
    print_section "TensorBoard Access Information"
    
    echo "🌐 TensorBoard Web Interfaces:"
    echo "  📊 Dashboard: http://localhost:6006"
    echo ""
    echo "📂 Log Directory: /densenet_optimization/logs/tensorboard/"
    echo ""
    echo "💡 Usage Tips:"
    echo "  - Use the SCALARS tab to compare optimization techniques"
    echo "  - Use the PROFILE tab to analyze detailed performance traces"
    echo "  - Compare different batch sizes using the X-axis controls"
    
    print_success "TensorBoard is accessible via the above URLs"
}

# Main execution function
main() {
    print_banner
    
    # Parse command line arguments
    case "${1:-benchmark}" in
        "benchmark")
            print_info "Running full benchmarking suite..."
            check_gpu
            check_environment
            setup_directories
            run_benchmarking
            generate_summary
            print_tensorboard_info
            ;;
        "test")
            print_info "Running environment test..."
            check_gpu
            check_environment
            print_success "Environment test completed"
            ;;
        "tensorboard")
            print_info "Starting TensorBoard only..."
            print_tensorboard_info
            # Keep container running
            tail -f /dev/null
            ;;
        "shell")
            print_info "Starting interactive shell..."
            exec /bin/bash
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Available commands: benchmark, test, tensorboard, shell"
            exit 1
            ;;
    esac
    
    print_success "MLOps DenseNet Optimization completed successfully!"
    
    # Keep container running to allow TensorBoard access
    print_info "Container will keep running for TensorBoard access..."
    print_info "Press Ctrl+C to stop or use 'docker-compose down'"
    
    # Wait indefinitely
    tail -f /dev/null
}

# Execute main function with all arguments
main "$@"
