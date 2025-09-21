#!/bin/bash
# MLOps DenseNet Optimization - Build and Run Automation Script
# Compatible with GTX 1650 and CUDA 13.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./results"
GPU_ENABLED="true"
TENSORBOARD_PORT="6006"
PROFILER_PORT="6007"
BUILD_ONLY="false"
DETACHED="false"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         MLOps DenseNet Optimization - Build & Run           ║"
    echo "║              GTX 1650 + CUDA 13.0 Compatible                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR     Output directory for results (default: ./results)"
    echo "  --gpu-enabled BOOL   Enable GPU support (default: true)"
    echo "  --tensorboard-port N TensorBoard port (default: 6006)"
    echo "  --profiler-port N    Profiler port (default: 6007)"
    echo "  --build-only         Only build, don't run"
    echo "  --detached          Run in detached mode"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Run with defaults"
    echo "  $0 --output-dir ./my-results --gpu-enabled true"
    echo "  $0 --build-only                             # Build only"
    echo "  $0 --detached                               # Run in background"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-enabled)
            GPU_ENABLED="$2"
            shift 2
            ;;
        --tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --profiler-port)
            PROFILER_PORT="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY="true"
            shift
            ;;
        --detached)
            DETACHED="true"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Print configuration
print_config() {
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  GPU Enabled: $GPU_ENABLED"
    echo "  TensorBoard Port: $TENSORBOARD_PORT"
    echo "  Profiler Port: $PROFILER_PORT"
    echo "  Build Only: $BUILD_ONLY"
    echo "  Detached Mode: $DETACHED"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found: $(docker --version)${NC}"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker Compose found: $(docker-compose --version)${NC}"
    
    # Check NVIDIA Docker (if GPU enabled)
    if [ "$GPU_ENABLED" = "true" ]; then
        if ! docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}⚠️  GPU support requested but NVIDIA Docker runtime not available${NC}"
            echo -e "${YELLOW}   Continuing with CPU-only mode...${NC}"
            GPU_ENABLED="false"
        else
            echo -e "${GREEN}✅ NVIDIA Docker runtime available${NC}"
            echo -e "${GREEN}   GPU Info:${NC}"
            docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
        fi
    fi
}

# Setup directories
setup_directories() {
    echo -e "${YELLOW}Setting up directories...${NC}"
    
    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/profiles"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "./logs/tensorboard"
    mkdir -p "./data"
    
    # Set permissions
    chmod 755 "$OUTPUT_DIR"
    chmod 755 "$OUTPUT_DIR/profiles"
    chmod 755 "$OUTPUT_DIR/models"
    chmod 755 "./logs"
    
    echo -e "${GREEN}✅ Directories created:${NC}"
    echo "  📁 $OUTPUT_DIR/ (main results)"
    echo "  📁 $OUTPUT_DIR/profiles/ (profiling reports)"
    echo "  📁 $OUTPUT_DIR/models/ (model checkpoints)"
    echo "  📁 ./logs/tensorboard/ (TensorBoard logs)"
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    
    # Update docker-compose with custom ports
    export TENSORBOARD_PORT="$TENSORBOARD_PORT"
    export PROFILER_PORT="$PROFILER_PORT"
    
    # Build the image
    if docker-compose build --no-cache; then
        echo -e "${GREEN}✅ Docker image built successfully${NC}"
    else
        echo -e "${RED}❌ Failed to build Docker image${NC}"
        exit 1
    fi
}

# Run the container
run_container() {
    echo -e "${YELLOW}Starting MLOps DenseNet benchmarking...${NC}"
    
    # Update docker-compose ports
    sed -i.bak "s/\"6006:6006\"/\"$TENSORBOARD_PORT:6006\"/g" docker-compose.yml
    sed -i.bak "s/\"6007:6007\"/\"$PROFILER_PORT:6007\"/g" docker-compose.yml
    
    # Set GPU environment
    if [ "$GPU_ENABLED" = "true" ]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES=""
    fi
    
    # Update volume mounts to use custom output directory
    export RESULTS_DIR="$OUTPUT_DIR"
    
    # Run container
    if [ "$DETACHED" = "true" ]; then
        echo -e "${BLUE}Starting in detached mode...${NC}"
        docker-compose up -d
        
        echo -e "${GREEN}✅ Container started in background${NC}"
        echo -e "${CYAN}Monitor progress with: docker-compose logs -f mlops-densenet${NC}"
    else
        echo -e "${BLUE}Starting in interactive mode...${NC}"
        docker-compose up
    fi
}

# Wait for completion and display results
wait_and_display_results() {
    if [ "$DETACHED" = "true" ]; then
        echo -e "${YELLOW}Waiting for benchmarking to complete...${NC}"
        
        # Wait for the main container to finish
        while [ "$(docker-compose ps -q mlops-densenet)" ]; do
            if [ "$(docker-compose ps mlops-densenet | grep 'Exit')" ]; then
                break
            fi
            echo -e "${BLUE}Benchmarking in progress... (check with: docker-compose logs mlops-densenet)${NC}"
            sleep 30
        done
        
        echo -e "${GREEN}✅ Benchmarking completed${NC}"
    fi
    
    # Display results summary
    display_results_summary
}

# Display results summary
display_results_summary() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                      RESULTS SUMMARY                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    
    # Check if results file exists
    if [ -f "$OUTPUT_DIR/benchmark_results.csv" ]; then
        echo -e "${GREEN}📊 Benchmark Results:${NC}"
        echo -e "   📄 CSV File: $OUTPUT_DIR/benchmark_results.csv"
        
        # Show basic stats
        python3 << EOF
import pandas as pd
import os

try:
    df = pd.read_csv('$OUTPUT_DIR/benchmark_results.csv')
    print(f"   📈 Total Experiments: {len(df)}")
    print(f"   🔧 Optimization Techniques: {', '.join(df['optimization_technique'].unique())}")
    print(f"   📦 Batch Sizes: {sorted(df['batch_size'].unique())}")
    print(f"   💻 Device: {df['device'].iloc[0]}")
    
    if df['accuracy_top1'].notna().any():
        best_acc = df.loc[df['accuracy_top1'].idxmax()]
        print(f"   🎯 Best Accuracy: {best_acc['accuracy_top1']:.2f}% ({best_acc['optimization_technique']})")
    
    best_throughput = df.loc[df['throughput_samples_sec'].idxmax()]
    print(f"   ⚡ Best Throughput: {best_throughput['throughput_samples_sec']:.2f} samples/sec ({best_throughput['optimization_technique']})")
    
except Exception as e:
    print(f"   ❌ Error reading results: {e}")
EOF
    else
        echo -e "${RED}❌ Results file not found: $OUTPUT_DIR/benchmark_results.csv${NC}"
    fi
    
    # TensorBoard info
    echo -e "\n${GREEN}📊 TensorBoard Visualization:${NC}"
    echo -e "   🌐 Dashboard: http://localhost:$TENSORBOARD_PORT"
    echo -e "   📁 Logs Directory: ./logs/tensorboard/"
    
    # Additional outputs
    echo -e "\n${GREEN}📁 Additional Outputs:${NC}"
    echo -e "   📊 Profiling Reports: $OUTPUT_DIR/profiles/"
    echo -e "   🤖 Model Checkpoints: $OUTPUT_DIR/models/"
    
    # File sizes
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "\n${GREEN}💾 Output Directory Size:${NC}"
        du -sh "$OUTPUT_DIR" 2>/dev/null || echo "   Unable to calculate size"
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # # Restore original docker-compose.yml
    # if [ -f "docker-compose.yml.bak" ]; then
    #     mv docker-compose.yml.bak docker-compose.yml
    # fi
    
    # Stop containers if running
    if [ "$DETACHED" = "true" ]; then
        docker-compose down
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    print_banner
    print_config
    check_prerequisites
    setup_directories
    build_image
    
    if [ "$BUILD_ONLY" = "true" ]; then
        echo -e "${GREEN}✅ Build completed. Use 'docker-compose up' to run.${NC}"
        exit 0
    fi
    
    run_container
    wait_and_display_results
    
    echo -e "\n${GREEN}🎉 MLOps DenseNet Optimization completed successfully!${NC}"
    echo -e "${CYAN}💡 Keep containers running to access TensorBoard visualizations${NC}"
    echo -e "${CYAN}   Stop with: docker-compose down${NC}"
}

# Execute main function
main
