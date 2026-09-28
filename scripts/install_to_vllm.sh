#!/bin/bash
# Installation script for MiniCPM Attention Optimization
# Copies kernels to vLLM plugin directory

set -e

echo "==================================="
echo "MiniCPM Attention Optimization"
echo "Installation Script"
echo "==================================="

# Detect vLLM plugin directory
VLLM_PLUGIN_DIR="${VLLM_PLUGIN_DIR:-}"

if [ -z "$VLLM_PLUGIN_DIR" ]; then
    # Try common locations
    if [ -d "$HOME/vllm-plugin-FL" ]; then
        VLLM_PLUGIN_DIR="$HOME/vllm-plugin-FL"
    elif [ -d "./vllm-plugin-FL" ]; then
        VLLM_PLUGIN_DIR="./vllm-plugin-FL"
    else
        echo "Error: vLLM plugin directory not found."
        echo "Please set VLLM_PLUGIN_DIR environment variable or install vllm-plugin-FL first."
        exit 1
    fi
fi

TARGET_DIR="$VLLM_PLUGIN_DIR/vllm_fl/dispatch/backends/vendor/iluvatar/impl/ops"

echo "Target directory: $TARGET_DIR"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory does not exist: $TARGET_DIR"
    exit 1
fi

# Create backup
BACKUP_DIR="$TARGET_DIR/backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

if [ -f "$TARGET_DIR/triton_unified_attention_optimized.py" ]; then
    cp "$TARGET_DIR/triton_unified_attention_optimized.py" "$BACKUP_DIR/"
fi
if [ -f "$TARGET_DIR/triton_split_kv_paged.py" ]; then
    cp "$TARGET_DIR/triton_split_kv_paged.py" "$BACKUP_DIR/"
fi

# Copy optimized kernels
echo "Installing optimized kernels..."
cp kernels/triton_unified_attention_optimized.py "$TARGET_DIR/"
cp kernels/triton_split_kv_paged.py "$TARGET_DIR/"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To enable the validated Split-KV path, set environment variables:"
echo "  export ILUVATAR_USE_OPTIMIZED=1"
echo "  export ILUVATAR_SPLIT_KV=1"
echo "  export ILUVATAR_NUM_SPLITS=4"
echo ""
echo "Backup saved at: $BACKUP_DIR"
