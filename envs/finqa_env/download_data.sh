#!/bin/bash
# Download FinQA data from HuggingFace
#
# This script downloads all FinQA data from HuggingFace:
# 1. Benchmark questions CSV
# 2. Company financial documents (preprocessed SEC 10-K filings)
#
# Usage:
#   ./download_data.sh <hf_repo_or_url> [output_dir]

set -e

HF_REPO_OR_URL="${1}"
OUTPUT_DIR="${2:-./data}"

if [ -z "$HF_REPO_OR_URL" ]; then
    echo "Usage: $0 <hf_repo_or_url> [output_dir]"
    echo "Example: $0 snorkelai/finqa-data ./data"
    exit 1
fi

echo "========================================"
echo "FinQA Data Download"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if data already exists
if [ -f "$OUTPUT_DIR/benchmark_questions/finqa.csv" ] && [ -d "$OUTPUT_DIR/input_companies" ]; then
    echo "Data already exists in $OUTPUT_DIR, skipping download."
    exit 0
fi

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo "Install it with: uv pip install huggingface_hub[cli]"
    exit 1
fi

# Download from HuggingFace
echo "Downloading from HuggingFace: $HF_REPO_OR_URL"
if ! huggingface-cli download "$HF_REPO_OR_URL" --repo-type dataset --local-dir "$OUTPUT_DIR"; then
    echo "Error: Failed to download dataset"
    exit 1
fi

# Verify downloaded data
if [ ! -f "$OUTPUT_DIR/benchmark_questions/finqa.csv" ]; then
    echo "Error: benchmark_questions/finqa.csv not found in downloaded data"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR/input_companies" ]; then
    echo "Error: input_companies/ directory not found in downloaded data"
    exit 1
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo "Data location: $OUTPUT_DIR"
echo ""

# Export data path
export FINQA_DATA_PATH="$OUTPUT_DIR"
echo "Exported: FINQA_DATA_PATH=$FINQA_DATA_PATH"
