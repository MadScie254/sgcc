#!/bin/bash
# Download SGCC dataset from Kaggle
# Requires Kaggle API credentials (kaggle.json in ~/.kaggle/)

set -e

echo "[INFO] Downloading SGCC dataset from Kaggle..."

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "[ERROR] Kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Check for credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "$KAGGLE_USERNAME" ]; then
    echo "[ERROR] Kaggle credentials not found."
    echo "Either:"
    echo "  1. Place kaggle.json in ~/.kaggle/"
    echo "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download dataset
echo "[INFO] Downloading from bensalem14/sgcc-dataset..."
kaggle datasets download -d bensalem14/sgcc-dataset -p data --unzip

echo "[SUCCESS] Dataset downloaded to data/"

# List downloaded files
echo "[INFO] Downloaded files:"
ls -lh data/

echo "[INFO] Dataset ready for processing!"
