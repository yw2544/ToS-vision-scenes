#!/usr/bin/env bash
# ToS-vision-scenes Setup Script
# ===============================
# This module is part of Theory-of-Space.
# Run this AFTER setting up Theory-of-Space (source ../setup.sh)
#
# Usage: source setup.sh

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run: source setup.sh  (to keep conda env active)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the tos environment
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if tos conda environment exists
if ! conda env list | grep -q "^tos "; then
  echo "Error: 'tos' conda environment not found."
  echo "Please set up Theory-of-Space first:"
  echo "  cd .. && source setup.sh"
  return 1
fi

conda activate tos

echo "Installing ToS-vision-scenes dependencies..."
pip install -r requirements.txt

# Download model assets (public dataset, no login required)
echo ""
echo "Downloading model assets from Hugging Face..."
MODEL_LIB_DIR="$SCRIPT_DIR/models/model_import/model_lib"
if [ ! -d "$MODEL_LIB_DIR" ]; then
  huggingface-cli download yw12356/ToS_model_lib --repo-type dataset --local-dir "$MODEL_LIB_DIR"
  echo "Model library downloaded to $MODEL_LIB_DIR"
else
  echo "Model library already exists at $MODEL_LIB_DIR, skipping download."
fi

echo ""
echo "============================================="
echo "ToS-vision-scenes setup complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Configure Unity path in config.yaml:"
echo "   model_import:"
echo "     unity_path: \"/path/to/Unity\""
echo ""
echo "2. Build asset bundles:"
echo "   cd models/model_import && ./build_all_bundles.sh"
echo ""
echo "3. Run the pipeline:"
echo "   python generate_pipeline.py --config config.yaml"
echo ""
