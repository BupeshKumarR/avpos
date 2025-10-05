#!/bin/bash
# HPC Setup Script for Adaptive Privacy-Preserving Face Obfuscation
# This script sets up the environment on HPC clusters

set -e  # Exit on any error

echo "🚀 Setting up HPC environment for Adaptive Privacy-Preserving Face Obfuscation"
echo "=================================================================="

# Configuration
PYTHON_VERSION="3.12"
CONDA_ENV_NAME="avpos"
PROJECT_DIR="$(pwd)"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for conda/mamba
if command_exists mamba; then
    CONDA_CMD="mamba"
    echo "✅ Found mamba"
elif command_exists conda; then
    CONDA_CMD="conda"
    echo "✅ Found conda"
else
    echo "❌ Neither conda nor mamba found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment: $CONDA_ENV_NAME"
$CONDA_CMD env create -f environment.yml --force

# Activate environment
echo "🔄 Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Verify installation
echo "🔍 Verifying installation..."

# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check other key packages
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
python -c "import seaborn; print(f'Seaborn version: {seaborn.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import facenet_pytorch; print('FaceNet-PyTorch: OK')"
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
python -c "import kornia; print(f'Kornia version: {kornia.__version__}')"

# Test data loading
echo "🧪 Testing data loading pipeline..."
python sanity_check_data_loader.py

# Create HPC-specific directories
echo "📁 Creating HPC directories..."
mkdir -p $PROJECT_DIR/data
mkdir -p $PROJECT_DIR/outputs
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/checkpoints
mkdir -p $PROJECT_DIR/results

# Set permissions
chmod +x *.py
chmod +x scripts/*.py

echo "✅ HPC environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your dataset to: $PROJECT_DIR/data/"
echo "2. Run training: python master_training.py --data_root $PROJECT_DIR/data/your_dataset"
echo "3. Monitor logs: tail -f logs/training.log"
echo ""
echo "🔧 Environment info:"
echo "   Python: $(python --version)"
echo "   Conda env: $CONDA_ENV_NAME"
echo "   Project dir: $PROJECT_DIR"
echo "   CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
