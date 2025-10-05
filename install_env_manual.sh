#!/bin/bash

# Manual Environment Installation Script
# Run this inside an interactive session: srun --partition=gpu-interactive --gres=gpu:h200:1 --mem=32GB --time=01:00:00 --pty /bin/bash

echo "=== Manual Environment Installation ==="
echo "Make sure you're in an interactive session!"
echo ""

# Load conda module
module load anaconda3/2022.05

# Create conda environment
echo "📦 Creating conda environment..."
conda create --prefix /courses/DS5500.202610/shared/team11/envs/avpos_env python=3.11 -y

# Activate environment
source activate /courses/DS5500.202610/shared/team11/envs/avpos_env

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install additional packages
echo "📚 Installing additional packages..."
pip install facenet-pytorch mlflow tqdm kornia pandas matplotlib seaborn scikit-learn psutil

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "🎉 Environment setup complete!"
echo "Environment location: /courses/DS5500.202610/shared/team11/envs/avpos_env"


