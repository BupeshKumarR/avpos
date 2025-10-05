#!/bin/bash

# NEU HPC Setup Script for Explorer Cluster
# This script sets up the environment using proper HPC practices

echo "=== NEU HPC Setup for Privacy-Preserving Video Processing ==="
echo "Cluster: Explorer"
echo "Course: DS5500.202610"
echo "Team: team11"
echo ""

# Check if we're on the correct cluster
if [[ "$HOSTNAME" != *"explorer"* ]]; then
    echo "❌ ERROR: This script is designed for Explorer cluster"
    echo "Current hostname: $HOSTNAME"
    echo "Please connect to Explorer: ssh username@login.explorer.northeastern.edu"
    exit 1
fi

echo "✅ Connected to Explorer cluster"

# Create team directories
echo "📁 Creating team directories..."
mkdir -p /courses/DS5500.202610/data/team11
mkdir -p /courses/DS5500.202610/shared/team11/envs
mkdir -p /courses/DS5500.202610/shared/team11/results
mkdir -p /courses/DS5500.202610/shared/team11/logs

echo "✅ Team directories created"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "📦 Loading conda module..."
    module load anaconda3/2022.05
fi

# Create conda environment in interactive session
echo "🚀 Starting interactive session for conda installation..."
echo "This will request an interactive session with GPU access for installation"
echo ""

# Submit interactive job for environment setup
cat > setup_env_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/courses/DS5500.202610/shared/team11/logs/setup_env_%j.out

echo "=== Interactive Environment Setup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
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
EOF

# Submit the environment setup job
echo "📤 Submitting environment setup job..."
sbatch setup_env_job.sh

echo ""
echo "⏳ Environment setup job submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs with: tail -f /courses/DS5500.202610/shared/team11/logs/setup_env_*.out"
echo ""
echo "Once the environment is ready, run: ./submit_hpc_jobs.sh"
echo ""

# Clean up temporary file
rm -f setup_env_job.sh


