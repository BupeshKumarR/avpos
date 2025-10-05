#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --array=1-4
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gamma_sweep_%A_%a.out
#SBATCH --error=logs/gamma_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rameskumar.b@northeastern.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Define gamma values for each array task
declare -a GAMMA_VALUES=(0.05 0.10 0.25 0.50)
GAMMA=${GAMMA_VALUES[$((SLURM_ARRAY_TASK_ID-1))]}

# Define output directory for this gamma value
OUTPUT_DIR="/courses/DS5500.202610/shared/team11/results/gamma_${GAMMA}"

echo "=========================================="
echo "Starting HPC Gamma Sweep Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Gamma Value: $GAMMA"
echo "Output Directory: $OUTPUT_DIR"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load modules for Explorer
module load anaconda3/2024.06
module load cuda/12.0

# Create conda environment in course storage if it doesn't exist
ENV_PATH="/courses/DS5500.202610/shared/team11/envs/avpos_env"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda environment..."
    conda create --prefix $ENV_PATH python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia -y
    conda activate $ENV_PATH
    pip install facenet-pytorch mlflow tqdm kornia pandas matplotlib seaborn scikit-learn psutil
else
    echo "Activating existing conda environment..."
    source activate $ENV_PATH
fi

# Verify GPU availability
nvidia-smi

# Copy dataset to scratch for faster I/O
echo "Copying dataset to scratch for faster processing..."
cp -r /courses/DS5500.202610/data/team11/VGG-Face2 /scratch/$USER/
DATASET_PATH=/courses/DS5500.202610/data/team11/VGG-Face2

# Run the training with validated parameters
python master_training.py \
    --dataset vggface2 \
    --data_root "$DATASET_PATH" \
    --max_classes 100 \
    --max_per_class 50 \
    --batch_size 32 \
    --num_utility_classes 100 \
    --num_identities 100 \
    --pretrain_epochs 5 \
    --joint_epochs 200 \
    --alpha 1.0 \
    --beta 10.0 \
    --gamma $GAMMA \
    --lr_obfuscator 5e-4 \
    --lr_adversary 1e-5 \
    --adv_steps 1 \
    --obfuscator_type feature \
    --obf_base 64 \
    --save_every 10 \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --target_utility_accuracy 0.9 \
    --target_privacy_accuracy 0.0 \
    --initial_budget 0.5 \
    --min_budget 0.1 \
    --max_budget 0.9

echo "=========================================="
echo "Job completed for gamma = $GAMMA"
echo "Final results saved to: $OUTPUT_DIR"
echo "Cleaning up scratch directory..."
rm -rf /scratch/$USER/VGG-Face2
echo "=========================================="
