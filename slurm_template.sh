#!/bin/bash
#SBATCH --job-name=avpos_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=33:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Load modules
module load python/3.12
module load cuda/11.8

# Activate environment
source activate avpos

# Run training
python master_training.py \
    --dataset vggface2 \
    --data_root ${SLURM_SUBMIT_DIR}/data \
    --batch_size 64 \
    --joint_epochs 100 \
    --output_dir ${SLURM_SUBMIT_DIR}/results/run_${SLURM_JOB_ID}
