#!/bin/bash
# HPC Job Submission Script for Gamma Sweep - Explorer Cluster
# This script submits the 4-parameter gamma sweep to NEU Explorer HPC

echo "=========================================="
echo "NEU Explorer HPC Gamma Sweep Submission"
echo "Course: DS5500.202610 - Team 11"
echo "=========================================="

# Check if we're on the Explorer cluster
if [[ "$HOSTNAME" == *"explorer"* ]] || [[ "$HOSTNAME" == *"login"* ]]; then
    echo "Running on NEU Explorer cluster: $HOSTNAME"
else
    echo "Not on NEU Explorer cluster. Current host: $HOSTNAME"
    echo "Please run this script on the Explorer cluster."
    exit 1
fi

# Check if required files exist
if [ ! -f "hpc_gamma_sweep.sh" ]; then
    echo "hpc_gamma_sweep.sh not found"
    exit 1
fi

if [ ! -f "master_training.py" ]; then
    echo " master_training.py not found"
    exit 1
fi

# Check if team directories exist
if [ ! -d "/courses/DS5500.202610/data/team11" ]; then
    echo "Team11 data directory not found"
    echo "Please create: mkdir -p /courses/DS5500.202610/data/team11"
    exit 1
fi

if [ ! -d "/courses/DS5500.202610/shared/team11" ]; then
    echo " Team11 shared directory not found"
    echo "Please create: mkdir -p /courses/DS5500.202610/shared/team11"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Submit the job array
echo "Submitting gamma sweep job array..."
JOB_ID=$(sbatch hpc_gamma_sweep.sh | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Array tasks: 1-4 (gamma values: 0.05, 0.10, 0.25, 0.50)"
    echo "Partition: courses-gpu"
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "Check logs with:"
    echo "  tail -f logs/gamma_sweep_${JOB_ID}_1.out"
    echo "  tail -f logs/gamma_sweep_${JOB_ID}_2.out"
    echo "  tail -f logs/gamma_sweep_${JOB_ID}_3.out"
    echo "  tail -f logs/gamma_sweep_${JOB_ID}_4.out"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
    echo ""
    echo "Results will be saved to:"
    echo "  /courses/DS5500.202610/shared/team11/results/"
else
    echo "Job submission failed"
    exit 1
fi

echo "=========================================="
echo "Submission complete!"
echo "=========================================="
