#!/bin/bash
# HPC Job Monitoring Script for Gamma Sweep
# This script monitors the progress of the gamma sweep jobs

echo "=========================================="
echo "NEU HPC Gamma Sweep Monitoring"
echo "=========================================="

# Get job ID from user or find the most recent gamma_sweep job
if [ -z "$1" ]; then
    JOB_ID=$(squeue -u $USER --format="%.10i %.20j" | grep gamma_sweep | head -1 | awk '{print $1}')
    if [ -z "$JOB_ID" ]; then
        echo "❌ No gamma_sweep jobs found. Please provide job ID:"
        echo "Usage: $0 <job_id>"
        exit 1
    fi
    echo "Found gamma_sweep job: $JOB_ID"
else
    JOB_ID=$1
fi

echo "Monitoring job: $JOB_ID"
echo ""

# Function to check job status
check_job_status() {
    echo "=== Job Status ==="
    squeue -j $JOB_ID
    echo ""
    
    echo "=== GPU Usage ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
}

# Function to show recent logs
show_logs() {
    echo "=== Recent Log Output ==="
    for i in {1..4}; do
        LOG_FILE="logs/gamma_sweep_${JOB_ID}_${i}.out"
        if [ -f "$LOG_FILE" ]; then
            echo "--- Task $i (gamma=${GAMMA_VALUES[$((i-1))]}) ---"
            tail -5 "$LOG_FILE" 2>/dev/null || echo "No recent output"
            echo ""
        fi
    done
}

# Function to check results
check_results() {
    echo "=== Results Check ==="
    for gamma in 0.05 0.10 0.25 0.50; do
        RESULT_DIR="results/hpc_gamma_${gamma}"
        if [ -d "$RESULT_DIR" ]; then
            echo "✅ Gamma $gamma: $RESULT_DIR"
            if [ -f "$RESULT_DIR/training_history.json" ]; then
                echo "   Training history: ✅"
            fi
            if [ -f "$RESULT_DIR/best_obfuscator.pth" ]; then
                echo "   Best model: ✅"
            fi
        else
            echo "⏳ Gamma $gamma: Not started"
        fi
    done
    echo ""
}

# Main monitoring loop
GAMMA_VALUES=(0.05 0.10 0.25 0.50)

while true; do
    clear
    echo "=========================================="
    echo "NEU HPC Gamma Sweep Monitoring"
    echo "Job ID: $JOB_ID"
    echo "Time: $(date)"
    echo "=========================================="
    
    check_job_status
    show_logs
    check_results
    
    echo "Press Ctrl+C to exit monitoring"
    echo "Refreshing in 30 seconds..."
    
    sleep 30
done
