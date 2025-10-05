# NEU Explorer HPC Deployment Instructions
# Course: DS5500.202610 - Team 11
# Project: Adaptive Privacy-Preserving Face Obfuscation

## Prerequisites
- NEU credentials (username: rameskumar.b)
- VGG-Face2 dataset (~90GB)
- Local project code

## Step 1: Upload Dataset (Run from Local Machine)
```bash
# Make upload script executable
chmod +x upload_dataset.sh

# Upload VGG-Face2 dataset (30-60 minutes)
./upload_dataset.sh
```

## Step 2: Connect to Explorer Cluster
```bash
# SSH to Explorer
ssh rameskumar.b@login.explorer.northeastern.edu
```

## Step 3: Setup Team Directories (Run on Explorer)
```bash
# Upload and run setup script
scp setup_team11_dirs.sh rameskumar.b@xfer.discovery.neu.edu:~/
ssh rameskumar.b@login.explorer.northeastern.edu
chmod +x setup_team11_dirs.sh
./setup_team11_dirs.sh
```

## Step 4: Upload Project Code (Run from Local Machine)
```bash
# Upload entire project directory
scp -r avpos/ rameskumar.b@xfer.discovery.neu.edu:~/
```

## Step 5: Submit Jobs (Run on Explorer)
```bash
# SSH to Explorer
ssh rameskumar.b@login.explorer.northeastern.edu

# Navigate to project directory
cd avpos

# Make scripts executable
chmod +x *.sh

# Submit gamma sweep jobs
./submit_hpc_jobs.sh
```

## Step 6: Monitor Jobs
```bash
# Check job status
squeue -u rameskumar.b

# Monitor logs
tail -f logs/gamma_sweep_<job_id>_1.out
tail -f logs/gamma_sweep_<job_id>_2.out
tail -f logs/gamma_sweep_<job_id>_3.out
tail -f logs/gamma_sweep_<job_id>_4.out
```

## Expected Results
- **4 parallel jobs** running simultaneously
- **200 epochs each** for deep convergence
- **Results saved to**: `/courses/DS5500.202610/shared/team11/results/`
- **Privacy-Utility trade-off curve** with 4 data points

## Resource Usage
- **Partition**: courses-gpu
- **GPU**: 1 GPU per job (4 total)
- **Memory**: 32GB RAM per job
- **Time**: 24 hours per job
- **Storage**: Results in course shared directory

## Troubleshooting
- **Job fails**: Check logs in `logs/` directory
- **Storage full**: Clean up scratch directories
- **GPU issues**: Verify with `nvidia-smi`
- **Module errors**: Check `module avail` for available software

## Success Criteria
- All 4 jobs complete successfully
- A_acc ≤ 0.80 for all gamma values
- Utility accuracy ≥ 0.90 maintained
- Complete privacy-utility frontier mapped
