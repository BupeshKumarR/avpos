#!/bin/bash
# Dataset upload script for VGG-Face2 to Explorer cluster
# Run this from your local machine

echo "=========================================="
echo "VGG-Face2 Dataset Upload to Explorer"
echo "Course: DS5500.202610 - Team 11"
echo "=========================================="

# Configuration
USERNAME="rameskumar.b"
TRANSFER_NODE="xfer.discovery.neu.edu"
LOCAL_DATASET_PATH="./VGG-Face2"
REMOTE_DATASET_PATH="/courses/DS5500.202610/data/team11/VGG-Face2"

# Check if local dataset exists
if [ ! -d "$LOCAL_DATASET_PATH" ]; then
    echo " Local dataset not found at: $LOCAL_DATASET_PATH"
    echo "Please ensure VGG-Face2 directory exists in current location"
    exit 1
fi

echo "✅ Local dataset found: $LOCAL_DATASET_PATH"
echo "📊 Dataset size: $(du -sh $LOCAL_DATASET_PATH | cut -f1)"

# Check dataset structure
echo "Checking dataset structure..."
if [ ! -d "$LOCAL_DATASET_PATH/train" ] || [ ! -d "$LOCAL_DATASET_PATH/test" ]; then
    echo "Invalid dataset structure. Expected train/ and test/ directories"
    exit 1
fi

echo "Dataset structure looks correct"

# Upload using rsync (recommended for large datasets)
echo "Starting upload using rsync..."
echo "This may take 30-60 minutes depending on your internet speed..."

rsync -avz --progress \
    --exclude='*.tmp' \
    --exclude='*.log' \
    --exclude='__pycache__/' \
    "$LOCAL_DATASET_PATH/" \
    "$USERNAME@$TRANSFER_NODE:$REMOTE_DATASET_PATH/"

if [ $? -eq 0 ]; then
    echo "Dataset upload completed successfully!"
    echo "Dataset location: $REMOTE_DATASET_PATH"
    echo ""
    echo "Next steps:"
    echo "1. SSH to Explorer: ssh $USERNAME@login.explorer.northeastern.edu"
    echo "2. Run setup script: ./setup_team11_dirs.sh"
    echo "3. Upload code: scp -r avpos/ $USERNAME@$TRANSFER_NODE:~/"
    echo "4. Submit jobs: ./submit_hpc_jobs.sh"
else
    echo "Upload failed. Please check your connection and try again."
    exit 1
fi

echo "=========================================="
