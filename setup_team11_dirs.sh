#!/bin/bash
# Setup script for Team 11 directories on Explorer cluster
# Run this script after logging into Explorer

echo "=========================================="
echo "Setting up Team 11 directories on Explorer"
echo "Course: DS5500.202610"
echo "=========================================="

# Create team directories
echo "Creating team directories..."
mkdir -p /courses/DS5500.202610/data/team11
mkdir -p /courses/DS5500.202610/shared/team11/results
mkdir -p /courses/DS5500.202610/shared/team11/envs
mkdir -p /courses/DS5500.202610/shared/team11/logs

echo "✅ Directories created:"
echo "  Data: /courses/DS5500.202610/data/team11"
echo "  Shared: /courses/DS5500.202610/shared/team11"
echo "  Results: /courses/DS5500.202610/shared/team11/results"
echo "  Environments: /courses/DS5500.202610/shared/team11/envs"
echo "  Logs: /courses/DS5500.202610/shared/team11/logs"

# Set permissions
echo "Setting permissions..."
chmod 755 /courses/DS5500.202610/data/team11
chmod 755 /courses/DS5500.202610/shared/team11
chmod 755 /courses/DS5500.202610/shared/team11/results
chmod 755 /courses/DS5500.202610/shared/team11/envs
chmod 755 /courses/DS5500.202610/shared/team11/logs

echo "✅ Permissions set"

# Check available space
echo "Checking available space..."
df -h /courses/DS5500.202610/

echo "=========================================="
echo "Setup complete! Ready for deployment."
echo "Next steps:"
echo "1. Upload dataset to /courses/DS5500.202610/data/team11/"
echo "2. Upload code to your home directory"
echo "3. Run submit_hpc_jobs.sh"
echo "=========================================="
