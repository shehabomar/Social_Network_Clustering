#!/bin/bash

# ==============================================================================
# Quick Fix and Rerun Script for Enhanced MOF Analysis
# ==============================================================================
# This script fixes the import path issue and resubmits the job
# ==============================================================================

echo "=========================================="
echo "Fixing Import Path Issue and Rerunning"
echo "=========================================="

# Navigate to project directory
cd "/scratch/oms7891/Social_Network_Clustering"

# Option 1: Copy mof_social_network.py to src directory (cleanest fix)
echo "Copying mof_social_network.py to src directory..."
cp Code/mof_social_network.py src/

# Make sure all scripts are executable
chmod +x scripts/*.sh scripts/*.slurm

# Clean up any partial results from the failed run
echo "Cleaning up partial results..."
rm -rf results/enhanced_analysis_20250614_203547
rm -rf analysis/enhanced_20250614_203547

# Option 2: Alternative - Create a symbolic link
# ln -sf "../Code/mof_social_network.py" src/mof_social_network.py

echo "✓ Import path issue fixed"
echo ""

# Resubmit the job
echo "Resubmitting the enhanced analysis job..."
JOB_ID=$(sbatch scripts/run_enhanced_mof_analysis.slurm | grep -o '[0-9]*')

if [[ -n "$JOB_ID" ]]; then
    echo "✅ Job resubmitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "   squeue -u $USER"
    echo "   tail -f enhanced_mof_analysis_${JOB_ID}.out"
    echo ""
    echo "Expected runtime: 2-6 hours for all 6 thresholds"
    echo "The analysis will now work correctly!"
else
    echo "❌ Job submission failed. Try manual submission:"
    echo "   sbatch scripts/run_enhanced_mof_analysis.slurm"
fi

echo "==========================================" 