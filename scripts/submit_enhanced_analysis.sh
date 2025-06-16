#!/bin/bash

# ==============================================================================
# Enhanced MOF Analysis Job Submission Script
# ==============================================================================
# This script submits the enhanced MOF community analysis to SLURM
# Usage: ./submit_enhanced_analysis.sh
# ==============================================================================

echo "=========================================="
echo "Enhanced MOF Analysis - Job Submission"
echo "=========================================="

# Set the working directory
WORK_DIR="/scratch/oms7891/Social_Network_Clustering"
cd "$WORK_DIR"

# SLURM script location
SLURM_SCRIPT="scripts/run_enhanced_mof_analysis.slurm"

# Check if SLURM script exists
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "❌ ERROR: SLURM script not found: $SLURM_SCRIPT"
    echo "Make sure you're in the correct directory: $WORK_DIR"
    exit 1
fi

echo "✓ SLURM script found: $SLURM_SCRIPT"

# Check if data file exists
DATA_FILE="/scratch/oms7891/selected data/selected_data.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "❌ WARNING: Data file not found: $DATA_FILE"
    echo "Please verify the data file path in the SLURM script"
fi

# Submit the job
echo "Submitting enhanced MOF analysis job..."
echo "----------------------------------------"

JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')

if [[ -n "$JOB_ID" ]]; then
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  Script: $SLURM_SCRIPT"
    echo ""
    echo "Monitor your job with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "Check job output:"
    echo "  tail -f enhanced_mof_analysis_${JOB_ID}.out"
    echo "  tail -f enhanced_mof_analysis_${JOB_ID}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "❌ Job submission failed!"
    echo "Check SLURM configuration and try again"
    exit 1
fi

echo "=========================================="
echo "Enhanced MOF Analysis Job Submitted"
echo "Job ID: $JOB_ID"
echo "==========================================" 