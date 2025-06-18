#!/bin/bash

# ==============================================================================
# Enhanced MOF Analysis Job Submission Script
# ==============================================================================
# This script generates and submits the enhanced MOF community analysis to SLURM
# Usage: ./submit_enhanced_analysis.sh [options]
# ==============================================================================

echo "=========================================="
echo "Enhanced MOF Analysis - Job Submission"
echo "=========================================="

# Set the working directory
WORK_DIR="/scratch/oms7891/Social_Network_Clustering"
cd "$WORK_DIR"

# Default parameters
DATA_FILE="/scratch/oms7891/selected data/selected_data.csv"
OUTPUT_DIR="enhanced_community_results"
THRESHOLDS="0.7 0.75"
ALGORITHMS="louvain girvan_newman"
RESOLUTION="1.0"
SLURM_SCRIPT="scripts/run_enhanced_mof_analysis.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

echo "✓ Data file found: $DATA_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Thresholds: $THRESHOLDS"
echo "  Algorithms: $ALGORITHMS"
echo "  Resolution: $RESOLUTION"

# First, generate the SLURM script using run_enhanced_analysis.py
echo ""
echo "Generating SLURM job script..."
echo "----------------------------------------"

python3 scripts/run_enhanced_analysis.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --thresholds $THRESHOLDS \
    --algorithms $ALGORITHMS \
    --resolution $RESOLUTION \
    --create_slurm \
    --slurm_script "$SLURM_SCRIPT"

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to generate SLURM script!"
    exit 1
fi

# Check if SLURM script was created
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "❌ ERROR: SLURM script not created: $SLURM_SCRIPT"
    exit 1
fi

echo "✓ SLURM script generated: $SLURM_SCRIPT"

# Submit the job
echo ""
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
    echo "  tail -f $OUTPUT_DIR/slurm_output_${JOB_ID}.log"
    echo "  tail -f $OUTPUT_DIR/slurm_error_${JOB_ID}.log"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "❌ Job submission failed!"
    echo "Check SLURM configuration and try again"
    exit 1
fi

echo ""
echo "=========================================="
echo "Enhanced MOF Analysis Job Submitted"
echo "Job ID: $JOB_ID"
echo "==========================================" 