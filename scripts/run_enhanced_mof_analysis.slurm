#!/bin/bash
#SBATCH --job-name=enhanced_mof_analysis
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=enhanced_mof_analysis_%j.out
#SBATCH --error=enhanced_mof_analysis_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oms7891@nyu.edu

# ==============================================================================
# Enhanced MOF Social Network Analysis - HPC Script
# ==============================================================================
# This script runs the complete enhanced analysis pipeline including:
# - Multi-threshold analysis (6 thresholds)
# - Enhanced conductance calculation  
# - Girvan-Newman algorithm testing
# - Outlier filtering analysis
# - Comprehensive comparisons
# ==============================================================================

echo "=========================================="
echo "Enhanced MOF Social Network Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# ==============================================================================
# CRITICAL: Environment Setup to Fix OpenBLAS Threading Issues
# ==============================================================================

# Fix OpenBLAS threading issues (addresses the segfault you encountered)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NPROC_LIMIT=4

# Python optimization
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Memory settings
export OMP_STACKSIZE=2G
export KMP_STACKSIZE=2G

# Use the system Python that works
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

echo "Environment configured:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "  Python: $(which python3)"
echo "  Available memory: ${SLURM_MEM_PER_NODE}GB"
echo "  CPUs: $SLURM_CPUS_PER_TASK"

# ==============================================================================
# Directory Setup
# ==============================================================================

# Set working directory
WORK_DIR="/scratch/oms7891/Social_Network_Clustering"
cd $WORK_DIR

echo "Working directory: $(pwd)"

# Data file location
DATA_FILE="/scratch/oms7891/selected data/selected_data.csv"

# Output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${WORK_DIR}/results/enhanced_analysis_${TIMESTAMP}"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p ${WORK_DIR}/analysis/enhanced_${TIMESTAMP}
mkdir -p ${WORK_DIR}/results/logs

echo "Output directory: $OUTPUT_DIR"
echo "Data file: $DATA_FILE"

# ==============================================================================
# Verify Dependencies and Data
# ==============================================================================

echo "----------------------------------------"
echo "Verifying dependencies and data files..."
echo "----------------------------------------"

# Check if data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

echo "✓ Data file found: $DATA_FILE ($(ls -lh "$DATA_FILE" | cut -d' ' -f5))"

# Test Python environment
echo "Testing Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__} loaded successfully')
except Exception as e:
    print(f'❌ NumPy import failed: {e}')
    sys.exit(1)

try:
    import pandas as pd
    print(f'✓ Pandas {pd.__version__} loaded successfully')
except Exception as e:
    print(f'❌ Pandas import failed: {e}')
    sys.exit(1)

try:
    import networkx as nx
    print(f'✓ NetworkX {nx.__version__} loaded successfully')
except Exception as e:
    print(f'❌ NetworkX import failed: {e}')
    sys.exit(1)

try:
    import community
    print(f'✓ Community (python-louvain) loaded successfully')
except Exception as e:
    print(f'❌ Community import failed: {e}')
    sys.exit(1)

print('✓ All dependencies verified')
"

if [[ $? -ne 0 ]]; then
    echo "❌ Dependency check failed"
    exit 1
fi

# ==============================================================================
# Step 1: Enhanced Multi-Threshold Analysis
# ==============================================================================

echo "=========================================="
echo "STEP 1: Enhanced Multi-Threshold Analysis"
echo "=========================================="

STEP1_START=$(date +%s)

python3 src/community_network_analysis.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --thresholds 0.7 0.75 0.8 0.85 0.9 0.95 \
    --algorithms louvain girvan_newman \
    --resolution 1.0

STEP1_EXIT_CODE=$?
STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))

if [[ $STEP1_EXIT_CODE -eq 0 ]]; then
    echo "✓ Step 1 completed successfully in ${STEP1_DURATION} seconds"
else
    echo "❌ Step 1 failed with exit code $STEP1_EXIT_CODE"
    echo "Check the error messages above for details"
    exit $STEP1_EXIT_CODE
fi

# ==============================================================================
# Step 2: Outlier Filtering Analysis
# ==============================================================================

echo "=========================================="
echo "STEP 2: Outlier Filtering Analysis"
echo "=========================================="

STEP2_START=$(date +%s)

python3 utils/create_filtered_analysis.py \
    --results_dir "$OUTPUT_DIR" \
    --thresholds 0.7 0.75 0.8 0.85 0.9 0.95 \
    --algorithms louvain girvan_newman \
    --min_size 10 \
    --max_conductance 0.1 \
    --degree_zscore_threshold 2.0 \
    --min_density 0.01

STEP2_EXIT_CODE=$?
STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

if [[ $STEP2_EXIT_CODE -eq 0 ]]; then
    echo "✓ Step 2 completed successfully in ${STEP2_DURATION} seconds"
else
    echo "❌ Step 2 failed with exit code $STEP2_EXIT_CODE"
    echo "Warning: Continuing without filtered analysis"
fi

# ==============================================================================
# Step 3: Copy Results to Analysis Directory
# ==============================================================================

echo "=========================================="
echo "STEP 3: Organizing Results"
echo "=========================================="

# Copy key results to analysis directory
ANALYSIS_DIR="${WORK_DIR}/analysis/enhanced_${TIMESTAMP}"

if [[ -d "$OUTPUT_DIR/comparison_analysis" ]]; then
    cp -r "$OUTPUT_DIR/comparison_analysis" "$ANALYSIS_DIR/"
    echo "✓ Comparison analysis copied to $ANALYSIS_DIR"
fi

if [[ -f "$OUTPUT_DIR/all_results_summary.json" ]]; then
    cp "$OUTPUT_DIR/all_results_summary.json" "$ANALYSIS_DIR/"
    echo "✓ Results summary copied to $ANALYSIS_DIR"
fi

if [[ -d "$OUTPUT_DIR/filtered_results" ]]; then
    cp -r "$OUTPUT_DIR/filtered_results" "$ANALYSIS_DIR/"
    echo "✓ Filtered results copied to $ANALYSIS_DIR"
fi

# Create a final summary report
SUMMARY_FILE="$ANALYSIS_DIR/FINAL_ANALYSIS_REPORT.md"

cat > "$SUMMARY_FILE" << EOF
# Enhanced MOF Social Network Analysis Report

## Job Information
- **Job ID**: $SLURM_JOB_ID  
- **Start Time**: $(date)
- **Node**: $SLURMD_NODENAME
- **Data File**: $DATA_FILE
- **Output Directory**: $OUTPUT_DIR

## Analysis Components
1. **Multi-Threshold Analysis**: 6 thresholds tested
2. **Enhanced Conductance**: Proper φ(S) = cut(S) / min(vol(S), vol(V\\S)) formula
3. **Girvan-Newman Enhancement**: Modularity-based stopping criterion
4. **Outlier Filtering**: Statistical outlier removal
5. **Comprehensive Comparison**: Cross-method analysis

## Execution Summary
- **Step 1 Duration**: ${STEP1_DURATION} seconds
- **Step 2 Duration**: ${STEP2_DURATION} seconds
- **Total Duration**: $((STEP1_DURATION + STEP2_DURATION)) seconds

## Key Output Files
- \`comparison_analysis/threshold_algorithm_comparison.csv\`: Main comparison table
- \`comparison_analysis/performance_ranking.csv\`: Performance rankings
- \`filtered_results/filtering_summary.csv\`: Outlier analysis summary
- \`all_results_summary.json\`: Complete results in JSON format

## Literature Comparison Targets (Jalali et al. 2023)
- **Community Count**: Target ~246 (Girvan-Newman)
- **Mean Degree**: Target ~19.256
- **Threshold Performance**: 0.7 expected to outperform 0.9
- **Modularity**: Target >0.9

## Next Steps
1. Review performance ranking to identify best threshold-algorithm combination
2. Compare community counts against literature targets
3. Analyze conductance improvements from outlier filtering
4. Validate threshold effectiveness (0.7 vs 0.9 performance)

Analysis completed successfully!
EOF

echo "✓ Final summary report created: $SUMMARY_FILE"

# ==============================================================================
# Cleanup and Final Status
# ==============================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - SLURM_JOB_START_TIME))

echo "=========================================="
echo "Enhanced MOF Analysis Completed"
echo "=========================================="
echo "Total execution time: ${TOTAL_DURATION} seconds ($((TOTAL_DURATION/3600))h $((TOTAL_DURATION%3600/60))m)"
echo "Results location: $OUTPUT_DIR"
echo "Analysis summary: $ANALYSIS_DIR"
echo "Job completed at: $(date)"

# Display final directory sizes
echo "----------------------------------------"
echo "Final output sizes:"
du -sh "$OUTPUT_DIR" 2>/dev/null || echo "Output directory size calculation failed"
du -sh "$ANALYSIS_DIR" 2>/dev/null || echo "Analysis directory size calculation failed"

echo "=========================================="
echo "🎉 Enhanced MOF Social Network Analysis Complete!"
echo "Check the results in: $ANALYSIS_DIR"
echo "==========================================" 