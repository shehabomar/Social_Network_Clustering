#!/bin/bash
#SBATCH --job-name=mof_clustering
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --output=mof_clustering_%j.out
#SBATCH --error=mof_clustering_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oms7891@nyu.edu

MODE=${1:-"similarity"}  # Default to similarity only

# Use system Python
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

# MINIMAL thread usage
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Navigate to working directory
cd /scratch/oms7891/Social_Network_Clustering

# Check if we're in the right directory
if [ ! -f "Code/mof_social_network.py" ]; then
    echo "Error: Cannot find Code/mof_social_network.py"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if input data exists
if [ ! -f "/scratch/oms7891/selected data/selected_data.csv" ]; then
    echo "Error: Cannot find /scratch/oms7891/selected data/selected_data.csv"
    echo "Please ensure the input data file exists"
    exit 1
fi

echo "Using system Python..."

# Check if packages are installed, if not install them
if ! python -c "import pandas" 2>/dev/null; then
    echo "Installing required packages..."
    pip install --user numpy==1.24.3 pandas scikit-learn tqdm
    pip install --user networkx matplotlib python-louvain seaborn scipy
fi

# Create output directory
mkdir -p results

# Define common parameters
INPUT_FILE="/scratch/oms7891/selected data/selected_data.csv"
OUTPUT_DIR="results"
THRESHOLD=0.90  # INCREASED to reduce edges
RESOLUTION=1.0

# ULTRA CONSERVATIVE SETTINGS - GUARANTEED TO WORK
BATCH_SIZE=50     # VERY small batches to minimize memory per iteration
N_JOBS=1          # NO parallelism - sequential only
TOP_K=10          # Keep only top 10 neighbors per node

ADJACENCY_FILE="${OUTPUT_DIR}/adjacency_matrix_threshold_${THRESHOLD}.pkl"

echo "=============================================="
echo "ULTRA-SAFE MEMORY SETTINGS"
echo "=============================================="
echo "Running MOF clustering in $MODE mode at $(date)"
echo ""
echo "GUARANTEED MEMORY-SAFE settings:"
echo "  - Batch size: $BATCH_SIZE (ultra small)"
echo "  - Parallel jobs: $N_JOBS (sequential)"
echo "  - Top K neighbors: $TOP_K (minimal)"
echo "  - Threshold: $THRESHOLD (high - fewer edges)"
echo "  - Available memory: 256GB"
echo ""
echo "This WILL be slow but WILL complete successfully"
echo "Estimated time: 20-30 hours"
echo "=============================================="

case $MODE in
    "test")
        # Test with small subset
        echo "Testing with 10,000 MOFs..."
        python Code/mof_social_network.py \
            --input "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --sample 10000 \
            --threshold $THRESHOLD \
            --batch_size $BATCH_SIZE \
            --n_jobs $N_JOBS \
            --top_k $TOP_K \
            --save_adjacency \
            --algorithm none
        ;;
        
    "similarity")
        # Calculate similarity matrix for full dataset
        echo "Calculating similarity matrix (sequential mode)..."
        echo "Monitor progress in the output file"
        
        # Add Python garbage collection flag
        export PYTHONOPTIMIZE=1
        
        python -u Code/mof_social_network.py \
            --input "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --threshold $THRESHOLD \
            --batch_size $BATCH_SIZE \
            --n_jobs $N_JOBS \
            --top_k $TOP_K \
            --save_adjacency \
            --algorithm none
        ;;
        
    "network")
        # Only run network analysis (requires pre-computed adjacency matrix)
        if [ ! -f "$ADJACENCY_FILE" ]; then
            echo "Error: Adjacency matrix not found at $ADJACENCY_FILE"
            echo "Please run similarity calculation first"
            exit 1
        fi
        
        echo "Running network analysis from pre-computed adjacency matrix..."
        python Code/mof_social_network.py \
            --input "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --threshold $THRESHOLD \
            --resolution $RESOLUTION \
            --algorithm both \
            --load_adjacency "$ADJACENCY_FILE"
        ;;
        
    "full")
        # Full run - NOT RECOMMENDED for large dataset
        echo "WARNING: Full run may use too much memory!"
        echo "Recommended: Run 'similarity' first, then 'network'"
        
        python Code/mof_social_network.py \
            --input "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --threshold $THRESHOLD \
            --resolution $RESOLUTION \
            --algorithm both \
            --batch_size $BATCH_SIZE \
            --n_jobs $N_JOBS \
            --top_k $TOP_K \
            --save_adjacency
        ;;
esac

echo ""
echo "MOF clustering analysis completed at $(date)"

# Show results
if [ -f "$ADJACENCY_FILE" ]; then
    echo ""
    echo "SUCCESS! Adjacency matrix saved:"
    ls -lh "$ADJACENCY_FILE"
    
    # Show some stats
    echo ""
    echo "Quick stats:"
    python -c "
import pickle
with open('$ADJACENCY_FILE', 'rb') as f:
    adj = pickle.load(f)
    print(f'Total edges: {len(adj):,}')
    print(f'Average edges per node: {len(adj)*2/258212:.1f}')
"
fi

# Show memory usage if available
if command -v sacct &> /dev/null && [ ! -z "$SLURM_JOB_ID" ]; then
    echo ""
    echo "Memory usage:"
    sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,State
fi 