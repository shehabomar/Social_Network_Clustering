#!/bin/bash
#SBATCH --job-name=mof_clustering
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=mof_clustering_%j.out
#SBATCH --error=mof_clustering_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oms7891@nyu.edu

# Use system Python
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip first
python -m pip install --upgrade pip

# Install packages if needed
echo "Installing required packages..."
pip install numpy pandas scikit-learn networkx matplotlib tqdm python-louvain seaborn scipy

# Create output directory
mkdir -p results

# Run the full MOF clustering analysis
echo "Starting MOF clustering analysis at $(date)"
python Code/mof_social_network.py \
    --input "/scratch/oms7891/selected data/selected_data.csv" \
    --output_dir "results" \
    --threshold 0.85 \
    --resolution 1.0 \
    --algorithm girvan_newman \
    --batch_size 2000

echo "MOF clustering analysis completed at $(date)"

# Show some basic statistics about the results
if [ -f "results/louvain_community_analysis.csv" ]; then
    echo "Number of MOFs processed (Louvain):"
    wc -l results/louvain_community_analysis.csv

    echo "Number of unique communities (Louvain):"
    tail -n +2 results/louvain_community_analysis.csv | cut -d',' -f2 | sort -u | wc -l
fi

if [ -f "results/girvan_newman_community_analysis.csv" ]; then
    echo "Number of MOFs processed (Girvan-Newman):"
    wc -l results/girvan_newman_community_analysis.csv
    
    echo "Number of unique communities (Girvan-Newman):"
    tail -n +2 results/girvan_newman_community_analysis.csv | cut -d',' -f2 | sort -u | wc -l
fi