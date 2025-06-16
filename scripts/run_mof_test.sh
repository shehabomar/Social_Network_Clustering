#!/bin/bash
#SBATCH --job-name=mof_test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1024GB
#SBATCH --time=48:00:00
#SBATCH --output=mof_test_%j.out
#SBATCH --error=mof_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oms7891@nyu.edu

# Use system Python
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

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
pip install numpy pandas scikit-learn networkx matplotlib tqdm python-louvain seaborn

# Create test output directory
mkdir -p test_results

echo "Starting MOF test run at $(date)"
python Code/mof_social_network.py \
    --input "/scratch/oms7891/selected data/selected_data.csv" \
    --output_dir "test_results" \
    --sample 258212 \
    --threshold 0.85 \
    --resolution 1.0

echo "MOF test completed at $(date)"