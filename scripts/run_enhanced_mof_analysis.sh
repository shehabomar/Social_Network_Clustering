#!/bin/bash
#SBATCH --job-name=mof_enhanced_analysis
#SBATCH --partition=serial
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=enhanced_community_results/slurm_output_%j.log
#SBATCH --error=enhanced_community_results/slurm_error_%j.log

# Load modules if needed
# module load python/3.9

# Set environment variables
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Add Python path
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

# Change to the working directory
cd /scratch/oms7891/Social_Network_Clustering/scripts

# Run the enhanced analysis
python3 run_enhanced_analysis.py \
    --data_file "/scratch/oms7891/selected data/selected_data.csv" \
    --output_dir "enhanced_community_results" \
    --thresholds 0.7 0.75 \
    --algorithms louvain girvan_newman \
    --resolution 1.0

echo "Enhanced MOF analysis job completed at $(date)"
