#!/bin/bash
#--------------------------------------------------------------
# SLURM batch script for MOF community network analysis
#--------------------------------------------------------------
#SBATCH --job-name=mof_community_analysis
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --output=community_analysis_%j.out
#SBATCH --error=community_analysis_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oms7891@nyu.edu

# Abort on any error, unset var usage, or pipefail
set -euo pipefail

echo "============================================================="
echo "MOF COMMUNITY NETWORK ANALYSIS JOB STARTED: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Requested CPUs      : $SLURM_CPUS_PER_TASK"
echo "============================================================="

#--------------------------------------------------------------
# Environment Setup
#--------------------------------------------------------------
# Use the same Miniconda Python that worked for the clustering step
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

#--------------------------------------------------------------
# Threading & BLAS limits (match settings inside the wrapper)
#--------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Threading limits set to 1 to avoid BLAS oversubscription"

#--------------------------------------------------------------
# Move to code directory and launch analysis
#--------------------------------------------------------------
# Change to the project's Code directory to run the script
cd /scratch/oms7891/Social_Network_Clustering/Code

echo "Launching run_community_analysis.py ..."
echo "Using Python: $(which python)"

# Use srun to benefit from SLURM's resource tracking
# The -u flag ensures unbuffered Python output, which is better for logging
srun python -u run_community_analysis.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "============================================================="
  echo "JOB COMPLETED SUCCESSFULLY: $(date)"
  echo "Results located in: /scratch/oms7891/Social_Network_Clustering/community_analysis_results"
  echo "============================================================="
else
  echo "============================================================="
  echo "JOB FAILED with exit code $EXIT_CODE: $(date)"
  echo "Check community_analysis_${SLURM_JOB_ID}.err file for details."
  echo "============================================================="
fi 