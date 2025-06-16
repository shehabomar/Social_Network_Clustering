#!/usr/bin/env python3

"""
Complete Enhanced MOF Community Analysis Pipeline
This script runs all the enhanced features systematically
"""

import os
import sys
import subprocess
import argparse
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete Enhanced MOF Community Analysis')
    parser.add_argument('--data_file', required=True, help='Path to MOF data CSV file')
    parser.add_argument('--output_dir', default='enhanced_results', help='Output directory')
    parser.add_argument('--create_slurm', action='store_true', help='Create SLURM script only')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED MOF COMMUNITY ANALYSIS PIPELINE")
    print("="*80)
    print("Features: Multi-threshold, Enhanced conductance, Outlier filtering")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Enhanced Analysis
    if not args.create_slurm:
        enhanced_script = os.path.join(base_dir, "community_network_analysis.py")
        cmd = [
            sys.executable, enhanced_script,
            "--data_file", args.data_file,
            "--output_dir", args.output_dir,
            "--thresholds", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95",
            "--algorithms", "louvain", "girvan_newman"
        ]
        
        if not run_command(cmd, "Enhanced multi-threshold analysis"):
            return False
        
        # Step 2: Outlier Filtering
        filtering_script = os.path.join(base_dir, "create_filtered_analysis.py")
        cmd_filter = [
            sys.executable, filtering_script,
            "--results_dir", args.output_dir,
            "--thresholds", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95",
            "--algorithms", "louvain", "girvan_newman"
        ]
        
        if not run_command(cmd_filter, "Outlier filtering analysis"):
            print("Warning: Filtering analysis failed")
    
    # Create SLURM script
    slurm_script = os.path.join(args.output_dir, "run_enhanced_mof_analysis.sh")
    os.makedirs(args.output_dir, exist_ok=True)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=mof_enhanced_analysis
#SBATCH --partition=serial
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output={args.output_dir}/slurm_output_%j.log

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

cd {base_dir}

python3 community_network_analysis.py \\
    --data_file "{args.data_file}" \\
    --output_dir "{args.output_dir}" \\
    --thresholds 0.7 0.75 0.8 0.85 0.9 0.95 \\
    --algorithms louvain girvan_newman

python3 create_filtered_analysis.py \\
    --results_dir "{args.output_dir}" \\
    --thresholds 0.7 0.75 0.8 0.85 0.9 0.95 \\
    --algorithms louvain girvan_newman

echo "Enhanced MOF analysis completed at $(date)"
"""
    
    with open(slurm_script, 'w') as f:
        f.write(slurm_content)
    os.chmod(slurm_script, 0o755)
    
    print(f"\n✓ SLURM script created: {slurm_script}")
    
    if args.create_slurm:
        print("\nTo submit to HPV:")
        print(f"sbatch {slurm_script}")
    else:
        print("\n🎉 Enhanced analysis pipeline completed!")
        print(f"Results in: {args.output_dir}")

if __name__ == "__main__":
    main() 