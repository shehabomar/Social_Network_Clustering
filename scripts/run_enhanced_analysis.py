#!/usr/bin/env python3

"""
Enhanced wrapper script for multi-threshold MOF community network analysis
This script is optimized for HPV execution with resource management and checkpointing
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path

def setup_environment():
    """Set up the environment for HPV execution"""
    
    # CRITICAL: Set threading limits to prevent OpenBLAS conflicts
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
    # Use the system Python that worked
    os.environ['PATH'] = "/share/apps/NYUAD5/miniconda/3-4.11.0/bin:" + os.environ.get('PATH', '')
    
    # Add Python optimization flags
    os.environ['PYTHONOPTIMIZE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'  # For better logging
    
    print("Environment configured for HPV execution:")
    print(f"  Threading limit: {os.environ['OMP_NUM_THREADS']}")
    print(f"  Python path: {os.environ['PATH'].split(':')[0]}")

def check_dependencies():
    """Check if required packages are available"""
    print("Checking Python dependencies...")
    
    try:
        import numpy
        import pandas
        import networkx
        import community
        import sklearn
        import matplotlib
        import seaborn
        print("✓ All core packages available")
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        return False

def create_checkpoint_file(checkpoint_dir, completed_thresholds):
    """Create checkpoint file to track progress"""
    checkpoint_file = os.path.join(checkpoint_dir, 'analysis_checkpoint.json')
    checkpoint_data = {
        'completed_thresholds': completed_thresholds,
        'timestamp': time.time(),
        'status': 'in_progress'
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(checkpoint_dir):
    """Load checkpoint file to resume analysis"""
    checkpoint_file = os.path.join(checkpoint_dir, 'analysis_checkpoint.json')
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'completed_thresholds': [], 'status': 'not_started'}

def monitor_resources():
    """Monitor system resources (memory, disk space)"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3)
        }
    except ImportError:
        return {'memory_percent': 0, 'memory_available_gb': 0, 'disk_free_gb': 0}

def run_enhanced_analysis(args):
    """Run the enhanced multi-threshold analysis"""
    
    # Set up environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to set up dependencies")
        return False
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    
    # Input files
    data_file = args.data_file
    
    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoint management
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_dir)
    completed_thresholds = set(checkpoint['completed_thresholds'])
    
    # Enhanced community analysis script
    # Look for the script in multiple possible locations
    possible_paths = [
        os.path.join(project_dir, "src", "community_network_analysis.py"),
        os.path.join(project_dir, "Code", "community_network_analysis.py"),
        os.path.join(base_dir, "community_network_analysis.py")
    ]
    
    analysis_script = None
    for path in possible_paths:
        if os.path.exists(path):
            analysis_script = path
            break
    
    if not analysis_script:
        print("Error: community_network_analysis.py not found in any expected location")
        return False
    
    # Check if input files exist
    if not os.path.exists(data_file):
        print(f"Error: Missing data file: {data_file}")
        return False
    
    print("✓ All input files found")
    
    # Use the system Python that worked for clustering
    python_executable = "/share/apps/NYUAD5/miniconda/3-4.11.0/bin/python"
    
    print(f"Starting enhanced multi-threshold analysis...")
    print(f"Thresholds to analyze: {args.thresholds}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Output directory: {output_dir}")
    
    # Monitor initial resources
    resources = monitor_resources()
    print(f"Initial resources: {resources['memory_available_gb']:.1f} GB RAM, {resources['disk_free_gb']:.1f} GB disk")
    
    # Create command for the enhanced analysis
    cmd = [
        python_executable, analysis_script,
        "--data_file", data_file,
        "--output_dir", output_dir,
        "--thresholds"] + [str(t) for t in args.thresholds] + [
        "--algorithms"] + args.algorithms + [
        "--resolution", str(args.resolution)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the enhanced analysis
    start_time = time.time()
    
    try:
        # Run with environment variables set and capture output
        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress and save logs
        log_file = os.path.join(output_dir, 'analysis_log.txt')
        with open(log_file, 'w') as log:
            for line in process.stdout:
                print(line.strip())
                log.write(line)
                log.flush()
                
                # Check for threshold completion
                if "Threshold" in line and "analysis completed!" in line:
                    # Extract threshold from log message
                    try:
                        threshold = float(line.split("Threshold")[1].split("analysis")[0].strip())
                        completed_thresholds.add(threshold)
                        create_checkpoint_file(checkpoint_dir, list(completed_thresholds))
                    except:
                        pass
        
        process.wait()
        
        if process.returncode == 0:
            # Mark as completed
            checkpoint_data = {
                'completed_thresholds': list(completed_thresholds),
                'timestamp': time.time(),
                'status': 'completed',
                'total_time_hours': (time.time() - start_time) / 3600
            }
            
            with open(os.path.join(checkpoint_dir, 'analysis_checkpoint.json'), 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print("\nEnhanced analysis completed successfully!")
            print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
            print(f"Results saved to: {output_dir}")
            
            # Final resource check
            final_resources = monitor_resources()
            print(f"Final resources: {final_resources['memory_available_gb']:.1f} GB RAM, {final_resources['disk_free_gb']:.1f} GB disk")
            
            return True
        else:
            print(f"Analysis failed with return code: {process.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running enhanced analysis: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check available memory and disk space")
        print("2. Try reducing the number of thresholds or algorithms")
        print("3. Check the analysis log for specific errors")
        return False
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        print(f"Progress saved. Completed thresholds: {list(completed_thresholds)}")
        return False

def create_slurm_job_script(args, output_file):
    """Create a SLURM job script for HPV execution"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=mof_enhanced_analysis
#SBATCH --partition=serial
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output={args.output_dir}/slurm_output_%j.log
#SBATCH --error={args.output_dir}/slurm_error_%j.log

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
cd {os.path.dirname(os.path.abspath(__file__))}

# Run the enhanced analysis
python3 run_enhanced_analysis.py \\
    --data_file "{args.data_file}" \\
    --output_dir "{args.output_dir}" \\
    --thresholds {' '.join(map(str, args.thresholds))} \\
    --algorithms {' '.join(args.algorithms)} \\
    --resolution {args.resolution}

echo "Enhanced MOF analysis job completed at $(date)"
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_file, 0o755)
    print(f"SLURM job script created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-Threshold MOF Community Analysis - HPV Compatible')
    parser.add_argument('--data_file', required=True,
                       help='Path to the MOF data CSV file')
    parser.add_argument('--output_dir', default='enhanced_community_results',
                       help='Output directory for results')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.7, 0.75],
                       help='Similarity thresholds to test')
    parser.add_argument('--algorithms', nargs='+', choices=['louvain', 'girvan_newman'],
                       default=['louvain', 'girvan_newman'],
                       help='Algorithms to run')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Resolution parameter for community detection')
    parser.add_argument('--create_slurm', action='store_true',
                       help='Create SLURM job script instead of running directly')
    parser.add_argument('--slurm_script', default='run_enhanced_mof_analysis.sh',
                       help='Name of SLURM script file to create')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED MOF COMMUNITY NETWORK ANALYSIS - HPV COMPATIBLE")
    print("="*80)
    
    if args.create_slurm:
        # Create SLURM job script
        create_slurm_job_script(args, args.slurm_script)
        print(f"\nSLURM job script created: {args.slurm_script}")
        print("To submit the job, run:")
        print(f"sbatch {args.slurm_script}")
        print("\nTo monitor the job:")
        print("squeue -u $USER")
        print("To cancel the job:")
        print("scancel <job_id>")
    else:
        # Run directly
        success = run_enhanced_analysis(args)
        
        if success:
            print("\n" + "="*80)
            print("ENHANCED COMMUNITY NETWORK ANALYSIS COMPLETED")
            print("="*80)
            print("Generated directory structure:")
            print(f"  {args.output_dir}/")
            print("  ├── threshold_analysis/")
            print("  │   ├── louvain_t0.7/")
            print("  │   ├── girvan_newman_t0.7/")
            print("  │   └── ... (for each threshold)")
            print("  ├── comparison_analysis/")
            print("  │   ├── threshold_algorithm_comparison.csv")
            print("  │   ├── performance_ranking.csv")
            print("  │   └── threshold_comparison_plots.png")
            print("  ├── checkpoints/")
            print("  └── all_results_summary.json")
        else:
            print("\n" + "="*80)
            print("ENHANCED ANALYSIS FAILED")
            print("="*80)
            print("Check the error messages above and:")
            print("1. Verify input file paths")
            print("2. Check available system resources")
            print("3. Review the analysis log for details")
            print("4. Consider using SLURM for HPV execution:")
            print(f"   python3 {sys.argv[0]} --create_slurm [other args]")

if __name__ == "__main__":
    main() 