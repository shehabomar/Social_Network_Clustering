#!/usr/bin/env python3

"""
Fixed wrapper script to run community network analysis
This script includes the same environment setup that worked for the main clustering
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up the same environment variables that worked for the main clustering"""
    
    # CRITICAL: Set threading limits to prevent OpenBLAS conflicts
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Use the same Python path that worked
    os.environ['PATH'] = "/share/apps/NYUAD5/miniconda/3-4.11.0/bin:" + os.environ.get('PATH', '')
    
    # Add Python optimization flag
    os.environ['PYTHONOPTIMIZE'] = '1'
    
    print("Environment configured with threading limits:")
    print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print(f"  OPENBLAS_NUM_THREADS: {os.environ['OPENBLAS_NUM_THREADS']}")
    print(f"  Python path: {os.environ['PATH'].split(':')[0]}")

def check_dependencies():
    """Check if required packages are available, install if needed"""
    print("Checking Python dependencies...")
    
    required_packages = [
        'numpy==1.24.3',
        'pandas', 
        'scikit-learn', 
        'tqdm',
        'networkx', 
        'matplotlib', 
        'python-louvain', 
        'seaborn', 
        'scipy'
    ]
    
    try:
        # Test critical imports
        import numpy
        import pandas
        import networkx
        print("✓ Core packages available")
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Installing required packages...")
        
        cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + required_packages
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages: {e}")
            return False

def run_analysis():
    """Run the community network analysis with proper environment setup"""
    
    # Set up environment first
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to set up dependencies")
        return False
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    
    # Input files
    adjacency_matrix = os.path.join(project_dir, "results", "adjacency_matrix_threshold_0.9.pkl")
    metadata = os.path.join(project_dir, "results", "adjacency_matrix_metadata_0.9.pkl")
    data_file = "/scratch/oms7891/selected data/selected_data.csv"
    
    # Output directory
    output_dir = os.path.join(project_dir, "community_analysis_results")
    
    # Community analysis script
    analysis_script = os.path.join(base_dir, "community_network_analysis.py")
    
    # Check if input files exist
    missing_files = []
    if not os.path.exists(adjacency_matrix):
        missing_files.append(f"Adjacency matrix: {adjacency_matrix}")
    
    if not os.path.exists(metadata):
        missing_files.append(f"Metadata: {metadata}")
    
    if not os.path.exists(data_file):
        missing_files.append(f"Data file: {data_file}")
    
    if not os.path.exists(analysis_script):
        missing_files.append(f"Analysis script: {analysis_script}")
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✓ All input files found")
    
    # Use the system Python that worked for clustering
    python_executable = "/share/apps/NYUAD5/miniconda/3-4.11.0/bin/python"
    
    # Create command
    cmd = [
        python_executable, analysis_script,
        "--adjacency_matrix", adjacency_matrix,
        "--metadata", metadata,
        "--data_file", data_file,
        "--output_dir", output_dir,
        "--resolution", "1.0"
    ]
    
    print("Starting community network analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the analysis with the fixed environment
    try:
        # Run with environment variables set
        result = subprocess.run(cmd, 
                              check=True, 
                              env=os.environ.copy(),  # Use our modified environment
                              text=True)
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Try running the analysis script directly with environment variables:")
        print(f"   export OMP_NUM_THREADS=1")
        print(f"   export OPENBLAS_NUM_THREADS=1")
        print(f"   {python_executable} {analysis_script} --adjacency_matrix {adjacency_matrix} --metadata {metadata} --data_file {data_file} --output_dir {output_dir}")
        print("\n2. If NumPy issues persist, try reinstalling in a clean environment:")
        print(f"   pip uninstall numpy && pip install numpy==1.24.3")
        return False

if __name__ == "__main__":
    print("="*60)
    print("FIXED MOF COMMUNITY NETWORK ANALYSIS")
    print("="*60)
    
    success = run_analysis()
    
    if success:
        print("\n" + "="*60)
        print("COMMUNITY NETWORK ANALYSIS COMPLETED")
        print("="*60)
        print("Generated files:")
        print("  network_overview.png - Overview of the MOF network")
        print("  community_network.png - Inter-community connections")
        print("  community_analysis_dashboard.png - Detailed analysis plots")
        print("  analysis_report.md - Comprehensive summary report")
        print("  community_assignments.csv - MOF-to-community mapping")
        print("  detailed_community_analysis.csv - Community metrics")
        print("  community_network.gexf - Network file for Gephi/Cytoscape")
    else:
        print("\n" + "="*60)
        print("ANALYSIS FAILED")
        print("="*60)
        print("Please check the error messages above and try the suggested troubleshooting steps.")