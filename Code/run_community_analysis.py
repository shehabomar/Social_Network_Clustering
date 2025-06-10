#!/usr/bin/env python3

"""
Wrapper script to run community network analysis
This script will load your existing adjacency matrix and create beautiful visualizations
"""

import os
import sys
import subprocess

def run_analysis():
    """Run the community network analysis"""
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    
    # Input files
    adjacency_matrix = os.path.join(project_dir, "results", "adjacency_matrix_threshold_0.9.pkl")
    metadata = os.path.join(project_dir, "results", "adjacency_matrix_metadata_0.9.pkl")
    data_file = "/scratch/oms7891/selected data/selected_data.csv"  # Original data path from the output
    
    # Output directory
    output_dir = os.path.join(project_dir, "community_analysis_results")
    
    # Community analysis script
    analysis_script = os.path.join(base_dir, "community_network_analysis.py")
    
    # Check if input files exist
    if not os.path.exists(adjacency_matrix):
        print(f"Error: Adjacency matrix not found at {adjacency_matrix}")
        return False
    
    if not os.path.exists(metadata):
        print(f"Error: Metadata file not found at {metadata}")
        return False
    
    if not os.path.exists(data_file):
        print(f"Warning: Original data file not found at {data_file}")
        print("You may need to update the data_file path in this script")
        return False
    
    # Create command
    cmd = [
        sys.executable, analysis_script,
        "--adjacency_matrix", adjacency_matrix,
        "--metadata", metadata,
        "--data_file", data_file,
        "--output_dir", output_dir,
        "--resolution", "1.0"  # You can adjust this parameter
    ]
    
    print("Starting community network analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_analysis()
    if success:
        print("\n" + "="*60)
        print("COMMUNITY NETWORK ANALYSIS COMPLETED")
        print("="*60)
        print("Generated files:")
        print("  📊 network_overview.png - Overview of the MOF network")
        print("  🌐 community_network.png - Inter-community connections")
        print("  📈 community_analysis_dashboard.png - Detailed analysis plots")
        print("  📝 analysis_report.md - Comprehensive summary report")
        print("  📋 community_assignments.csv - MOF-to-community mapping")
        print("  📊 detailed_community_analysis.csv - Community metrics")
        print("  🔗 community_network.gexf - Network file for Gephi/Cytoscape")
    else:
        print("Analysis failed. Please check the error messages above.") 