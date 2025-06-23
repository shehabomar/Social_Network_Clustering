import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from AnalysisController import MOFAnalysisController


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run integrated centrality analysis.')
    parser.add_argument('--adjacency_matrix', required=True, help='Path to adjacency matrix pickle file')
    parser.add_argument('--communities_file', required=True, help='Path to community assignments CSV file')
    parser.add_argument('--output_dir', required=True, help='Directory to save analysis results')
    args = parser.parse_args()

    analyser = MOFAnalysisController(
        adj_file=args.adjacency_matrix, 
        communities_file=args.communities_file, 
        output_dir=args.output_dir
    )
    analyser.run_integrated_analysis()
    print("DONE!")