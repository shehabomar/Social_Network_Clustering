import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from AnalysisController import MOFAnalysisController


if __name__ == "__main__":
    adj_file = "results/3k/louvain_0.9t/adj_matrix/adjacency_matrix_t0.9.pkl"
    communities_file = "results/3k/louvain_0.9t/comm_analysis/community_assignments.csv"
    output_dir = "results/3k/integrated_analysis_louvain_t0.9"

    analyser = MOFAnalysisController(adj_file, communities_file, output_dir)
    analyser.run_integrated_analysis()
    print("DONE!")