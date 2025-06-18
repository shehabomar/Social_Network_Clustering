# Social Network Clustering Project

## Project Structure

```
Social_Network_Clustering/
├── scripts/              # Executable scripts
│   ├── run_enhanced_analysis.py
│   ├── submit_enhanced_analysis.sh
│   ├── community_network_analysis.py
│   └── ...
├── src/                  # Source code modules
│   └── ...
├── Code/                 # Legacy code directory
│   └── ...
├── config/               # Configuration files
│   └── project_config.json
├── docs/                 # Documentation
│   └── README_ENHANCED.md
├── notebooks/            # Jupyter notebooks
│   ├── mof_cluster_analysis.ipynb
│   └── playing_with_data.ipynb
├── logs/                 # SLURM and execution logs
│   └── enhanced_mof_analysis_*.{out,err}
├── results/              # Analysis results
│   ├── enhanced_community_results/
│   ├── community_analysis_results/
│   └── ...
├── data/                 # Local data cache (if needed)
└── temp/                 # Temporary files

```

## Quick Start

1. Submit enhanced analysis job:
   ```bash
   cd /scratch/oms7891/Social_Network_Clustering
   ./scripts/submit_enhanced_analysis.sh
   ```

2. Monitor job:
   ```bash
   squeue -u $USER
   ```

3. Check results in the output directory specified (default: `enhanced_community_results/`)

## Data Location

Main data file: `/scratch/oms7891/selected data/selected_data.csv`

