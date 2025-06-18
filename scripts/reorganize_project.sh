#!/bin/bash

# ==============================================================================
# Project Reorganization Script for Social Network Clustering
# ==============================================================================
# This script reorganizes the project structure for better organization
# ==============================================================================

echo "=========================================="
echo "Social Network Clustering Project Reorganization"
echo "=========================================="

# Set the project root
PROJECT_ROOT="/scratch/oms7891/Social_Network_Clustering"
cd "$PROJECT_ROOT"

echo "Current directory: $(pwd)"
echo ""

# Create new directory structure
echo "Creating new directory structure..."
mkdir -p {config,docs,notebooks,logs,temp}

# Move and organize files
echo ""
echo "Reorganizing files..."

# Move SLURM output/error files to logs
echo "- Moving SLURM output files to logs/"
mv enhanced_mof_analysis_*.out logs/ 2>/dev/null || echo "  No .out files to move"
mv enhanced_mof_analysis_*.err logs/ 2>/dev/null || echo "  No .err files to move"

# Move README files to docs
echo "- Moving documentation to docs/"
mv README_ENHANCED.md docs/ 2>/dev/null || echo "  README_ENHANCED.md already moved or not found"

# Move notebook files from Code to notebooks
echo "- Moving notebooks to notebooks/"
if [ -f "Code/mof_cluster_analysis.ipynb" ]; then
    mv Code/mof_cluster_analysis.ipynb notebooks/
fi
if [ -f "Code/playing_with_data.ipynb" ]; then
    mv Code/playing_with_data.ipynb notebooks/
fi

# Copy the main analysis script to scripts
echo "- Copying community_network_analysis.py to scripts/"
if [ -f "src/community_network_analysis.py" ]; then
    cp src/community_network_analysis.py scripts/
elif [ -f "Code/community_network_analysis.py" ]; then
    cp Code/community_network_analysis.py scripts/
fi

# Create a main configuration file
echo "- Creating config file..."
cat > config/project_config.json << EOF
{
  "project_name": "Social Network Clustering",
  "data_path": "/scratch/oms7891/selected data/selected_data.csv",
  "default_output_dir": "results",
  "python_executable": "/share/apps/NYUAD5/miniconda/3-4.11.0/bin/python",
  "resource_limits": {
    "OMP_NUM_THREADS": 4,
    "memory_gb": 64,
    "time_hours": 48
  },
  "default_parameters": {
    "thresholds": [0.7, 0.75],
    "algorithms": ["louvain", "girvan_newman"],
    "resolution": 1.0
  }
}
EOF

# Create a project README
echo "- Creating main README..."
cat > README.md << 'EOF'
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

EOF

# Set executable permissions
echo ""
echo "Setting executable permissions..."
chmod +x scripts/*.sh scripts/*.py 2>/dev/null

# Clean up
echo ""
echo "Cleaning up..."
# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# Remove .pyc files
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "=========================================="
echo "Reorganization Complete!"
echo "=========================================="
echo ""
echo "New structure:"
echo "  - config/       : Configuration files"
echo "  - docs/         : Documentation"
echo "  - notebooks/    : Jupyter notebooks"
echo "  - logs/         : Execution logs"
echo "  - scripts/      : Executable scripts"
echo "  - results/      : Analysis results"
echo ""
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Update any hardcoded paths in your scripts"
echo "3. Run: ./scripts/submit_enhanced_analysis.sh" 