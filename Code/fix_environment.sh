#!/bin/bash

# Script to fix the virtual environment

echo "Fixing virtual environment..."

# Remove old virtual environment
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Use system Python
export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"

# Create new virtual environment
echo "Creating new virtual environment..."
python -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages with specific numpy version
echo "Installing packages with compatible numpy version..."
pip install numpy==1.24.3
pip install pandas scikit-learn networkx matplotlib tqdm python-louvain seaborn scipy

echo "Environment fixed! Testing numpy import..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo "Done!" 