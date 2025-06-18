#!/bin/bash

# ==============================================================================
# Master Script for Social Network Clustering Analysis
# ==============================================================================
# This script provides a menu-driven interface to manage the analysis workflow
# ==============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "success")
            echo -e "${GREEN}✓ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠ $message${NC}"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

# Main menu function
show_menu() {
    echo ""
    echo "=========================================="
    echo "Social Network Clustering Analysis Menu"
    echo "=========================================="
    echo "1. Fix fork resource error"
    echo "2. Reorganize project structure"
    echo "3. Submit enhanced analysis job"
    echo "4. Check job status"
    echo "5. View recent logs"
    echo "6. Kill all Python processes"
    echo "7. Setup environment"
    echo "8. Exit"
    echo "=========================================="
    echo -n "Enter your choice [1-8]: "
}

# Function to check and fix fork error
fix_fork_error() {
    echo "Running fork error diagnostic..."
    if [ -f "scripts/fix_fork_error.sh" ]; then
        bash scripts/fix_fork_error.sh
    else
        print_status "error" "fix_fork_error.sh not found!"
    fi
}

# Function to reorganize project
reorganize_project() {
    echo "Reorganizing project structure..."
    if [ -f "scripts/reorganize_project.sh" ]; then
        bash scripts/reorganize_project.sh
    else
        print_status "error" "reorganize_project.sh not found!"
    fi
}

# Function to submit analysis
submit_analysis() {
    echo "Submitting enhanced analysis job..."
    if [ -f "scripts/submit_enhanced_analysis.sh" ]; then
        bash scripts/submit_enhanced_analysis.sh
    else
        print_status "error" "submit_enhanced_analysis.sh not found!"
    fi
}

# Function to check job status
check_job_status() {
    echo "Checking SLURM job status..."
    echo "----------------------------------------"
    squeue -u $USER
    echo ""
    echo "Recent jobs:"
    sacct -u $USER --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS --starttime=$(date -d '7 days ago' +%Y-%m-%d)
}

# Function to view logs
view_logs() {
    echo "Recent log files:"
    echo "----------------------------------------"
    
    # Check for logs directory
    if [ -d "logs" ]; then
        echo "In logs directory:"
        ls -lt logs/*.{out,err} 2>/dev/null | head -10
    fi
    
    # Check for recent SLURM logs
    echo ""
    echo "Recent SLURM output files:"
    ls -lt enhanced_mof_analysis_*.{out,err} 2>/dev/null | head -10
    
    echo ""
    echo -n "Enter log file name to view (or 'q' to go back): "
    read logfile
    
    if [ "$logfile" != "q" ] && [ -f "$logfile" ]; then
        tail -50 "$logfile"
    fi
}

# Function to kill Python processes
kill_python_processes() {
    echo "Checking Python processes..."
    PYTHON_COUNT=$(ps aux | grep python | grep -v grep | wc -l)
    
    if [ $PYTHON_COUNT -gt 0 ]; then
        print_status "warning" "Found $PYTHON_COUNT Python processes"
        ps aux | grep python | grep -v grep
        echo ""
        echo -n "Kill all Python processes? (y/n): "
        read confirm
        
        if [ "$confirm" = "y" ]; then
            pkill -u $USER python
            print_status "success" "Python processes killed"
        fi
    else
        print_status "success" "No Python processes found"
    fi
}

# Function to setup environment
setup_environment() {
    echo "Setting up environment..."
    echo "----------------------------------------"
    
    # Set resource limits
    ulimit -u 4096
    ulimit -n 4096
    ulimit -s unlimited
    
    # Set environment variables
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export VECLIB_MAXIMUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    
    # Add Python path
    export PATH="/share/apps/NYUAD5/miniconda/3-4.11.0/bin:$PATH"
    
    print_status "success" "Environment configured"
    echo "Resource limits:"
    ulimit -a | grep -E "processes|files|stack"
    echo ""
    echo "Python path: $(which python3)"
}

# Main script
cd /scratch/oms7891/Social_Network_Clustering

# Make all scripts executable
chmod +x scripts/*.sh 2>/dev/null

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            fix_fork_error
            ;;
        2)
            reorganize_project
            ;;
        3)
            submit_analysis
            ;;
        4)
            check_job_status
            ;;
        5)
            view_logs
            ;;
        6)
            kill_python_processes
            ;;
        7)
            setup_environment
            ;;
        8)
            print_status "success" "Exiting..."
            exit 0
            ;;
        *)
            print_status "error" "Invalid choice. Please try again."
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
done 