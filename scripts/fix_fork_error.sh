#!/bin/bash

# ==============================================================================
# Fix Fork Resource Error Script
# ==============================================================================
# This script helps resolve "bash: fork: retry: Resource temporarily unavailable"
# ==============================================================================

echo "=========================================="
echo "Fixing Fork Resource Error"
echo "=========================================="

# Check current resource usage
echo "1. Checking current resource usage..."
echo "----------------------------------------"

# Check process count
PROC_COUNT=$(ps aux | wc -l)
echo "Current process count: $PROC_COUNT"

# Check memory usage
echo "Memory usage:"
free -h

# Check current user limits
echo ""
echo "2. Current resource limits:"
echo "----------------------------------------"
ulimit -a

# Kill zombie processes
echo ""
echo "3. Cleaning up zombie processes..."
echo "----------------------------------------"
ZOMBIES=$(ps aux | grep -w Z | grep -v grep | wc -l)
if [ $ZOMBIES -gt 0 ]; then
    echo "Found $ZOMBIES zombie processes"
    ps aux | grep -w Z | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    echo "Cleaned up zombie processes"
else
    echo "No zombie processes found"
fi

# Kill unnecessary python processes
echo ""
echo "4. Checking for stale Python processes..."
echo "----------------------------------------"
PYTHON_PROCS=$(ps aux | grep python | grep -v grep | wc -l)
echo "Found $PYTHON_PROCS Python processes"

if [ $PYTHON_PROCS -gt 5 ]; then
    echo "WARNING: Many Python processes running. Consider killing stale ones:"
    ps aux | grep python | grep -v grep | head -10
    echo ""
    echo "To kill a specific process: kill -9 <PID>"
    echo "To kill all your Python processes: pkill -u $USER python"
fi

# Clean up temporary files
echo ""
echo "5. Cleaning up temporary files..."
echo "----------------------------------------"
if [ -d "/tmp" ]; then
    find /tmp -user $USER -type f -mtime +1 -delete 2>/dev/null
    echo "Cleaned old temporary files"
fi

# Clear bash history if it's too large
echo ""
echo "6. Checking bash history size..."
echo "----------------------------------------"
HIST_SIZE=$(wc -c < ~/.bash_history 2>/dev/null || echo 0)
if [ $HIST_SIZE -gt 1048576 ]; then  # If larger than 1MB
    echo "Bash history is large ($(($HIST_SIZE / 1024))KB), truncating..."
    tail -n 1000 ~/.bash_history > ~/.bash_history.tmp
    mv ~/.bash_history.tmp ~/.bash_history
    echo "Bash history truncated to last 1000 commands"
else
    echo "Bash history size is OK"
fi

# Suggest increasing limits
echo ""
echo "7. Recommendations to prevent this error:"
echo "----------------------------------------"
echo "Add these to your ~/.bashrc file:"
echo ""
echo "# Increase resource limits"
echo "ulimit -u 4096      # Increase max user processes"
echo "ulimit -n 4096      # Increase max open files"
echo "ulimit -s unlimited # Increase stack size"
echo ""
echo "For immediate effect, run:"
echo "source ~/.bashrc"

# Check if we're in a SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "8. SLURM job detected!"
    echo "----------------------------------------"
    echo "Current SLURM job ID: $SLURM_JOB_ID"
    echo "Consider requesting more resources in your SLURM script:"
    echo "  --mem=64G"
    echo "  --cpus-per-task=8"
fi

echo ""
echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "If the error persists:"
echo "1. Log out and log back in"
echo "2. Request a new compute node"
echo "3. Contact system administrator"
echo ""
echo "To free up more resources, you can:"
echo "- Kill all your processes: pkill -u $USER"
echo "- Check disk usage: du -sh ~/* | sort -h"
echo "- Clean conda cache: conda clean --all" 