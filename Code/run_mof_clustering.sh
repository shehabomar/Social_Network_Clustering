#!/bin/bash
# MOF Social Network Clustering Pipeline
# Run script for simplifying the execution of the clustering pipeline

# Default parameters
INPUT_FILE="../selected data/selected_data.csv"
OUTPUT_DIR="../results"
SAMPLE_SIZE=""
THRESHOLD=0.85
RESOLUTION=1.0

# Help message
function show_help {
    echo "MOF Social Network Clustering Pipeline"
    echo ""
    echo "Usage: ./run_mof_clustering.sh [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE       Input CSV file with MOF features (default: $INPUT_FILE)"
    echo "  -o, --output DIR       Output directory (default: $OUTPUT_DIR)"
    echo "  -s, --sample SIZE      Number of MOFs to sample (for testing)"
    echo "  -t, --threshold VAL    Similarity threshold (default: $THRESHOLD)"
    echo "  -r, --resolution VAL   Community detection resolution (default: $RESOLUTION)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_mof_clustering.sh                     # Run with default parameters"
    echo "  ./run_mof_clustering.sh -s 10000            # Run with 10,000 sample MOFs"
    echo "  ./run_mof_clustering.sh -t 0.9 -r 1.2       # Custom threshold and resolution"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Prepare the command
CMD="python mof_social_network.py --input \"$INPUT_FILE\" --output_dir \"$OUTPUT_DIR\" --threshold $THRESHOLD --resolution $RESOLUTION"

# Add sample size if provided
if [ ! -z "$SAMPLE_SIZE" ]; then
    CMD="$CMD --sample $SAMPLE_SIZE"
fi

# Print configuration
echo "=== MOF Social Network Clustering ==="
echo "Input file:       $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Threshold:        $THRESHOLD"
echo "Resolution:       $RESOLUTION"
if [ ! -z "$SAMPLE_SIZE" ]; then
    echo "Sample size:      $SAMPLE_SIZE"
else
    echo "Sample size:      [Full dataset]"
fi
echo "================================="

# Run the command
echo "Starting clustering pipeline..."
echo "$CMD"
eval "$CMD"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "Clustering completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "To analyze the results, run the Jupyter notebook:"
    echo "jupyter notebook mof_cluster_analysis.ipynb"
else
    echo "Error running the clustering pipeline."
    exit 1
fi 