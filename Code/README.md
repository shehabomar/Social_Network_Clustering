# MOF Social Network Clustering

This tool builds a social network representation for clustering Metal-Organic Framework (MOF) structures based on their geometric, compositional, and other properties. The approach follows a research-based methodology for identifying similar MOFs through community detection in a graph representation.

## Requirements

Install the required packages:

```bash
pip install numpy pandas scikit-learn networkx matplotlib tqdm python-louvain
```

## Usage

The main script `mof_social_network.py` takes several arguments:

```bash
python mof_social_network.py --input PATH_TO_FEATURES_CSV --output_dir RESULTS_DIRECTORY [--sample SAMPLE_SIZE] [--threshold SIMILARITY_THRESHOLD] [--resolution COMMUNITY_RESOLUTION]
```

### Arguments:

- `--input`: (Required) Path to the CSV file containing MOF features (e.g., 'selected_data/selected_data.csv')
- `--output_dir`: Directory to save results (default: 'results')
- `--sample`: Optional number of MOFs to use (for testing with a smaller dataset)
- `--threshold`: Similarity threshold for building the network (default: 0.9)
- `--resolution`: Resolution parameter for community detection (default: 1.0)

## Example Usage

For a full run with all MOFs:

```bash
python mof_social_network.py --input "../selected data/selected_data.csv" --output_dir "../results" --threshold 0.85
```

For a test run with a sample of 10,000 MOFs:

```bash
python mof_social_network.py --input "../selected data/selected_data.csv" --output_dir "../results" --sample 10000 --threshold 0.85
```

## Pipeline Steps

1. **Data Preprocessing**: Features are loaded, missing values are handled, and all features are normalized using Min-Max scaling to range [0,1].

2. **Similarity Calculation**: Cosine similarity is calculated between all pairs of MOF feature vectors. For large datasets, this is done in batches to manage memory usage.

3. **Network Construction**: A graph is built where each node is a MOF, and edges connect MOFs with similarity above the specified threshold.

4. **Community Detection**: The Louvain algorithm is applied to detect communities of similar MOFs in the network.

5. **Output Generation**:
   - A CSV file with MOF identifiers and their assigned community IDs
   - A visualization of a sample of the network, colored by community

## Computational Considerations

- The similarity calculation step can be memory-intensive for large datasets (>10,000 MOFs). The script handles this by processing in batches when needed.
- For the full dataset of ~280,000 MOFs, consider running on a machine with at least 32GB of RAM.
- The script shows progress bars for time-consuming operations.

## Tuning Parameters

- **Similarity Threshold**: Controls the density of the network. Lower values create more connections but may lead to less distinct communities.
- **Resolution Parameter**: Controls the granularity of communities. Higher values result in more, smaller communities.

## Output Files

- `mof_communities.csv`: CSV file with MOF identifiers and their assigned community IDs
- `mof_network_visualization.png`: Visualization of the MOF network (sampled for large networks) 