import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import community as community_louvain  
import argparse

def load_and_preprocess_data(file_path, sample_size=None):
    """
    Load the MOF feature data and preprocess it
    
    Args:
        file_path: Path to CSV file with MOF features
        sample_size: Optional number of samples to use (for testing)
        
    Returns:
        DataFrame with preprocessed features and MOF identifiers
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using a sample of {sample_size} MOFs for testing")
    else:
        print(f"Processing all {len(df)} MOFs")
    
    # Identify the ID column (assuming it's 'Folder Name' or the first column)
    id_column = 'Folder Name' if 'Folder Name' in df.columns else df.columns[0]
    
    # Extract identifiers
    mof_ids = df[id_column].values
    
    # Extract features (all columns except the ID column)
    feature_columns = [col for col in df.columns if col != id_column]
    features_df = df[feature_columns]
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    # Normalize the features using Min-Max scaling as mentioned in the paper
    print("Normalizing features...")
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_df)
    
    # Create a DataFrame with normalized features
    normalized_df = pd.DataFrame(features_normalized, columns=feature_columns)
    normalized_df[id_column] = mof_ids
    
    print(f"Preprocessing complete. Data shape: {normalized_df.shape}")
    return normalized_df, id_column

def calculate_similarity_matrix(df, id_column, threshold=0.9):
    """
    Calculate the similarity matrix between MOFs and apply threshold
    
    Args:
        df: DataFrame with normalized features
        id_column: Name of the column containing MOF identifiers
        threshold: Minimum similarity score to keep
        
    Returns:
        Sparse adjacency matrix as a dictionary of {(node1, node2): similarity}
    """
    # Extract features (all columns except the ID column)
    feature_columns = [col for col in df.columns if col != id_column]
    features = df[feature_columns].values
    
    # Get MOF identifiers
    mof_ids = df[id_column].values
    
    # Calculate cosine similarity
    print("Calculating cosine similarity matrix...")
    # For large datasets, this might need to be done in batches
    if len(features) > 10000:
        # Process in batches for very large datasets
        print("Large dataset detected. Processing similarity in batches...")
        adjacency_dict = {}
        batch_size = 1000
        num_samples = len(features)
        
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_end = min(i + batch_size, num_samples)
            batch_features = features[i:batch_end]
            
            # Calculate similarities between this batch and all features
            batch_similarities = cosine_similarity(batch_features, features)
            
            # Apply threshold and add to adjacency dict
            for batch_idx, similarities in enumerate(batch_similarities):
                global_idx = i + batch_idx
                for j, sim in enumerate(similarities):
                    if global_idx != j and sim >= threshold:
                        adjacency_dict[(mof_ids[global_idx], mof_ids[j])] = sim
    else:
        # Process all at once for smaller datasets
        similarity_matrix = cosine_similarity(features)
        
        # Convert to sparse representation using a dictionary
        print("Converting to sparse adjacency dictionary...")
        adjacency_dict = {}
        for i in tqdm(range(len(similarity_matrix))):
            for j in range(i+1, len(similarity_matrix)):  # Only upper triangle to avoid duplicates
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    adjacency_dict[(mof_ids[i], mof_ids[j])] = sim
    
    print(f"Created adjacency matrix with {len(adjacency_dict)} edges above threshold {threshold}")
    return adjacency_dict

def build_network(adjacency_dict):
    """
    Build a NetworkX graph from the adjacency dictionary
    
    Args:
        adjacency_dict: Dictionary of {(node1, node2): weight}
        
    Returns:
        NetworkX graph
    """
    print("Building network graph...")
    G = nx.Graph()
    
    # Add weighted edges from the adjacency dictionary
    for (node1, node2), weight in tqdm(adjacency_dict.items()):
        G.add_edge(node1, node2, weight=weight)
    
    print(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def detect_communities(G, resolution=1.0):
    """
    Detect communities in the graph using the Louvain algorithm
    
    Args:
        G: NetworkX graph
        resolution: Resolution parameter for community detection
        
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    print("Detecting communities...")
    # Use Louvain community detection algorithm
    communities = community_louvain.best_partition(G, resolution=resolution)
    
    # Count communities
    unique_communities = set(communities.values())
    print(f"Detected {len(unique_communities)} communities")
    
    # Count members per community
    community_counts = {}
    for community_id in communities.values():
        community_counts[community_id] = community_counts.get(community_id, 0) + 1
    
    # Print community size distribution
    print("\nCommunity size distribution:")
    for community_id, count in sorted(community_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Community {community_id}: {count} members")
    
    return communities

def save_communities(communities, output_file):
    """
    Save the community assignments to a CSV file
    
    Args:
        communities: Dictionary mapping node IDs to community IDs
        output_file: Path to save the CSV file
    """
    print(f"Saving community assignments to {output_file}...")
    community_df = pd.DataFrame({
        'MOF_ID': list(communities.keys()),
        'Community': list(communities.values())
    })
    community_df.to_csv(output_file, index=False)
    print("Save complete")

def visualize_network_sample(G, communities, output_file, sample_size=1000):
    """
    Visualize a sample of the network colored by community
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping node IDs to community IDs
        output_file: Path to save the visualization
        sample_size: Number of nodes to include in the visualization
    """
    if G.number_of_nodes() > sample_size:
        # Sample nodes for visualization
        sampled_nodes = list(G.nodes())[:sample_size]
        G_sample = G.subgraph(sampled_nodes)
    else:
        G_sample = G
    
    print(f"Visualizing a sample of {G_sample.number_of_nodes()} nodes...")
    
    # Create a spring layout
    pos = nx.spring_layout(G_sample, seed=42)
    
    plt.figure(figsize=(12, 12))
    
    # Color nodes according to community
    cmap = plt.cm.get_cmap('viridis', max(communities.values()) + 1)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_sample, 
        pos, 
        node_size=50, 
        node_color=[communities[node] for node in G_sample.nodes()],
        cmap=cmap
    )
    
    # Draw edges with low alpha for better visibility
    nx.draw_networkx_edges(G_sample, pos, alpha=0.1)
    
    plt.title(f"MOF Social Network (Sample of {G_sample.number_of_nodes()} nodes)", fontsize=15)
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='MOF Social Network Clustering')
    parser.add_argument('--input', required=True, help='Input CSV file with MOF features')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--sample', type=int, help='Optional sample size for testing')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold (default: 0.9)')
    parser.add_argument('--resolution', type=float, default=1.0, help='Community detection resolution (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    df, id_column = load_and_preprocess_data(args.input, sample_size=args.sample)
    
    # Calculate similarity matrix
    adjacency_dict = calculate_similarity_matrix(df, id_column, threshold=args.threshold)
    
    # Build network
    G = build_network(adjacency_dict)
    
    # Detect communities
    communities = detect_communities(G, resolution=args.resolution)
    
    # Save communities to file
    output_file = os.path.join(args.output_dir, 'mof_communities.csv')
    save_communities(communities, output_file)
    
    # Visualize network (sample)
    viz_file = os.path.join(args.output_dir, 'mof_network_visualization.png')
    visualize_network_sample(G, communities, viz_file)
    
    print("MOF Social Network Clustering completed successfully!")

if __name__ == "__main__":
    main() 