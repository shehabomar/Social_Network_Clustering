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
from sklearn.metrics import silhouette_score
import seaborn as sns
from collections import defaultdict
from scipy import sparse
import multiprocessing as mp
from functools import partial

# Default maximum number of neighbors to keep per node (to cap memory)
DEFAULT_TOP_K = 50

def process_batch(batch_features, all_features, threshold, start_idx, mof_ids):
    """
    Process a batch of features and calculate similarities
    
    Args:
        batch_features: Features for the current batch
        all_features: All features
        threshold: Similarity threshold
        start_idx: Starting index of the batch
        mof_ids: List of MOF identifiers
        
    Returns:
        Dictionary of {(node1, node2): similarity} for this batch
    """
    batch_similarities = cosine_similarity(batch_features, all_features)
    adjacency_dict = {}
    
    for batch_idx, similarities in enumerate(batch_similarities):
        global_idx = start_idx + batch_idx
        for j, sim in enumerate(similarities):
            if global_idx != j and sim >= threshold:
                adjacency_dict[(mof_ids[global_idx], mof_ids[j])] = sim
    
    return adjacency_dict

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
    
    # Normalize the features using Min-Max scaling
    print("Normalizing features...")
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_df).astype(np.float32)
    
    # Create a DataFrame with normalized features
    normalized_df = pd.DataFrame(features_normalized, columns=feature_columns)
    normalized_df[id_column] = mof_ids
    
    print(f"Preprocessing complete. Data shape: {normalized_df.shape}")
    return normalized_df, id_column

def calculate_similarity_matrix(df, id_column, threshold=0.9, batch_size=10000, top_k=DEFAULT_TOP_K):
    """
    Calculate the similarity matrix between MOFs and apply threshold using parallel processing
    
    Args:
        df: DataFrame with normalized features
        id_column: Name of the column containing MOF identifiers
        threshold: Minimum similarity score to keep
        batch_size: Size of batches for parallel processing
        top_k: Maximum number of neighbors to keep per node
        
    Returns:
        Sparse adjacency matrix as a dictionary of {(node1, node2): similarity}
    """
    # Extract features (all columns except the ID column)
    feature_columns = [col for col in df.columns if col != id_column]
    features = df[feature_columns].values
    
    # Get MOF identifiers
    mof_ids = df[id_column].values
    
    # Cast features to float32 to reduce memory footprint
    features = features.astype(np.float32)
    num_samples = len(features)
    adjacency_dict = {}

    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Similarity batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_features = features[start_idx:end_idx]

        # Compute cosine similarity between the batch and all features
        sims = cosine_similarity(batch_features, features)

        # Iterate over similarities and apply threshold
        for local_i, similarities in enumerate(sims):
            global_i = start_idx + local_i
            for j, sim in enumerate(similarities):
                if sim < threshold:
                    continue

                # Keep only top_k highest sims per node to cap memory
                # We will collect all candidate edges first, then prune
                adjacency_dict.setdefault(mof_ids[global_i], []).append((mof_ids[j], float(sim)))

        # After processing the batch rows, prune each node list to top_k
        for src, neighbors in adjacency_dict.items():
            if len(neighbors) > top_k:
                neighbors.sort(key=lambda x: x[1], reverse=True)
                adjacency_dict[src] = neighbors[:top_k]

    # Flatten adjacency_dict into edge dictionary without duplicates
    edge_dict = {}
    for src, neighbors in adjacency_dict.items():
        for dst, sim in neighbors:
            if src < dst:  # ensure single direction edge
                edge_dict[(src, dst)] = sim
            else:
                edge_dict[(dst, src)] = sim  # maintain ordering
    
    print(f"Created adjacency matrix with {len(edge_dict)} edges above threshold {threshold}")
    return edge_dict

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

def detect_communities_louvain(G, resolution=1.0):
    """
    Detect communities using the Louvain algorithm
    
    Args:
        G: NetworkX graph
        resolution: Resolution parameter for community detection
        
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    print("Detecting communities using Louvain algorithm...")
    communities = community_louvain.best_partition(G, resolution=resolution)
    
    # Count communities
    unique_communities = set(communities.values())
    print(f"Detected {len(unique_communities)} communities using Louvain")
    
    return communities

def detect_communities_girvan_newman(G, num_communities=None):
    """
    Detect communities using the Girvan-Newman algorithm
    
    Args:
        G: NetworkX graph
        num_communities: Target number of communities (optional)
        
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    print("Detecting communities using Girvan-Newman algorithm...")
    
    # If num_communities is not specified, use modularity to determine optimal number
    if num_communities is None:
        communities_generator = nx.community.girvan_newman(G)
        best_communities = None
        best_modularity = -1
        
        for communities in tqdm(communities_generator):
            current_modularity = nx.community.modularity(G, communities)
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_communities = communities
            if len(communities) > 20:  # Limit the number of communities
                break
    else:
        communities_generator = nx.community.girvan_newman(G)
        for i, communities in enumerate(communities_generator):
            if i == num_communities - 1:
                best_communities = communities
                break
    
    # Convert to dictionary format
    community_dict = {}
    for i, community in enumerate(best_communities):
        for node in community:
            community_dict[node] = i
    
    print(f"Detected {len(best_communities)} communities using Girvan-Newman")
    return community_dict

def analyze_communities(G, communities, df, id_column):
    """
    Analyze communities and extract useful information
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping node IDs to community IDs
        df: Original DataFrame with MOF features
        id_column: Name of the column containing MOF identifiers
        
    Returns:
        Dictionary with community analysis results
    """
    print("Analyzing communities...")
    
    # Group MOFs by community
    community_mofs = defaultdict(list)
    for mof_id, community_id in communities.items():
        community_mofs[community_id].append(mof_id)
    
    # Calculate community statistics
    community_stats = {}
    for community_id, mofs in community_mofs.items():
        # Get features for MOFs in this community
        community_df = df[df[id_column].isin(mofs)]
        
        # Calculate statistics
        stats = {
            'size': len(mofs),
            'avg_degree': np.mean([G.degree(mof) for mof in mofs]),
            'density': nx.density(G.subgraph(mofs)),
            'central_mofs': sorted(mofs, key=lambda x: G.degree(x), reverse=True)[:5]
        }
        
        # Add feature statistics if available
        feature_columns = [col for col in df.columns if col != id_column]
        if feature_columns:
            stats['feature_means'] = community_df[feature_columns].mean().to_dict()
        
        community_stats[community_id] = stats
    
    return community_stats

def visualize_network(G, communities, output_file, title="MOF Social Network"):
    """
    Create an improved visualization of the network
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping node IDs to community IDs
        output_file: Path to save the visualization
        title: Title for the plot
    """
    print("Creating network visualization...")
    
    # Create a spring layout with better parameters
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    plt.figure(figsize=(15, 15))
    
    # Draw edges with low alpha for better visibility
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    
    # Draw nodes with community colors
    cmap = plt.cm.get_cmap('viridis', max(communities.values()) + 1)
    node_colors = [communities[node] for node in G.nodes()]
    
    # Draw nodes with size based on degree
    node_sizes = [G.degree(node) * 10 for node in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, 
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=cmap,
        alpha=0.7
    )
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(communities.values())))
    plt.colorbar(sm, label='Community ID')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Save the figure with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def save_community_analysis(community_stats, output_file):
    """
    Save community analysis results to a CSV file
    
    Args:
        community_stats: Dictionary with community analysis results
        output_file: Path to save the CSV file
    """
    print(f"Saving community analysis to {output_file}...")
    
    # Convert to DataFrame
    rows = []
    for community_id, stats in community_stats.items():
        row = {
            'Community_ID': community_id,
            'Size': stats['size'],
            'Average_Degree': stats['avg_degree'],
            'Density': stats['density'],
            'Central_MOFs': ', '.join(stats['central_mofs'])
        }
        
        # Add feature means if available
        if 'feature_means' in stats:
            row.update(stats['feature_means'])
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print("Analysis saved successfully")

def main():
    parser = argparse.ArgumentParser(description='MOF Social Network Clustering')
    parser.add_argument('--input', required=True, help='Input CSV file with MOF features')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--sample', type=int, help='Optional sample size for testing')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold (default: 0.9)')
    parser.add_argument('--resolution', type=float, default=1.0, help='Community detection resolution (default: 1.0)')
    parser.add_argument('--algorithm', choices=['louvain', 'girvan_newman', 'both'], default='both',
                      help='Community detection algorithm to use (default: both)')
    parser.add_argument('--batch_size', type=int, default=10000,
                      help='Batch size for parallel processing (default: 10000)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    df, id_column = load_and_preprocess_data(args.input, sample_size=args.sample)
    
    # Calculate similarity matrix
    adjacency_dict = calculate_similarity_matrix(df, id_column, threshold=args.threshold, batch_size=args.batch_size)
    
    # Build network
    G = build_network(adjacency_dict)
    
    # Detect communities using selected algorithm(s)
    if args.algorithm in ['louvain', 'both']:
        louvain_communities = detect_communities_louvain(G, resolution=args.resolution)
        
        # Analyze and visualize Louvain communities
        louvain_stats = analyze_communities(G, louvain_communities, df, id_column)
        visualize_network(G, louvain_communities, 
                         os.path.join(args.output_dir, 'louvain_network.png'),
                         "MOF Social Network (Louvain Communities)")
        save_community_analysis(louvain_stats, 
                              os.path.join(args.output_dir, 'louvain_community_analysis.csv'))
    
    if args.algorithm in ['girvan_newman', 'both']:
        girvan_newman_communities = detect_communities_girvan_newman(G)
        
        # Analyze and visualize Girvan-Newman communities
        gn_stats = analyze_communities(G, girvan_newman_communities, df, id_column)
        visualize_network(G, girvan_newman_communities, 
                         os.path.join(args.output_dir, 'girvan_newman_network.png'),
                         "MOF Social Network (Girvan-Newman Communities)")
        save_community_analysis(gn_stats, 
                              os.path.join(args.output_dir, 'girvan_newman_community_analysis.csv'))
    
    print("MOF Social Network Clustering completed successfully!")

if __name__ == "__main__":
    main() 