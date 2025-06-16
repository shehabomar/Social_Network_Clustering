import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
import pickle
from collections import defaultdict, Counter
import argparse
from scipy.stats import pearsonr, zscore
import warnings
import time
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_adjacency_matrix(adjacency_file):
    """Load the adjacency matrix from disk"""
    print(f"Loading adjacency matrix from {adjacency_file}...")
    with open(adjacency_file, 'rb') as f:
        adjacency_dict = pickle.load(f)
    print(f"Loaded adjacency matrix with {len(adjacency_dict):,} edges")
    return adjacency_dict

def load_metadata(metadata_file):
    """Load metadata about the adjacency matrix"""
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Dataset info: {metadata['num_nodes']:,} nodes, {metadata['num_edges']:,} edges, threshold={metadata['threshold']}")
    return metadata

def build_network_from_adjacency(adjacency_dict):
    """Build NetworkX graph from adjacency dictionary"""
    print("Building network graph...")
    G = nx.Graph()
    
    for (node1, node2), weight in tqdm(adjacency_dict.items(), desc="Adding edges"):
        G.add_edge(node1, node2, weight=weight)
    
    print(f"Network built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

def detect_communities_enhanced(G, resolution=1.0):
    """Enhanced community detection with multiple methods"""
    print("Detecting communities using enhanced Louvain algorithm...")
    
    # Use weighted edges for community detection
    communities = community_louvain.best_partition(G, weight='weight', resolution=resolution, random_state=42)
    
    # Calculate modularity
    modularity = community_louvain.modularity(communities, G, weight='weight')
    
    # Get community statistics
    community_sizes = Counter(communities.values())
    num_communities = len(community_sizes)
    
    print(f"Detected {num_communities} communities with modularity {modularity:.4f}")
    print(f"Community sizes: min={min(community_sizes.values())}, max={max(community_sizes.values())}, avg={np.mean(list(community_sizes.values())):.1f}")
    
    return communities, modularity

def detect_communities_girvan_newman_enhanced(G, max_communities=50):
    """Enhanced Girvan-Newman algorithm with better stopping criteria"""
    print("Detecting communities using enhanced Girvan-Newman algorithm...")
    
    # For large networks, work with the largest connected component
    if G.number_of_nodes() > 5000:
        print("Network is large. Using largest connected component only.")
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
        print(f"Working with subgraph of {G_sub.number_of_nodes()} nodes")
    else:
        G_sub = G
    
    # Use modularity-based stopping criterion
    communities_generator = nx.community.girvan_newman(G_sub)
    best_communities = None
    best_modularity = -1
    community_count = 0
    
    print("Running Girvan-Newman with modularity optimization...")
    
    for communities in tqdm(communities_generator, desc="GN iterations"):
        current_modularity = nx.community.modularity(G_sub, communities, weight='weight')
        community_count = len(communities)
        
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = communities
        
        # Stop if modularity starts decreasing significantly or we reach max communities
        if community_count > max_communities or (community_count > 10 and current_modularity < best_modularity - 0.01):
            break
    
    # Convert to dictionary format
    community_dict = {}
    for i, community in enumerate(best_communities):
        for node in community:
            community_dict[node] = i
    
    # Add isolated nodes from original graph if we used a subgraph
    if G.number_of_nodes() > G_sub.number_of_nodes():
        next_comm_id = len(best_communities)
        for node in G.nodes():
            if node not in community_dict:
                community_dict[node] = next_comm_id
                next_comm_id += 1
    
    print(f"Detected {len(set(community_dict.values()))} communities using Girvan-Newman with modularity {best_modularity:.4f}")
    return community_dict, best_modularity

def calculate_proper_conductance(G, community_nodes):
    """
    Calculate proper conductance using the formula: φ(S) = cut(S) / min(vol(S), vol(V\S))
    Where cut(S) = edges crossing community boundary, vol(S) = sum of degrees in community
    """
    total_degree = sum(dict(G.degree()).values())
    
    # Calculate volume of community
    vol_S = sum(G.degree(node) for node in community_nodes)
    vol_complement = total_degree - vol_S
    
    # Calculate cut (edges crossing boundary)
    cut_S = 0
    for node in community_nodes:
        for neighbor in G.neighbors(node):
            if neighbor not in community_nodes:
                cut_S += 1
    
    # Conductance formula
    if min(vol_S, vol_complement) == 0:
        return 1.0  # Maximum conductance for isolated components
    
    conductance = cut_S / min(vol_S, vol_complement)
    return conductance

def analyze_average_degree_outliers(community_stats):
    """
    Analyze average degree outliers and provide filtering recommendations
    """
    degrees = [stats['avg_degree'] for stats in community_stats.values()]
    sizes = [stats['size'] for stats in community_stats.values()]
    
    # Calculate statistics
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    
    # Identify outliers using 2-sigma rule
    outliers = {}
    for comm_id, stats in community_stats.items():
        is_outlier = False
        reasons = []
        
        # Size-based filtering
        if stats['size'] < 10:
            is_outlier = True
            reasons.append("size < 10")
        
        # Conductance-based filtering
        if stats['conductance'] > 0.1:
            is_outlier = True
            reasons.append("conductance > 0.1")
        
        # Statistical outlier detection
        z_score = abs(stats['avg_degree'] - mean_degree) / std_degree
        if z_score > 2:
            is_outlier = True
            reasons.append(f"avg_degree z-score > 2 ({z_score:.2f})")
        
        if is_outlier:
            outliers[comm_id] = {
                'reasons': reasons,
                'stats': stats
            }
    
    analysis = {
        'total_communities': len(community_stats),
        'outliers_count': len(outliers),
        'outliers': outliers,
        'mean_degree': mean_degree,
        'std_degree': std_degree,
        'filtering_recommendations': {
            'remove_small': sum(1 for s in community_stats.values() if s['size'] < 10),
            'remove_high_conductance': sum(1 for s in community_stats.values() if s['conductance'] > 0.1),
            'remove_degree_outliers': sum(1 for s in community_stats.values() if abs(s['avg_degree'] - mean_degree) / std_degree > 2)
        }
    }
    
    return analysis

def build_community_network(G, communities):
    """Build a meta-network where nodes are communities"""
    print("Building inter-community network...")
    
    # Group nodes by community
    community_nodes = defaultdict(list)
    for node, comm in communities.items():
        community_nodes[comm].append(node)
    
    # Create community network
    comm_graph = nx.Graph()
    
    # Add community nodes with attributes
    for comm_id, nodes in community_nodes.items():
        subgraph = G.subgraph(nodes)
        
        # Calculate community metrics
        size = len(nodes)
        internal_edges = subgraph.number_of_edges()
        total_degree = sum(G.degree(node) for node in nodes)
        avg_clustering = nx.average_clustering(subgraph) if size > 1 else 0
        density = nx.density(subgraph) if size > 1 else 0
        
        # Calculate proper conductance
        conductance = calculate_proper_conductance(G, set(nodes))
        
        # Add community node with attributes
        comm_graph.add_node(comm_id, 
                           size=size,
                           internal_edges=internal_edges,
                           total_degree=total_degree,
                           density=density,
                           avg_clustering=avg_clustering,
                           conductance=conductance)
    
    # Add edges between communities
    inter_community_edges = defaultdict(lambda: defaultdict(float))
    
    for edge in tqdm(G.edges(data=True), desc="Processing inter-community edges"):
        node1, node2, data = edge
        comm1, comm2 = communities[node1], communities[node2]
        
        if comm1 != comm2:  # Inter-community edge
            weight = data.get('weight', 1.0)
            inter_community_edges[comm1][comm2] += weight
            inter_community_edges[comm2][comm1] += weight
    
    # Add weighted edges between communities
    for comm1, connections in inter_community_edges.items():
        for comm2, total_weight in connections.items():
            if comm1 < comm2:  # Avoid duplicate edges
                # Normalize by community sizes
                size1 = comm_graph.nodes[comm1]['size']
                size2 = comm_graph.nodes[comm2]['size']
                normalized_weight = total_weight / (size1 * size2)
                
                comm_graph.add_edge(comm1, comm2, 
                                   weight=total_weight,
                                   normalized_weight=normalized_weight,
                                   edge_count=1)
    
    print(f"Community network: {comm_graph.number_of_nodes()} communities, {comm_graph.number_of_edges()} inter-community connections")
    return comm_graph, community_nodes

def create_network_overview(G, communities, output_file):
    """Create an overview visualization of the original network"""
    print("Creating network overview visualization...")
    
    # Sample nodes for visualization if network is too large
    if G.number_of_nodes() > 5000:
        print(f"Sampling {5000} nodes for visualization...")
        sampled_nodes = np.random.choice(list(G.nodes()), size=5000, replace=False)
        G_vis = G.subgraph(sampled_nodes)
        communities_vis = {node: communities[node] for node in sampled_nodes if node in communities}
    else:
        G_vis = G
        communities_vis = communities
    
    plt.figure(figsize=(20, 16))
    
    # Create layout
    print("Computing layout...")
    pos = nx.spring_layout(G_vis, k=0.5, iterations=50, seed=42)
    
    # Get unique communities and assign colors
    unique_communities = list(set(communities_vis.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: colors[i] for i, comm in enumerate(unique_communities)}
    
    # Draw edges
    nx.draw_networkx_edges(G_vis, pos, alpha=0.1, width=0.2, edge_color='gray')
    
    # Draw nodes colored by community
    for comm in unique_communities:
        nodes_in_comm = [node for node, c in communities_vis.items() if c == comm]
        if nodes_in_comm:
            nx.draw_networkx_nodes(G_vis, pos, 
                                 nodelist=nodes_in_comm,
                                 node_color=[community_colors[comm]], 
                                 node_size=20,
                                 alpha=0.8)
    
    plt.title(f'MOF Social Network Overview\n{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, {len(set(communities.values()))} communities', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Network overview saved to {output_file}")

def create_community_network_visualization(comm_graph, community_nodes, output_file):
    """Create a beautiful visualization of the community network"""
    print("Creating community network visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Left plot: Community network
    pos = nx.spring_layout(comm_graph, k=3, iterations=100, seed=42)
    
    # Node sizes based on community size
    node_sizes = [comm_graph.nodes[node]['size'] * 2 for node in comm_graph.nodes()]
    max_size = max(node_sizes)
    node_sizes = [size / max_size * 2000 + 100 for size in node_sizes]
    
    # Edge widths based on connection strength
    edge_weights = [comm_graph.edges[edge]['normalized_weight'] for edge in comm_graph.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [weight / max_weight * 10 + 0.5 for weight in edge_weights]
    else:
        edge_widths = [1.0] * len(comm_graph.edges())
    
    # Draw community network
    nx.draw_networkx_edges(comm_graph, pos, width=edge_widths, alpha=0.6, edge_color='gray', ax=ax1)
    nodes = nx.draw_networkx_nodes(comm_graph, pos, node_size=node_sizes, 
                                  node_color=range(len(comm_graph.nodes())), 
                                  cmap='viridis', alpha=0.8, ax=ax1)
    
    # Add labels for larger communities
    large_communities = [node for node in comm_graph.nodes() if comm_graph.nodes[node]['size'] >= 100]
    labels = {node: f"C{node}\n({comm_graph.nodes[node]['size']})" for node in large_communities}
    nx.draw_networkx_labels(comm_graph, pos, labels, font_size=8, ax=ax1)
    
    ax1.set_title(f'Inter-Community Network\n{comm_graph.number_of_nodes()} communities, {comm_graph.number_of_edges()} connections', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right plot: Community size distribution
    community_sizes = [comm_graph.nodes[node]['size'] for node in comm_graph.nodes()]
    ax2.hist(community_sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Community Size (number of MOFs)', fontsize=12)
    ax2.set_ylabel('Number of Communities', fontsize=12)
    ax2.set_title('Community Size Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add statistics
    stats_text = f'Total Communities: {len(community_sizes)}\n'
    stats_text += f'Largest Community: {max(community_sizes):,} MOFs\n'
    stats_text += f'Average Size: {np.mean(community_sizes):.1f} MOFs\n'
    stats_text += f'Median Size: {np.median(community_sizes):.1f} MOFs'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Community network visualization saved to {output_file}")

def analyze_community_properties(G, communities, comm_graph, community_nodes):
    """Analyze detailed properties of communities"""
    print("Analyzing community properties...")
    
    analysis_results = {}
    
    for comm_id, nodes in tqdm(community_nodes.items(), desc="Analyzing communities"):
        subgraph = G.subgraph(nodes)
        
        # Basic metrics
        size = len(nodes)
        internal_edges = subgraph.number_of_edges()
        
        # Network metrics
        if size > 1:
            density = nx.density(subgraph)
            avg_clustering = nx.average_clustering(subgraph)
            diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else np.inf
            avg_path_length = nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) else np.inf
        else:
            density = avg_clustering = diameter = avg_path_length = 0
        
        # Degree statistics
        degrees = [G.degree(node) for node in nodes]
        avg_degree = np.mean(degrees)
        max_degree = max(degrees) if degrees else 0
        
        # External connections
        external_edges = 0
        for node in nodes:
            for neighbor in G.neighbors(node):
                if communities[neighbor] != comm_id:
                    external_edges += 1
        
        # Enhanced conductance calculation
        conductance = calculate_proper_conductance(G, set(nodes))
        
        # Modularity contribution
        modularity_contrib = internal_edges / G.number_of_edges() - (sum(degrees) / (2 * G.number_of_edges()))**2
        
        analysis_results[comm_id] = {
            'size': size,
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'density': density,
            'avg_clustering': avg_clustering,
            'diameter': diameter,
            'avg_path_length': avg_path_length,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'modularity_contrib': modularity_contrib,
            'conductance': conductance
        }
    
    return analysis_results

def create_community_analysis_plots(analysis_results, output_dir):
    """Create comprehensive analysis plots"""
    print("Creating community analysis plots...")
    
    # Convert analysis results to DataFrame
    df = pd.DataFrame.from_dict(analysis_results, orient='index')
    df['community_id'] = df.index
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Community Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Size vs Density
    ax = axes[0, 0]
    scatter = ax.scatter(df['size'], df['density'], c=df['avg_clustering'], 
                        s=50, alpha=0.7, cmap='viridis')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Density')
    ax.set_title('Size vs Density (colored by clustering)')
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax, label='Avg Clustering')
    
    # Plot 2: Internal vs External edges
    ax = axes[0, 1]
    ax.scatter(df['internal_edges'], df['external_edges'], alpha=0.7, c='coral')
    ax.set_xlabel('Internal Edges')
    ax.set_ylabel('External Edges')
    ax.set_title('Internal vs External Connections')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 3: Conductance distribution
    ax = axes[0, 2]
    ax.hist(df['conductance'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Conductance')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Conductance Distribution')
    ax.axvline(df['conductance'].mean(), color='red', linestyle='--', label=f'Mean: {df["conductance"].mean():.3f}')
    ax.legend()
    
    # Plot 4: Average degree vs community size
    ax = axes[1, 0]
    ax.scatter(df['size'], df['avg_degree'], alpha=0.7, c='purple')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Average Degree')
    ax.set_title('Size vs Average Degree')
    ax.set_xscale('log')
    
    # Plot 5: Modularity contribution
    ax = axes[1, 1]
    ax.scatter(df['size'], df['modularity_contrib'], alpha=0.7, c='orange')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Modularity Contribution')
    ax.set_title('Size vs Modularity Contribution')
    ax.set_xscale('log')
    
    # Plot 6: Top communities by size
    ax = axes[1, 2]
    top_communities = df.nlargest(20, 'size')
    bars = ax.bar(range(len(top_communities)), top_communities['size'], color='steelblue')
    ax.set_xlabel('Community Rank')
    ax.set_ylabel('Community Size')
    ax.set_title('Top 20 Largest Communities')
    ax.set_xticks(range(len(top_communities)))
    ax.set_xticklabels([f'C{i}' for i in top_communities.index], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_analysis_dashboard.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save detailed analysis
    df.to_csv(os.path.join(output_dir, 'detailed_community_analysis.csv'), index=False)
    print(f"Community analysis plots saved to {output_dir}")

def create_summary_report(G, communities, comm_graph, analysis_results, algorithm_name, threshold, output_file):
    """Create a comprehensive summary report"""
    print(f"Creating summary report for {algorithm_name}...")
    
    df_analysis = pd.DataFrame.from_dict(analysis_results, orient='index')
    
    # Calculate overall network metrics
    avg_degree = np.mean([G.degree(node) for node in G.nodes()])
    network_density = nx.density(G)
    
    # Calculate modularity
    if algorithm_name.lower() == 'louvain':
        modularity = community_louvain.modularity(communities, G, weight='weight')
    else:
        modularity = nx.community.modularity(G, [set(nodes) for nodes in 
                                                  {comm: [node for node, c in communities.items() if c == comm] 
                                                   for comm in set(communities.values())}.values()], 
                                            weight='weight')
    
    report = f"""
# MOF Social Network Analysis Report - {algorithm_name}

## Analysis Parameters
- **Algorithm**: {algorithm_name}
- **Similarity Threshold**: {threshold}
- **Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Network Overview
- **Total MOFs**: {G.number_of_nodes():,}
- **Total Connections**: {G.number_of_edges():,}
- **Average Degree**: {avg_degree:.2f}
- **Network Density**: {network_density:.6f}

## Community Structure
- **Number of Communities**: {len(set(communities.values()))}
- **Modularity Score**: {modularity:.4f}

### Community Size Statistics
- **Largest Community**: {df_analysis['size'].max():,} MOFs
- **Smallest Community**: {df_analysis['size'].min()} MOFs
- **Average Community Size**: {df_analysis['size'].mean():.1f} MOFs
- **Median Community Size**: {df_analysis['size'].median():.1f} MOFs

### Quality Metrics
- **Average Conductance**: {df_analysis['conductance'].mean():.4f}
- **Average Internal Density**: {df_analysis['density'].mean():.4f}
- **Average Clustering Coefficient**: {df_analysis['avg_clustering'].mean():.4f}

### Top 10 Largest Communities
"""
    
    top_10 = df_analysis.nlargest(10, 'size')
    for i, (comm_id, row) in enumerate(top_10.iterrows()):
        report += f"{i+1:2d}. Community {comm_id}: {row['size']:,} MOFs, Density: {row['density']:.3f}, Conductance: {row['conductance']:.3f}\n"
    
    # Outlier analysis
    outlier_analysis = analyze_average_degree_outliers(analysis_results)
    
    report += f"""

## Outlier Analysis
- **Total Communities**: {outlier_analysis['total_communities']}
- **Outlier Communities**: {outlier_analysis['outliers_count']}
- **Mean Degree**: {outlier_analysis['mean_degree']:.2f}
- **Std Degree**: {outlier_analysis['std_degree']:.2f}

### Filtering Recommendations
- **Small Communities (size < 10)**: {outlier_analysis['filtering_recommendations']['remove_small']}
- **High Conductance (> 0.1)**: {outlier_analysis['filtering_recommendations']['remove_high_conductance']}
- **Degree Outliers (|z| > 2)**: {outlier_analysis['filtering_recommendations']['remove_degree_outliers']}

## Inter-Community Network
- **Community Connections**: {comm_graph.number_of_edges()}
- **Average Inter-Community Connections**: {comm_graph.number_of_edges() / comm_graph.number_of_nodes():.2f}

## Recommendations
1. **Large Communities**: Focus on the top {min(5, len(top_10))} communities which contain {top_10['size'].sum():,} MOFs ({100*top_10['size'].sum()/G.number_of_nodes():.1f}% of total)
2. **High-Quality Communities**: Communities with low conductance (<0.1) represent well-separated clusters
3. **Dense Communities**: Communities with high density (>0.5) show strong internal cohesion
4. **Outlier Removal**: Consider removing {outlier_analysis['outliers_count']} outlier communities for cleaner analysis
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {output_file}")
    return outlier_analysis

def run_single_threshold_analysis(threshold, data_file, output_base_dir, algorithms=['louvain', 'girvan_newman']):
    """
    Run complete analysis for a single threshold
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING THRESHOLD {threshold}")
    print(f"{'='*60}")
    
    # Import from the main MOF social network script
    import sys
    code_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Code')
    sys.path.append(code_dir)
    from mof_social_network import load_and_preprocess_data, calculate_similarity_matrix, save_adjacency_matrix
    
    # Create threshold-specific output directory
    threshold_dir = os.path.join(output_base_dir, 'threshold_analysis')
    os.makedirs(threshold_dir, exist_ok=True)
    
    # Load and preprocess data
    df, id_column = load_and_preprocess_data(data_file)
    
    # Calculate adjacency matrix for this threshold
    print(f"Calculating similarity matrix for threshold {threshold}...")
    adjacency_dict = calculate_similarity_matrix(
        df, id_column, 
        threshold=threshold, 
        batch_size=10000,
        top_k=100,
        n_jobs=4  # Use parallel processing
    )
    
    # Save adjacency matrix
    adj_file = os.path.join(threshold_dir, f'adjacency_matrix_t{threshold}.pkl')
    save_adjacency_matrix(adjacency_dict, adj_file)
    
    # Save metadata
    metadata = {
        'num_nodes': len(df),
        'num_edges': len(adjacency_dict),
        'threshold': threshold,
        'id_column': id_column,
        'processing_time': time.time()
    }
    metadata_file = os.path.join(threshold_dir, f'adjacency_metadata_t{threshold}.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Build network
    G = build_network_from_adjacency(adjacency_dict)
    
    results = {}
    
    # Run algorithms
    for algorithm in algorithms:
        print(f"\nRunning {algorithm} algorithm...")
        
        # Create algorithm-specific output directory
        alg_dir = os.path.join(threshold_dir, f'{algorithm}_t{threshold}')
        os.makedirs(alg_dir, exist_ok=True)
        
        # Detect communities
        if algorithm == 'louvain':
            communities, modularity = detect_communities_enhanced(G, resolution=1.0)
        elif algorithm == 'girvan_newman':
            communities, modularity = detect_communities_girvan_newman_enhanced(G, max_communities=50)
        else:
            continue
        
        # Build community network
        comm_graph, community_nodes = build_community_network(G, communities)
        
        # Analyze communities
        analysis_results = analyze_community_properties(G, communities, comm_graph, community_nodes)
        
        # Create visualizations
        create_network_overview(G, communities, 
                              os.path.join(alg_dir, 'network_overview.png'))
        
        create_community_network_visualization(comm_graph, community_nodes,
                                             os.path.join(alg_dir, 'community_network.png'))
        
        create_community_analysis_plots(analysis_results, alg_dir)
        
        # Create summary report
        outlier_analysis = create_summary_report(G, communities, comm_graph, analysis_results, 
                                                algorithm, threshold,
                                                os.path.join(alg_dir, 'analysis_report.md'))
        
        # Save community assignments
        community_df = pd.DataFrame(list(communities.items()), columns=['MOF_ID', 'Community_ID'])
        community_df.to_csv(os.path.join(alg_dir, 'community_assignments.csv'), index=False)
        
        # Save results for comparison
        results[algorithm] = {
            'num_communities': len(set(communities.values())),
            'modularity': modularity,
            'avg_conductance': np.mean([r['conductance'] for r in analysis_results.values()]),
            'avg_degree': np.mean([G.degree(node) for node in G.nodes()]),
            'largest_community': max(Counter(communities.values()).values()),
            'outlier_analysis': outlier_analysis,
            'analysis_results': analysis_results
        }
        
        print(f"{algorithm} analysis completed for threshold {threshold}")
    
    print(f"Threshold {threshold} analysis completed!")
    return results

def create_threshold_comparison_analysis(all_results, output_dir):
    """
    Create comprehensive comparison analysis across thresholds and algorithms
    """
    print("Creating threshold comparison analysis...")
    
    comparison_dir = os.path.join(output_dir, 'comparison_analysis')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Prepare comparison data
    comparison_data = []
    
    for threshold, threshold_results in all_results.items():
        for algorithm, results in threshold_results.items():
            comparison_data.append({
                'threshold': threshold,
                'algorithm': algorithm,
                'num_communities': results['num_communities'],
                'modularity': results['modularity'],
                'avg_conductance': results['avg_conductance'],
                'avg_degree': results['avg_degree'],
                'largest_community': results['largest_community'],
                'outliers_count': results['outlier_analysis']['outliers_count'],
                'total_communities': results['outlier_analysis']['total_communities']
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save comparison table
    df_comparison.to_csv(os.path.join(comparison_dir, 'threshold_algorithm_comparison.csv'), index=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Threshold and Algorithm Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of communities vs threshold
    ax = axes[0, 0]
    for algorithm in df_comparison['algorithm'].unique():
        alg_data = df_comparison[df_comparison['algorithm'] == algorithm]
        ax.plot(alg_data['threshold'], alg_data['num_communities'], 
                marker='o', label=algorithm, linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Communities vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Modularity vs threshold
    ax = axes[0, 1]
    for algorithm in df_comparison['algorithm'].unique():
        alg_data = df_comparison[df_comparison['algorithm'] == algorithm]
        ax.plot(alg_data['threshold'], alg_data['modularity'], 
                marker='s', label=algorithm, linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Modularity')
    ax.set_title('Modularity vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average conductance vs threshold
    ax = axes[0, 2]
    for algorithm in df_comparison['algorithm'].unique():
        alg_data = df_comparison[df_comparison['algorithm'] == algorithm]
        ax.plot(alg_data['threshold'], alg_data['avg_conductance'], 
                marker='^', label=algorithm, linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Average Conductance')
    ax.set_title('Conductance vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average degree vs threshold
    ax = axes[1, 0]
    for algorithm in df_comparison['algorithm'].unique():
        alg_data = df_comparison[df_comparison['algorithm'] == algorithm]
        ax.plot(alg_data['threshold'], alg_data['avg_degree'], 
                marker='d', label=algorithm, linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Average Degree')
    ax.set_title('Average Degree vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Largest community size vs threshold
    ax = axes[1, 1]
    for algorithm in df_comparison['algorithm'].unique():
        alg_data = df_comparison[df_comparison['algorithm'] == algorithm]
        ax.plot(alg_data['threshold'], alg_data['largest_community'], 
                marker='v', label=algorithm, linewidth=2)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Largest Community Size')
    ax.set_title('Largest Community vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Performance comparison heatmap
    ax = axes[1, 2]
    pivot_data = df_comparison.pivot(index='algorithm', columns='threshold', values='modularity')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Modularity Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'threshold_comparison_plots.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create performance ranking
    df_comparison['performance_score'] = (
        df_comparison['modularity'] * 0.4 +  # Modularity weight
        (1 - df_comparison['avg_conductance']) * 0.3 +  # Conductance weight (inverted)
        (df_comparison['num_communities'] / df_comparison['num_communities'].max()) * 0.3  # Community count weight
    )
    
    performance_ranking = df_comparison.sort_values('performance_score', ascending=False)
    performance_ranking.to_csv(os.path.join(comparison_dir, 'performance_ranking.csv'), index=False)
    
    print(f"Comparison analysis saved to {comparison_dir}")
    return df_comparison

def main():
    parser = argparse.ArgumentParser(description='Enhanced Community Network Analysis with Multi-Threshold Support')
    parser.add_argument('--data_file', required=True,
                       help='Path to the original MOF data CSV file')
    parser.add_argument('--output_dir', default='enhanced_community_results', 
                       help='Output directory for results')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help='Similarity thresholds to test')
    parser.add_argument('--algorithms', nargs='+', choices=['louvain', 'girvan_newman'], 
                       default=['louvain', 'girvan_newman'],
                       help='Algorithms to run')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Resolution parameter for community detection (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Enhanced MOF Social Network Analysis")
    print(f"Testing thresholds: {args.thresholds}")
    print(f"Using algorithms: {args.algorithms}")
    
    # Run analysis for each threshold
    all_results = {}
    
    for threshold in args.thresholds:
        try:
            results = run_single_threshold_analysis(
                threshold, 
                args.data_file, 
                args.output_dir, 
                args.algorithms
            )
            all_results[threshold] = results
        except Exception as e:
            print(f"Error analyzing threshold {threshold}: {e}")
            continue
    
    # Create comprehensive comparison
    if len(all_results) > 1:
        df_comparison = create_threshold_comparison_analysis(all_results, args.output_dir)
        
        # Save all results summary
        with open(os.path.join(args.output_dir, 'all_results_summary.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for threshold, threshold_results in all_results.items():
                json_results[str(threshold)] = {}
                for algorithm, results in threshold_results.items():
                    json_results[str(threshold)][algorithm] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in results.items() 
                        if k not in ['analysis_results', 'outlier_analysis']  # Skip complex nested objects
                    }
            json.dump(json_results, f, indent=2)
    
    print(f"\nEnhanced analysis completed! Results saved to {args.output_dir}")
    print(f"Key outputs:")
    print(f"  - threshold_analysis/: Individual threshold results")
    print(f"  - comparison_analysis/: Cross-threshold comparisons")
    print(f"  - all_results_summary.json: Summary of all results")

if __name__ == "__main__":
    main() 