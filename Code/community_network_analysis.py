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
from scipy.stats import pearsonr
import warnings
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
        
        # Add community node with attributes
        comm_graph.add_node(comm_id, 
                           size=size,
                           internal_edges=internal_edges,
                           total_degree=total_degree,
                           density=density,
                           avg_clustering=avg_clustering)
    
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
            'conductance': external_edges / (2 * internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
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

def create_summary_report(G, communities, comm_graph, analysis_results, output_file):
    """Create a comprehensive summary report"""
    print("Creating summary report...")
    
    df_analysis = pd.DataFrame.from_dict(analysis_results, orient='index')
    
    report = f"""
# MOF Social Network Analysis Report

## Network Overview
- **Total MOFs**: {G.number_of_nodes():,}
- **Total Connections**: {G.number_of_edges():,}
- **Average Degree**: {np.mean([G.degree(node) for node in G.nodes()]):.2f}
- **Network Density**: {nx.density(G):.6f}

## Community Structure
- **Number of Communities**: {len(set(communities.values()))}
- **Modularity Score**: {community_louvain.modularity(communities, G, weight='weight'):.4f}

### Community Size Statistics
- **Largest Community**: {df_analysis['size'].max():,} MOFs
- **Smallest Community**: {df_analysis['size'].min()} MOFs
- **Average Community Size**: {df_analysis['size'].mean():.1f} MOFs
- **Median Community Size**: {df_analysis['size'].median():.1f} MOFs

### Top 10 Largest Communities
"""
    
    top_10 = df_analysis.nlargest(10, 'size')
    for i, (comm_id, row) in enumerate(top_10.iterrows()):
        report += f"{i+1:2d}. Community {comm_id}: {row['size']:,} MOFs, Density: {row['density']:.3f}, Clustering: {row['avg_clustering']:.3f}\n"
    
    report += f"""

## Inter-Community Network
- **Community Connections**: {comm_graph.number_of_edges()}
- **Average Inter-Community Connections**: {comm_graph.number_of_edges() / comm_graph.number_of_nodes():.2f}

## Quality Metrics
- **Average Conductance**: {df_analysis['conductance'].mean():.4f}
- **Average Internal Density**: {df_analysis['density'].mean():.4f}
- **Average Clustering Coefficient**: {df_analysis['avg_clustering'].mean():.4f}

## Recommendations
1. **Large Communities**: Focus on the top {min(5, len(top_10))} communities which contain {top_10['size'].sum():,} MOFs ({100*top_10['size'].sum()/G.number_of_nodes():.1f}% of total)
2. **High-Quality Communities**: Communities with low conductance (<0.1) represent well-separated clusters
3. **Dense Communities**: Communities with high density (>0.5) show strong internal cohesion
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Community Network Analysis and Visualization')
    parser.add_argument('--adjacency_matrix', required=True, 
                       help='Path to the adjacency matrix pickle file')
    parser.add_argument('--metadata', required=True,
                       help='Path to the metadata pickle file')
    parser.add_argument('--data_file', required=True,
                       help='Path to the original MOF data CSV file')
    parser.add_argument('--output_dir', default='community_results', 
                       help='Output directory for results')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Resolution parameter for community detection (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    adjacency_dict = load_adjacency_matrix(args.adjacency_matrix)
    metadata = load_metadata(args.metadata)
    
    # Build network
    G = build_network_from_adjacency(adjacency_dict)
    
    # Detect communities
    communities, modularity = detect_communities_enhanced(G, resolution=args.resolution)
    
    # Build community network
    comm_graph, community_nodes = build_community_network(G, communities)
    
    # Create visualizations
    create_network_overview(G, communities, 
                          os.path.join(args.output_dir, 'network_overview.png'))
    
    create_community_network_visualization(comm_graph, community_nodes,
                                         os.path.join(args.output_dir, 'community_network.png'))
    
    # Analyze communities
    analysis_results = analyze_community_properties(G, communities, comm_graph, community_nodes)
    
    # Create analysis plots
    create_community_analysis_plots(analysis_results, args.output_dir)
    
    # Create summary report
    create_summary_report(G, communities, comm_graph, analysis_results,
                         os.path.join(args.output_dir, 'analysis_report.md'))
    
    # Save community assignments
    community_df = pd.DataFrame(list(communities.items()), columns=['MOF_ID', 'Community_ID'])
    community_df.to_csv(os.path.join(args.output_dir, 'community_assignments.csv'), index=False)
    
    # Save community network
    nx.write_gexf(comm_graph, os.path.join(args.output_dir, 'community_network.gexf'))
    
    print(f"\nAnalysis completed! Results saved to {args.output_dir}")
    print(f"Key files generated:")
    print(f"  - network_overview.png: Overview of the original network")
    print(f"  - community_network.png: Inter-community network visualization")
    print(f"  - community_analysis_dashboard.png: Comprehensive analysis plots")
    print(f"  - analysis_report.md: Detailed summary report")
    print(f"  - community_assignments.csv: MOF to community mapping")
    print(f"  - detailed_community_analysis.csv: Detailed community metrics")

if __name__ == "__main__":
    main() 