import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg') 


class MOFAnalysisController:
    def __init__(self, adj_file, communities_file, output_dir):
        self.adj_file = adj_file
        self.communities_file = communities_file
        self.output_dir = output_dir
        self.G = None
        self.communities = {}
        self.centralities = {}
        self.community_analysis = {}
        self.bridge_mofs = {}  # Initialize bridge_mofs to avoid AttributeError
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load network and community data with file existence checks."""
        print("loading data...")
        if not os.path.exists(self.adj_file):
            raise FileNotFoundError(f"Adjacency file not found: {self.adj_file}")
        if not os.path.exists(self.communities_file):
            raise FileNotFoundError(f"Communities file not found: {self.communities_file}")
        with open(self.adj_file, 'rb') as f:
            adj_matrix = pickle.load(f)
        
        self.G = nx.Graph()
        for(n1, n2), weight in adj_matrix.items():
            self.G.add_edge(n1, n2, weight=weight)
            
        print(f"loaded {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        if self.communities_file.endswith('.pkl'):
            comm_df = pd.read_pickle(self.communities_file)
            if 'MOF_ID' in comm_df.columns and 'Community_ID' in comm_df.columns:
                self.communities = dict(zip(comm_df['MOF_ID'], comm_df['Community_ID']))
            else:
                self.communities = dict(zip(comm_df.iloc[:, 0], comm_df.iloc[:, 1]))
        elif self.communities_file.endswith('.csv'):
            comm_df = pd.read_csv(self.communities_file)
            if 'MOF_ID' in comm_df.columns and 'Community_ID' in comm_df.columns:
                self.communities = dict(zip(comm_df['MOF_ID'], comm_df['Community_ID']))
            else:
                self.communities = dict(zip(comm_df.iloc[:, 0], comm_df.iloc[:, 1]))
        else:
            with open(self.communities_file, 'rb') as f:
                self.communities = pickle.load(f)
        
        print(f"loaded {len(set(self.communities.values()))} communities")
    
    def calculate_centralities(self):
        """Calculate degree, closeness, and betweenness centrality with error protection."""
        print("caclulating centrality measures...")
        
        self.centralities['degree'] = dict(nx.degree_centrality(self.G))
        
        if nx.is_connected(self.G):
            self.centralities['closeness'] = dict(nx.closeness_centrality(self.G))
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            closeness_centrality = nx.closeness_centrality(subgraph)
            
            self.centralities['closeness'] = {}
            for node in self.G.nodes():
                self.centralities['closeness'][node] = closeness_centrality.get(node, 0.0)

        print("caclulating betweenness centrality...")
        self.centralities['betweenness'] = nx.betweenness_centrality(self.G, k=min(1000, self.G.number_of_nodes()))
         
    def analyze_community_centralities(self):
        """Analyze and store community-level centralities and representatives."""
        print('analyzing community-level centralities...')

        for comm_id in set(self.communities.values()):
            comm_mofs = [mof for mof, c in self.communities.items() if c == comm_id]
            
            if len(comm_mofs) < 2:
                continue
            
            subgraph = self.G.subgraph(comm_mofs)
            
            comm_degree = nx.degree_centrality(subgraph) if subgraph.number_of_edges() > 0 else {}
            comm_closeness = nx.closeness_centrality(subgraph) if nx.is_connected(subgraph) and subgraph.number_of_edges() > 0 else {}
            
            if comm_degree:
                representative_mof = max(comm_degree, key=lambda x: comm_degree[x])
            else:
                representative_mof = comm_mofs[0]  # Always just the MOF ID
            
            avg_global_degree_centrality = np.mean([self.centralities['degree'].get(mof, 0) for mof in comm_mofs])
            self.community_analysis[comm_id] = {
                'size': len(comm_mofs),
                'representative_mof': representative_mof,
                'avg_global_degree_centrality': avg_global_degree_centrality,
                'mofs': comm_mofs
            }
    
    def plot_degree_distribution(self):
        """Plot and save the degree distribution of the network."""
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        if not degrees:
            print("No degrees to plot.")
            return
        plt.figure(figsize=(8,6))
        sns.histplot(degrees, bins=50, kde=False, color='skyblue')
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title("Degree Distribution of MOF Network")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "degree_distribution.png"), dpi=300)
        plt.close()
        print("Saved degree distribution plot.")

    def export_gexf(self):
        """Export the network as a GEXF file for Gephi."""
        nx.write_gexf(self.G, os.path.join(self.output_dir, "mof_network.gexf"))
        print("exported GEXF for Gephi.")

    def export_gephi_csv(self):
        """Export nodes and edges as Gephi-compatible CSVs."""
        nodes_data = []
        for node in self.G.nodes():
            nodes_data.append({
                "Id": node,
                "Degree": self.G.degree(node),
                "Degree_Centrality": self.centralities['degree'].get(node, 0),
                "Closeness_Centrality": self.centralities['closeness'].get(node, 0),
                "Betweenness_Centrality": self.centralities['betweenness'].get(node, 0),
                "Community": self.communities.get(node, "Unknown")
            })
        pd.DataFrame(nodes_data).to_csv(os.path.join(self.output_dir, "gephi_nodes.csv"), index=False)

        edges_data = []
        for u, v, d in self.G.edges(data=True):
            edges_data.append({
                "Source": u,
                "Target": v,
                "Weight": d.get("weight", 1.0)
            })
        pd.DataFrame(edges_data).to_csv(os.path.join(self.output_dir, "gephi_edges.csv"), index=False)
        print("exported Gephi CSVs for nodes and edges.")

    def find_bridge_mofs(self):
        """Identify MOFs that bridge communities (novel analysis)"""
        print("Identifying bridge MOFs between communities...")
        bridge_analysis = {}
        for mof in self.G.nodes():
            neighbors = list(self.G.neighbors(mof))
            mof_community = self.communities.get(mof)
            community_connections = defaultdict(int)
            for neighbor in neighbors:
                neighbor_community = self.communities.get(neighbor)
                if neighbor_community != mof_community:
                    community_connections[neighbor_community] += 1
            total_external_connections = sum(community_connections.values())
            num_communities_connected = len(community_connections)
            if total_external_connections > 0:
                bridge_analysis[mof] = {
                    'external_connections': total_external_connections,
                    'communities_connected': num_communities_connected,
                    'own_community': mof_community,
                    'bridge_score': total_external_connections * num_communities_connected,
                    'degree_centrality': self.centralities['degree'].get(mof, 0),
                    'closeness_centrality': self.centralities['closeness'].get(mof, 0)
                }
        self.bridge_mofs = bridge_analysis
        print(f"Identified {len(bridge_analysis)} bridge MOFs")

    def run_integrated_analysis(self):
        """Run the full integrated analysis workflow."""
        print("=" * 60)
        print("integrated MOF Analysis: Communities + Centrality")
        print("=" * 60)
        self.load_data()
        self.calculate_centralities()
        self.analyze_community_centralities()
        if not self.community_analysis:
            print("Warning: No communities with more than one MOF found. Skipping community-based analysis.")
        self.find_bridge_mofs()
        self.create_comprehensive_visualizations()
        self.create_comparison_report()
        self.plot_degree_distribution()
        self.export_gexf()
        self.export_gephi_csv()
        print("\n" + "=" * 60)
        print("Integrated Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
    
    def create_comprehensive_visualizations(self):
        """Create and save comprehensive visualizations of the analysis."""
        print("Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        ax = axes[0, 0]
        community_sizes = [self.community_analysis[c]['size'] for c in self.community_analysis.keys()]
        community_centralities = [self.community_analysis[c]['avg_global_degree_centrality'] 
                                for c in self.community_analysis.keys()]
        
        ax.scatter(community_sizes, community_centralities, alpha=0.7, s=60)
        ax.set_xlabel('Community Size')
        ax.set_ylabel('Average Degree Centrality')
        ax.set_title('Community Size vs Average Centrality')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        
        large_communities = [c for c in self.community_analysis.keys() 
                           if self.community_analysis[c]['size'] >= 1000]
        medium_communities = [c for c in self.community_analysis.keys() 
                            if 100 <= self.community_analysis[c]['size'] < 1000]
        small_communities = [c for c in self.community_analysis.keys() 
                           if self.community_analysis[c]['size'] < 100]
        
        large_cents = []
        medium_cents = []
        small_cents = []
        
        for comm_id in large_communities:
            mofs = self.community_analysis[comm_id]['mofs']
            large_cents.extend([self.centralities['degree'].get(mof, 0) for mof in mofs[:100]])  # Sample for speed
        
        for comm_id in medium_communities:
            mofs = self.community_analysis[comm_id]['mofs']
            medium_cents.extend([self.centralities['degree'].get(mof, 0) for mof in mofs])
            
        for comm_id in small_communities:
            mofs = self.community_analysis[comm_id]['mofs']
            small_cents.extend([self.centralities['degree'].get(mof, 0) for mof in mofs])
        
        data_to_plot = [large_cents, medium_cents, small_cents]
        labels = [f'Large\n(≥1000)\nn={len(large_communities)}', 
                 f'Medium\n(100-999)\nn={len(medium_communities)}',
                 f'Small\n(<100)\nn={len(small_communities)}']
        
        data_and_labels = [(d, l) for d, l in zip(data_to_plot, labels) if len(d) > 0]
        if data_and_labels:
            data, used_labels = zip(*data_and_labels)
            ax.boxplot(data, labels=used_labels)
        else:
            ax.text(0.5, 0.5, "No data for boxplot", ha='center', va='center')
        
        ax.set_ylabel('Degree Centrality')
        ax.set_title('Centrality Distribution by Community Size')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        
        top_degree = sorted(self.centralities['degree'].items(), key=lambda x: x[1], reverse=True)[:10]
        top_closeness = sorted(self.centralities['closeness'].items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(self.centralities['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10]
        
        positions = np.arange(10)
        width = 0.25
        
        degree_values = [cent for _, cent in top_degree]
        closeness_values = [self.centralities['closeness'].get(mof, 0) for mof, _ in top_degree]
        betweenness_values = [self.centralities['betweenness'].get(mof, 0) for mof, _ in top_degree]
        
        degree_norm = np.array(degree_values) / max(degree_values) if degree_values and max(degree_values) > 0 else np.zeros_like(degree_values)
        closeness_norm = np.array(closeness_values) / max(closeness_values) if closeness_values and max(closeness_values) > 0 else np.zeros_like(closeness_values)
        betweenness_norm = np.array(betweenness_values) / max(betweenness_values) if betweenness_values and max(betweenness_values) > 0 else np.zeros_like(betweenness_values)
        
        ax.bar(positions - width, degree_norm, width, label='Degree', alpha=0.8)
        ax.bar(positions, closeness_norm, width, label='Closeness', alpha=0.8)
        ax.bar(positions + width, betweenness_norm, width, label='Betweenness', alpha=0.8)
        
        ax.set_xlabel('Top 10 MOFs (by Degree Centrality)')
        ax.set_ylabel('Normalized Centrality')
        ax.set_title('Centrality Measure Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        
        if self.bridge_mofs:
            bridge_scores = [stats['bridge_score'] for stats in self.bridge_mofs.values()]
            communities_connected = [stats['communities_connected'] for stats in self.bridge_mofs.values()]
            
            ax.scatter(communities_connected, bridge_scores, alpha=0.7, s=60)
            ax.set_xlabel('Number of Communities Connected')
            ax.set_ylabel('Bridge Score')
            ax.set_title('Bridge MOFs: Inter-Community Connections')
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        
        rep_communities = []
        rep_centralities = []
        rep_sizes = []
        
        for comm_id, analysis in self.community_analysis.items():
            if analysis['size'] >= 10:
                rep_communities.append(comm_id)
                rep_mof = analysis['representative_mof']
                rep_centralities.append(self.centralities['degree'].get(rep_mof, 0))
                rep_sizes.append(analysis['size'])
        
        scatter = ax.scatter(rep_sizes, rep_centralities, 
                           c=rep_communities, cmap='tab20', alpha=0.7, s=60)
        ax.set_xlabel('Community Size')
        ax.set_ylabel('Representative MOF Degree Centrality')
        ax.set_title('Community Representatives')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        
        if self.G.number_of_nodes() > 2000:
            important_nodes = set()
            important_nodes.update([mof for mof, _ in sorted(self.centralities['degree'].items(), 
                                                           key=lambda x: x[1], reverse=True)[:50]])
            important_nodes.update([analysis['representative_mof'] for analysis in self.community_analysis.values()])
            if self.bridge_mofs:
                important_nodes.update([mof for mof, _ in sorted(self.bridge_mofs.items(),
                                                               key=lambda x: x[1]['bridge_score'], reverse=True)[:20]])
            
            remaining = list(set(self.G.nodes()) - important_nodes)
            sample_size = min(1000, len(remaining))
            sampled = np.random.choice(remaining, sample_size, replace=False)
            
            viz_nodes = list(important_nodes) + list(sampled)
            G_viz = self.G.subgraph(viz_nodes)
        else:
            G_viz = self.G
        
        pos = nx.spring_layout(G_viz, k=0.5, iterations=20, seed=42)
        
        node_colors = [self.communities.get(node, -1) for node in G_viz.nodes()]
        node_sizes = [self.centralities['degree'].get(node, 0) * 500 + 20 for node in G_viz.nodes()]
        
        nx.draw_networkx_nodes(G_viz, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, cmap='tab20')
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5)
        
        ax.set_title('Network: Communities & Centrality')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
