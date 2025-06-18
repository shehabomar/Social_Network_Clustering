import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter, deafultdict


class MOFAnalysisController:
    def __init__(self, adj_file, communities_file, output_dir='../results/integrated_analysis'):
        self.adj_file = adj_file
        self.communities_file = communities_file
        self.output_dir = output_dir
        self.G = None
        self.communities = {}
        self.centralities = {}
        self.communities_analysis = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        print("loading data...")
        with open(self.adj_file, 'rb') as f:
            adj_matrix = pickle.load(f)
        
        self.G = nx.Graph()
        for(n1, n2), weight in adj_matrix.items():
            self.G.add_edge(n1, n2, weight=weight)
            
        print(f"loaded {len(self.G.number_of_nodes())} nodes and {len(self.G.edges())} edges")
        
        if self.communities_file.endswith('.pkl'):
            comm_df = pd.read_pickle(self.communities_file)
            if 'MOF_ID' in comm_df.columns and 'Community_ID' in comm_df.columns:
                self.communities = dict(zip(comm_df['MOF_ID'], comm_df['Community_ID']))
            else:
                self.communities = dict(zip(comm_df.iloc[:, 0], comm_df.iloc[:, 1]))
                
        else:
            with open(self.communities_file, 'rb') as f:
                self.communities = pickle.load(f)
        
        print(f"loaded {len(set(self.communities.values()))} communities")
    
    def calculate_centralities(self):
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
        print('analyzing community-level centralities...')

        for comm_id in set(self.communities.values()):
            comm_mofs = [mof for mof, c in self.communities.items() if c == comm_id]
            
            if len(comm_mofs) < 2:
                continue
            
            subgraph = self.G.subgraph(comm_mofs)
            
            comm_degree = nx.degree_centrality(subgraph) if subgraph.number_of_edges() > 0 else {}
            comm_closeness = nx.closeness_centrality(subgraph) if nx.is_connected(subgraph) else {}
            
            if comm_degree:
                representative_mof = max(comm_degree, key=lambda x: comm_degree[x])
            else:
                representative_mof = (comm_mofs[0], 0)
            
            degrees = [self.G.degree(mof) for mof in comm_mofs]
            