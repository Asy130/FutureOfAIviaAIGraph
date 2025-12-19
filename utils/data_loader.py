# utils/data_loader.py - UPDATED VERSION
import pickle
import torch
import numpy as np
import os
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

class SemanticGraphLoader:
    def __init__(self, filepath):
        """
        Loader for SemanticGraph_*.pkl files
        Format: [train_edges, candidate_pairs, labels, year_start, delta, cutoff, min_edges]
        """
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        
        logger.info(f"Loading {filepath}")
        with open(filepath, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Extract components
        self.train_edges = raw_data[0]          # [N, 3] - [u, v, year]
        self.candidate_pairs = raw_data[1]      # [10_000_000, 2] - prediction pairs
        self.labels = raw_data[2]               # [10_000_000,] - labels (1/0)
        
        # Metadata
        self.year_start = raw_data[3]
        self.delta = raw_data[4]
        self.cutoff = raw_data[5]
        self.min_edges = raw_data[6]
        
        logger.info(f"Loaded: {len(self.train_edges)} training edges, "
                   f"{len(self.candidate_pairs)} candidate pairs, "
                   f"positives: {(self.labels == 1).sum()}, "
                   f"negatives: {(self.labels == 0).sum()}")
    
    def get_training_graph_tensors(self):
        """Returns tensors for training graph (edges only)."""
        # Take only nodes and edges (ignore year)
        edges_np = self.train_edges[:, :2].astype(np.int64)
        edge_index = torch.tensor(edges_np.T, dtype=torch.long)
        return edge_index
    
    def get_node_features(self, feature_dim=128):
        """Creates node features."""
        # Find all unique nodes
        all_edges = self.train_edges[:, :2].flatten()
        all_candidates = self.candidate_pairs.flatten()
        all_nodes = np.unique(np.concatenate([all_edges, all_candidates]))
        
        num_nodes = len(all_nodes)
        
        # Create mapping: original ID -> index 0..N-1
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Random features (can be replaced with meaningful ones)
        features = torch.randn(num_nodes, feature_dim)
        
        return features, num_nodes
    
    def get_candidate_tensors(self):
        """Returns tensors for candidate pairs with transformed indices."""
        # Convert node IDs to 0-based indices
        candidate_indices = np.zeros_like(self.candidate_pairs)
        for i in range(len(self.candidate_pairs)):
            for j in range(2):
                node = self.candidate_pairs[i, j]
                candidate_indices[i, j] = self.node_to_idx[node]
        
        edge_label_index = torch.tensor(candidate_indices.T, dtype=torch.long)
        edge_label = torch.tensor(self.labels, dtype=torch.float)
        
        return edge_label_index, edge_label
    
    def to_pyg_data(self, feature_dim=128):
        """
        Converts to PyTorch Geometric Data format.
        """
        # 1. Get node features
        x, num_nodes = self.get_node_features(feature_dim)
        
        # 2. Get graph edges (with transformed indices)
        edges_np = self.train_edges[:, :2]
        edge_indices = np.zeros_like(edges_np)
        
        for i in range(len(edges_np)):
            for j in range(2):
                node = edges_np[i, j]
                edge_indices[i, j] = self.node_to_idx[node]
        
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
        
        # 3. Get candidate pairs
        edge_label_index, edge_label = self.get_candidate_tensors()
        
        # 4. Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            num_nodes=num_nodes
        )
        
        # Save metadata
        data.year_start = self.year_start
        data.delta = self.delta
        data.cutoff = self.cutoff
        data.min_edges = self.min_edges
        
        return data
    
    def get_info(self):
        """Returns dataset information."""
        return {
            'file': os.path.basename(self.filepath),
            'train_edges': len(self.train_edges),
            'candidate_pairs': len(self.candidate_pairs),
            'positives': int((self.labels == 1).sum()),
            'negatives': int((self.labels == 0).sum()),
            'year_start': self.year_start,
            'delta': self.delta,
            'cutoff': self.cutoff,
            'min_edges': self.min_edges
        }