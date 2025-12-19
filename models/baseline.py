"""
Simple baseline model for comparison with GraphSAGE.
"""

import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    """
    Baseline model for link prediction based on node features.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, node_features):
        """
        Predicts the probability of a link between all pairs of nodes.
        
        Args:
            node_features: Tensor of node features [num_nodes, feature_dim]
            
        Returns:
            Matrix of link probabilities [num_nodes, num_nodes]
        """
        num_nodes = node_features.shape[0]
        
        # Create feature pairs for all possible links
        features_i = node_features.unsqueeze(1).expand(-1, num_nodes, -1)
        features_j = node_features.unsqueeze(0).expand(num_nodes, -1, -1)
        pair_features = torch.cat([features_i, features_j], dim=-1)
        
        # Pass through the network
        x = torch.relu(self.fc1(pair_features))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x).squeeze(-1)
        
        return torch.sigmoid(x)