"""
GraphSAGE model for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, ModuleList
from typing import Optional, Tuple

class GraphSAGELinkPred(nn.Module):
    """
    GraphSAGE model for link prediction.
    
    Args:
        in_channels: Dimension of input node features
        hidden_channels: Dimension of hidden layers
        out_channels: Dimension of output node embeddings
        num_layers: Number of GraphSAGE layers
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 num_layers: int = 2, 
                 dropout: float = 0.0):
        super().__init__()
        
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.lin = Linear(2 * out_channels, 1)
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encodes nodes into embeddings.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Predicts link probability for node pairs.
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_label_index: Edge indices for prediction [2, num_edges_to_predict]
            
        Returns:
            Predicted probabilities [num_edges_to_predict]
        """
        edge_features = torch.cat([
            z[edge_label_index[0]], 
            z[edge_label_index[1]]
        ], dim=-1)
        
        return self.lin(edge_features).view(-1)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encoding and decoding.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            edge_label_index: Edge indices for prediction
            
        Returns:
            Predicted link probabilities
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
    def predict_all_pairs(self, 
                         x: torch.Tensor, 
                         edge_index: torch.Tensor,
                         node_pairs: torch.Tensor) -> torch.Tensor:
        """
        Predicts probabilities for given node pairs.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            node_pairs: Node pairs for prediction [2, num_pairs]
            
        Returns:
            Link probabilities [num_pairs]
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x, edge_index)
            predictions = self.decode(z, node_pairs)
            return torch.sigmoid(predictions)
    
    def loss(self, pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes loss from predictions for positive and negative edges.
        
        Args:
            pos_pred: Predictions for positive edges
            neg_pred: Predictions for negative edges
            
        Returns:
            Loss value
        """
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, 
            torch.ones_like(pos_pred)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, 
            torch.zeros_like(neg_pred)
        )
        
        return pos_loss + neg_loss