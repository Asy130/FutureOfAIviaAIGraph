#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.graphsage_linkpred import GraphSAGELinkPred
    from utils.data_loader import SemanticGraphLoader
    from utils.metrics import calculate_auc_roc
    import torch
    import torch.nn.functional as F
    print(" All imports successful!")
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)

import argparse
import numpy as np

def split_data(data, train_ratio=0.8, val_ratio=0.1):
    n = data.edge_label_index.shape[1]
    indices = torch.randperm(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

def train_epoch(model, data, optimizer, device, batch_size=20000):
    model.train()
    
    train_idx = data.train_mask.nonzero().squeeze()
    edge_label_index = data.edge_label_index[:, train_idx]
    edge_label = data.edge_label[train_idx]
    
    pos_idx = (edge_label == 1).nonzero().squeeze()
    neg_idx = (edge_label == 0).nonzero().squeeze()
    
    batch_size = min(batch_size, len(pos_idx), len(neg_idx))
    if len(pos_idx) > batch_size:
        pos_idx = pos_idx[torch.randperm(len(pos_idx))[:batch_size]]
    if len(neg_idx) > batch_size:
        neg_idx = neg_idx[torch.randperm(len(neg_idx))[:batch_size]]
    
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    
    pos_pred = model.decode(z, edge_label_index[:, pos_idx])
    neg_pred = model.decode(z, edge_label_index[:, neg_idx])
    
    loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data, mask, device, max_samples=50000):
    model.eval()
    with torch.no_grad():
        idx = mask.nonzero().squeeze()
        if idx.numel() == 0:
            return 0.5
        
        edge_label_index = data.edge_label_index[:, idx]
        edge_label = data.edge_label[idx]
        
        subset_size = min(max_samples, len(idx))
        subset_idx = torch.randperm(len(idx))[:subset_size]
        
        z = model.encode(data.x, data.edge_index)
        pred = model.decode(z, edge_label_index[:, subset_idx])
        
        pos_mask = (edge_label[subset_idx] == 1)
        neg_mask = (edge_label[subset_idx] == 0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 0.5
        
        pos_pred = pred[pos_mask]
        neg_pred = pred[neg_mask]
        
        return calculate_auc_roc(pos_pred, neg_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to .pkl dataset file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=20000,
                       help='Batch size for training')
    args = parser.parse_args()
    
    print(f"\n STARTING GRAPHSAGE TRAINING")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}, LR: {args.lr}")
    
    # Data loading
    loader = SemanticGraphLoader(args.dataset)
    data = loader.to_pyg_data(feature_dim=64)
    
    # Data splitting
    data = split_data(data, train_ratio=0.8, val_ratio=0.1)
    
    print(f"\n DATA STATISTICS:")
    print(f"   Nodes: {data.num_nodes:,}")
    print(f"   Graph edges: {data.edge_index.shape[1]:,}")
    print(f"   Positive labels: {(data.edge_label == 1).sum().item():,}")
    print(f"   Negative labels: {(data.edge_label == 0).sum().item():,}")
    
    # Model
    model = GraphSAGELinkPred(
        in_channels=64,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim // 2,
        num_layers=2,
        dropout=0.2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    data = data.to(device)
    model = model.to(device)
    
    # Training
    print(f"\n STARTING TRAINING...")
    best_val_auc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer, device, args.batch_size)
        
        if epoch % 5 == 0 or epoch == args.epochs:
            val_auc = evaluate(model, data, data.val_mask, device)
            print(f"   Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'best_model.pth')
    
    # Testing
    print(f"\n TESTING...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_auc = evaluate(model, data, data.test_mask, device)
    print(f" TEST AUC: {test_auc:.4f}")
    
    print(f"\n COMPARISON:")
    print(f"   Your GraphSAGE: {test_auc:.4f}")
    print(f"   Baseline (M6): ~0.8201")
    print(f"   Best model (M1): ~0.8960")

if __name__ == '__main__':
    main()