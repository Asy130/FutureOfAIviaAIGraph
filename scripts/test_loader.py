#!/usr/bin/env python
# scripts/test_loader.py
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"Python path: {sys.path[0]}")

try:
    from utils.data_loader import SemanticGraphLoader
    import torch
    print(" Imports successful!")
except ImportError as e:
    print(f" Import error: {e}")
    print("Check file existence:")
    print(f"  utils/__init__.py: {os.path.exists(os.path.join(parent_dir, 'utils', '__init__.py'))}")
    print(f"  utils/data_loader.py: {os.path.exists(os.path.join(parent_dir, 'utils', 'data_loader.py'))}")
    sys.exit(1)

print("🧪 Testing updated data loader...")

# Use the smallest file
loader = SemanticGraphLoader("data/SemanticGraph_delta_5_cutoff_0_minedge_1.pkl")

print("\n📊 Dataset information:")
info = {
    'train_edges': len(loader.train_edges),
    'candidate_pairs': len(loader.candidate_pairs),
    'year_start': loader.year_start,
    'delta': loader.delta,
    'cutoff': loader.cutoff,
    'min_edges': loader.min_edges
}

for key, value in info.items():
    print(f"  {key}: {value}")

# Convert to PyG format
print("\n Converting to PyG format...")
try:
    data = loader.to_pyg_data(feature_dim=128)
    
    print(f"\n Data prepared:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Graph edges: {data.edge_index.shape[1]}")
    print(f"  Candidate pairs: {data.edge_label_index.shape[1]}")
    
    if hasattr(data, 'edge_label'):
        print(f"  Positive labels: {(data.edge_label == 1).sum().item()}")
        print(f"  Negative labels: {(data.edge_label == 0).sum().item()}")
    
    print("\n🎉 Loader works correctly! Ready to start training.")
    
except Exception as e:
    print(f" Conversion error: {e}")
    import traceback
    traceback.print_exc()