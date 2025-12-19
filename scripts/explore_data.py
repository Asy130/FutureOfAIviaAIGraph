#!/usr/bin/env python
# scripts/explore_data.py
import pickle
import sys
sys.path.append('..')

def explore_pkl_file(filepath):
    """Explores the structure of a .pkl file."""
    print(f"\n🔍 Exploring file: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        for i, item in enumerate(data[:3]):  # First 3 elements
            print(f"  Element {i}: type={type(item)}, shape/length={get_shape(item)}")
        
        # If this is data from the original repository
        if len(data) == 7:
            print("\nStructure (based on original repo code):")
            names = [
                "train_dynamic_graph_sparse",
                "train_edges_for_checking", 
                "train_edges_solution",
                "year_start",
                "current_delta", 
                "curr_vertex_degree_cutoff",
                "current_min_edges"
            ]
            for i, name in enumerate(names):
                print(f"  {name}: {type(data[i])}, value={data[i] if i>2 else '...'}")
    
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}")
    
    return data

def get_shape(item):
    """Gets the shape of data."""
    try:
        import numpy as np
        if isinstance(item, np.ndarray):
            return f"np.array {item.shape}"
        elif isinstance(item, list):
            return f"list[{len(item)}]"
        else:
            return str(item)[:100]
    except:
        return str(type(item))

if __name__ == "__main__":
    # Test on the smallest file
    explore_pkl_file("data/SemanticGraph_delta_5_cutoff_0_minedge_1.pkl")