#!/usr/bin/env python
"""
Script for evaluating GraphSAGE model on all datasets (analog of original evaluate_model.py).
"""

import sys
import os

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import glob
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from models.graphsage_linkpred import GraphSAGELinkPred
from utils.data_loader import SemanticGraphLoader
from utils.metrics import calculate_auc_roc

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_dataset_name(filename):
    """
    Parse dataset parameters from filename.
    
    Format: SemanticGraph_delta_N_cutoff_M_minedge_P.pkl
    """
    try:
        parts = filename.split('_')
        delta = int(parts[2])  # N
        cutoff = int(parts[4])  # M
        minedge = int(parts[6].split('.')[0])  # P
        return delta, cutoff, minedge
    except (IndexError, ValueError) as e:
        logger.warning(f"Failed to parse filename {filename}: {e}")
        return None, None, None

def load_candidate_pairs(data_loader, year_from, year_to):
    """
    Load candidate pairs for prediction.
    
    In real usage, need to load node pairs that
    are not connected in year_from but may connect in year_to.
    """
    # TODO: Implement loading real candidate pairs
    # Currently return random pairs for demonstration
    
    data_from = data_loader.to_pyg_data(year_from)
    if data_from is None:
        return None, None
    
    num_nodes = data_from.num_nodes
    
    # Random pairs for demonstration
    num_candidates = min(10000, num_nodes * 10)
    node_pairs = torch.randint(0, num_nodes, (2, num_candidates))
    
    # Remove pairs that are already connected
    edge_set = set([(u.item(), v.item()) for u, v in data_from.edge_index.t()])
    valid_pairs = []
    
    for i in range(num_candidates):
        u, v = node_pairs[0, i].item(), node_pairs[1, i].item()
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            valid_pairs.append([u, v])
    
    if len(valid_pairs) == 0:
        return None, None
    
    node_pairs = torch.tensor(valid_pairs).t()
    
    # TODO: Load real labels (whether connection appeared in year_to)
    # Currently use random labels for demonstration
    labels = torch.randint(0, 2, (node_pairs.shape[1],)).float()
    
    return node_pairs, labels

def evaluate_model_on_dataset(model, data_loader, year_from, year_to, device='cpu'):
    """
    Evaluate model on a single dataset.
    
    Args:
        model: Trained GraphSAGE model
        data_loader: Data loader
        year_from: Source year
        year_to: Target year
        device: Computation device
        
    Returns:
        AUC-ROC score or None on error
    """
    try:
        # Load graph for year_from
        data_from = data_loader.to_pyg_data(year_from)
        if data_from is None:
            logger.error(f"No data for year {year_from}")
            return None
        
        # Load candidate pairs and labels
        node_pairs, labels = load_candidate_pairs(data_loader, year_from, year_to)
        if node_pairs is None:
            logger.error(f"Failed to load candidate pairs for {year_from}->{year_to}")
            return None
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            data_from = data_from.to(device)
            predictions = model.predict_all_pairs(
                data_from.x,
                data_from.edge_index,
                node_pairs.to(device)
            )
        
        # Separate into positive and negative examples
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            logger.warning(f"Insufficient examples for AUC evaluation")
            return 0.5
        
        pos_scores = predictions[pos_indices]
        neg_scores = predictions[neg_indices]
        
        # Calculate AUC-ROC
        auc = calculate_auc_roc(pos_scores, neg_scores)
        return auc
        
    except Exception as e:
        logger.error(f"Error evaluating dataset: {e}")
        return None

def main():
    """Main evaluation function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    in_channels = 128  # Should match feature dimension in data
    hidden_channels = 256
    out_channels = 128
    num_layers = 2
    dropout = 0.0
    
    model = GraphSAGELinkPred(
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers, 
        dropout
    )
    
    # Load model weights (assuming model is already trained)
    checkpoint_path = os.path.join(parent_dir, 'checkpoints', 'graphsage_exp', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.warning(f"Model file {checkpoint_path} not found. Using randomly initialized model.")
    
    model = model.to(device)
    
    # Find all .pkl files in data/
    data_dir = os.path.join(parent_dir, 'data')
    dataset_files = glob.glob(os.path.join(data_dir, 'SemanticGraph_*.pkl'))
    
    if not dataset_files:
        # Try alternative paths
        alternative_paths = [
            os.path.join(parent_dir, 'data'),
            os.path.join(current_dir, '../data'),
            os.path.join(current_dir, '../../data'),
            'data',
            '../data'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                dataset_files = glob.glob(os.path.join(path, 'SemanticGraph_*.pkl'))
                if dataset_files:
                    data_dir = path
                    break
        
        if not dataset_files:
            logger.error("No datasets found in any expected location!")
            logger.info("Tried looking in:")
            for path in alternative_paths:
                logger.info(f"  - {path}")
            logger.info("\nPlease download datasets from Zenodo and place in the 'data' folder.")
            logger.info("The 'data' folder should be in the project root directory.")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Script location: {current_dir}")
            return
    
    logger.info(f"Found {len(dataset_files)} datasets in {data_dir}")
    
    # Evaluate on each dataset
    results = []
    
    for filepath in tqdm(dataset_files, desc="Evaluating datasets"):
        filename = os.path.basename(filepath)
        logger.info(f"Evaluating on {filename}")
        
        # Parse parameters from filename
        delta, cutoff, minedge = parse_dataset_name(filename)
        if delta is None:
            continue
        
        # Load dataset
        try:
            loader = SemanticGraphLoader(filepath)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
        
        # Years: predict from 2021-delta to 2021
        year_from = 2021 - delta
        year_to = 2021
        
        # Evaluate model
        auc = evaluate_model_on_dataset(model, loader, year_from, year_to, device)
        
        if auc is not None:
            results.append({
                'delta': delta,
                'cutoff': cutoff,
                'minedge': minedge,
                'auc': auc,
                'filename': filename
            })
            logger.info(f"  AUC: {auc:.4f}")
        else:
            logger.warning(f"  Failed to calculate AUC")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        
        # Create results directory
        results_dir = os.path.join(parent_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to CSV
        csv_path = os.path.join(results_dir, 'graphsage_results.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        # Save to pickle (for compatibility with original format)
        pkl_path = os.path.join(results_dir, 'graphsage_results.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(df.to_dict('records'), f)
        
        # Print results table
        print("\n" + "="*60)
        print("GraphSAGE Model Results:")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        # Print average AUC
        avg_auc = df['auc'].mean()
        print(f"\nAverage AUC: {avg_auc:.4f}")
        
        # Save summary statistics
        summary = {
            'average_auc': float(avg_auc),
            'num_datasets': len(results),
            'results': df.to_dict('records')
        }
        
        summary_path = os.path.join(results_dir, 'summary.yaml')
        import yaml
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
        
        logger.info(f"Summary statistics saved to {summary_path}")
    else:
        logger.error("Failed to get results for any dataset")

if __name__ == '__main__':
    main()