"""
Functions for calculating prediction quality metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_recall_curve,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score
)
from typing import Dict, Tuple, Optional, Union

def calculate_auc_roc(pos_scores: torch.Tensor, 
                     neg_scores: torch.Tensor) -> float:
    """
    Calculates AUC-ROC from predictions for positive and negative examples.
    
    Args:
        pos_scores: Predictions for positive examples
        neg_scores: Predictions for negative examples
        
    Returns:
        AUC-ROC value
    """
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([
        torch.ones_like(pos_scores), 
        torch.zeros_like(neg_scores)
    ]).cpu().numpy()
    
    if len(np.unique(labels)) < 2:
        return 0.5  # If only one class exists, return random value
    
    return roc_auc_score(labels, scores)

def calculate_metrics(pos_scores: torch.Tensor, 
                     neg_scores: torch.Tensor, 
                     threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Calculates multiple quality metrics: AUC-ROC, Average Precision, F1, Precision, Recall.
    
    Args:
        pos_scores: Predictions for positive examples
        neg_scores: Predictions for negative examples
        threshold: Threshold for binarization (if None, selected automatically)
        
    Returns:
        Dictionary with metrics
    """
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([
        torch.ones_like(pos_scores), 
        torch.zeros_like(neg_scores)
    ]).cpu().numpy()
    
    if len(np.unique(labels)) < 2:
        return {
            'auc_roc': 0.5,
            'average_precision': 0.5,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'threshold': 0.5 if threshold is None else threshold
        }
    
    # AUC-ROC
    auc_roc = roc_auc_score(labels, scores)
    
    # Average Precision
    ap = average_precision_score(labels, scores)
    
    # Optimal threshold by F1-score
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        if len(thresholds) > 0:
            threshold = thresholds[np.argmax(f1_scores[:-1])]
        else:
            threshold = 0.5
    
    # Binarized predictions and metrics
    binary_preds = (scores >= threshold).astype(int)
    
    f1 = f1_score(labels, binary_preds)
    precision = sk_precision_score(labels, binary_preds, zero_division=0)
    recall = sk_recall_score(labels, binary_preds, zero_division=0)
    
    return {
        'auc_roc': auc_roc,
        'average_precision': ap,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold
    }

def precision_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculates precision.
    
    Args:
        labels: True labels
        preds: Predicted labels
        
    Returns:
        Precision value
    """
    return sk_precision_score(labels, preds, zero_division=0)

def recall_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculates recall.
    
    Args:
        labels: True labels
        preds: Predicted labels
        
    Returns:
        Recall value
    """
    return sk_recall_score(labels, preds, zero_division=0)

def compute_all_metrics(predictions: torch.Tensor,
                       targets: torch.Tensor,
                       prefix: str = "") -> Dict[str, float]:
    """
    Calculates all main metrics for classification/regression tasks.
    
    Args:
        predictions: Model predictions
        targets: Target values
        prefix: Prefix for metric names
        
    Returns:
        Dictionary with metrics
    """
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    metrics = {}
    
    # For binary classification
    if len(preds_np.shape) == 1 or preds_np.shape[1] == 1:
        if len(np.unique(targets_np)) <= 2:  # Binary classification
            preds_binary = (preds_np > 0.5).astype(int)
            targets_binary = (targets_np > 0.5).astype(int)
            
            metrics.update({
                f"{prefix}accuracy": np.mean(preds_binary == targets_binary),
                f"{prefix}precision": precision_score(targets_binary, preds_binary),
                f"{prefix}recall": recall_score(targets_binary, preds_binary),
                f"{prefix}f1": f1_score(targets_binary, preds_binary)
            })
    
    # MSE for regression
    metrics[f"{prefix}mse"] = np.mean((preds_np - targets_np) ** 2)
    metrics[f"{prefix}mae"] = np.mean(np.abs(preds_np - targets_np))
    
    return metrics