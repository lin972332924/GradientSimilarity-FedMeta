"""
Evaluation Metrics Module

Implements various evaluation metrics for liver cancer recurrence prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> float:
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average)


def compute_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    multi_class: str = 'ovr'
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        multi_class: Multi-class strategy ('ovr' or 'ovo')
        
    Returns:
        AUC score (returns 0.0 if computation fails due to single class in y_true
        or other issues)
    """
    try:
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
            # Binary classification
            if len(y_prob.shape) > 1:
                y_prob = y_prob[:, 1]
            return roc_auc_score(y_true, y_prob)
        else:
            # Multi-class
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
    except ValueError as e:
        # Common cases: only one class present in y_true, or y_prob has wrong shape
        import warnings
        warnings.warn(f"AUC computation failed: {str(e)}. Returning 0.0")
        return 0.0


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Tuple[float, float]:
    """
    Compute precision and recall scores.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method
        
    Returns:
        Tuple of (precision, recall)
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    return precision, recall


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute specificity (true negative rate).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def compute_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute sensitivity (true positive rate / recall).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Sensitivity score
    """
    return recall_score(y_true, y_pred, average='binary', zero_division=0)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device for computation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    precision, recall = compute_precision_recall(y_true, y_pred)
    
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'f1_score': compute_f1_score(y_true, y_pred),
        'auc': compute_auc(y_true, y_prob),
        'precision': precision,
        'recall': recall,
        'specificity': compute_specificity(y_true, y_pred),
        'sensitivity': compute_sensitivity(y_true, y_pred)
    }
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Optional class names
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names)


class MetricsTracker:
    """
    Track and store evaluation metrics over training.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            'accuracy': [],
            'f1_score': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'loss': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """Add metrics from current epoch/round."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best(self, metric: str = 'accuracy') -> Tuple[int, float]:
        """Get best value and index for specified metric."""
        values = self.history.get(metric, [])
        if not values:
            return -1, 0.0
        best_idx = np.argmax(values)
        return best_idx, values[best_idx]
    
    def get_latest(self) -> Dict[str, float]:
        """Get most recent metrics."""
        return {key: values[-1] if values else 0.0 
                for key, values in self.history.items()}
    
    def reset(self):
        """Clear all tracked metrics."""
        for key in self.history:
            self.history[key] = []
