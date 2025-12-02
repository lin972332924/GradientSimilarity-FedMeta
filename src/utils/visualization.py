"""
Visualization Module

Implements visualization utilities for federated learning results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List],
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None,
    title: str = 'Training History'
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary containing training metrics
        metrics: List of metrics to plot
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        test_key = f'test_{metric}'
        
        if train_key in history:
            ax.plot(history[train_key], label='Train', linewidth=2)
        if test_key in history:
            ax.plot(history[test_key], label='Test', linewidth=2)
        elif metric in history:
            ax.plot(history[metric], label=metric.capitalize(), linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Rounds')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Optional class names
        save_path: Optional path to save figure
        title: Plot title
        normalize: Whether to normalize values
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_client_similarity(
    similarity_matrix: np.ndarray,
    client_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Client Gradient Similarity'
) -> plt.Figure:
    """
    Plot client gradient similarity heatmap.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        client_names: Optional client names
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    num_clients = len(similarity_matrix)
    
    if client_names is None:
        client_names = [f'Client {i}' for i in range(num_clients)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=client_names,
        yticklabels=client_names,
        ax=ax
    )
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_client_selection_frequency(
    selection_history: List[List[int]],
    num_clients: int,
    save_path: Optional[str] = None,
    title: str = 'Client Selection Frequency'
) -> plt.Figure:
    """
    Plot frequency of client selection across rounds.
    
    Args:
        selection_history: List of selected client indices per round
        num_clients: Total number of clients
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    selection_counts = np.zeros(num_clients)
    
    for selected in selection_history:
        for client_id in selected:
            selection_counts[client_id] += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    client_ids = range(num_clients)
    bars = ax.bar(client_ids, selection_counts, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Selection Count')
    ax.set_title(title)
    ax.set_xticks(client_ids)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, selection_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            int(count),
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = 'Method Comparison'
) -> plt.Figure:
    """
    Plot comparison of metrics across different methods.
    
    Args:
        metrics_dict: Dictionary mapping method names to their metrics
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    methods = list(metrics_dict.keys())
    metrics = list(metrics_dict[methods[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method)
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_comparison(
    histories: Dict[str, Dict[str, List]],
    metric: str = 'test_accuracy',
    save_path: Optional[str] = None,
    title: str = 'Convergence Comparison'
) -> plt.Figure:
    """
    Plot convergence curves for multiple methods.
    
    Args:
        histories: Dictionary mapping method names to their training histories
        metric: Metric to plot
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, history in histories.items():
        if metric in history:
            ax.plot(history[metric], label=method_name, linewidth=2)
    
    ax.set_xlabel('Round')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
