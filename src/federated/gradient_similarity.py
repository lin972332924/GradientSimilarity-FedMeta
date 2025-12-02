"""
Gradient Similarity Computation Module

Implements gradient similarity-based client selection for federated learning.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def flatten_gradients(gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten gradient dictionary to a single vector.
    
    Args:
        gradients: Dictionary mapping parameter names to gradient tensors
        
    Returns:
        Flattened gradient vector
    """
    flat_grads = []
    for name in sorted(gradients.keys()):
        flat_grads.append(gradients[name].view(-1))
    return torch.cat(flat_grads)


def compute_gradient_similarity(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute cosine similarity between two gradient dictionaries.
    
    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary
        
    Returns:
        Cosine similarity value in range [-1, 1]
    """
    flat1 = flatten_gradients(grad1)
    flat2 = flatten_gradients(grad2)
    
    # Compute cosine similarity
    dot_product = torch.dot(flat1, flat2)
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()


def compute_similarity_matrix(
    client_gradients: List[Dict[str, torch.Tensor]]
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for all client gradients.
    
    Args:
        client_gradients: List of gradient dictionaries from each client
        
    Returns:
        Similarity matrix of shape (num_clients, num_clients)
    """
    num_clients = len(client_gradients)
    similarity_matrix = np.zeros((num_clients, num_clients))
    
    for i in range(num_clients):
        for j in range(i, num_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = compute_gradient_similarity(
                    client_gradients[i],
                    client_gradients[j]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    return similarity_matrix


def select_clients_by_similarity(
    client_gradients: List[Dict[str, torch.Tensor]],
    global_gradient: Optional[Dict[str, torch.Tensor]] = None,
    num_select: int = 5,
    strategy: str = 'diverse',
    similarity_threshold: float = 0.7
) -> List[int]:
    """
    Select clients based on gradient similarity.
    
    Args:
        client_gradients: List of gradient dictionaries from each client
        global_gradient: Optional global gradient for reference
        num_select: Number of clients to select
        strategy: Selection strategy ('diverse', 'similar', 'threshold')
        similarity_threshold: Threshold for 'threshold' strategy
        
    Returns:
        List of selected client indices
    """
    num_clients = len(client_gradients)
    
    if num_select >= num_clients:
        return list(range(num_clients))
    
    if strategy == 'diverse':
        return _select_diverse_clients(client_gradients, num_select)
    elif strategy == 'similar':
        return _select_similar_clients(client_gradients, global_gradient, num_select)
    elif strategy == 'threshold':
        return _select_by_threshold(
            client_gradients, global_gradient, num_select, similarity_threshold
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _select_diverse_clients(
    client_gradients: List[Dict[str, torch.Tensor]],
    num_select: int
) -> List[int]:
    """
    Select diverse clients by maximizing gradient diversity.
    
    Uses a greedy algorithm to select clients with maximum diversity.
    """
    num_clients = len(client_gradients)
    similarity_matrix = compute_similarity_matrix(client_gradients)
    
    # Start with random client
    selected = [np.random.randint(num_clients)]
    
    while len(selected) < num_select:
        min_max_similarity = float('inf')
        best_candidate = -1
        
        for i in range(num_clients):
            if i in selected:
                continue
            
            # Maximum similarity to already selected clients
            max_sim = max(similarity_matrix[i, j] for j in selected)
            
            # Select client with minimum maximum similarity (most diverse)
            if max_sim < min_max_similarity:
                min_max_similarity = max_sim
                best_candidate = i
        
        if best_candidate >= 0:
            selected.append(best_candidate)
    
    return selected


def _select_similar_clients(
    client_gradients: List[Dict[str, torch.Tensor]],
    global_gradient: Optional[Dict[str, torch.Tensor]],
    num_select: int
) -> List[int]:
    """
    Select clients most similar to global gradient.
    """
    if global_gradient is None:
        # If no global gradient, use average
        global_gradient = _compute_average_gradient(client_gradients)
    
    similarities = []
    for i, grad in enumerate(client_gradients):
        sim = compute_gradient_similarity(grad, global_gradient)
        similarities.append((i, sim))
    
    # Sort by similarity (descending) and select top
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in similarities[:num_select]]


def _select_by_threshold(
    client_gradients: List[Dict[str, torch.Tensor]],
    global_gradient: Optional[Dict[str, torch.Tensor]],
    num_select: int,
    threshold: float
) -> List[int]:
    """
    Select clients with similarity above threshold.
    """
    if global_gradient is None:
        global_gradient = _compute_average_gradient(client_gradients)
    
    above_threshold = []
    for i, grad in enumerate(client_gradients):
        sim = compute_gradient_similarity(grad, global_gradient)
        if sim >= threshold:
            above_threshold.append((i, sim))
    
    # If not enough clients above threshold, select top by similarity
    if len(above_threshold) < num_select:
        return _select_similar_clients(
            client_gradients, global_gradient, num_select
        )
    
    # Select randomly from clients above threshold
    np.random.shuffle(above_threshold)
    return [idx for idx, _ in above_threshold[:num_select]]


def _compute_average_gradient(
    client_gradients: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Compute average gradient across all clients.
    """
    if not client_gradients:
        return {}
    
    avg_gradient = {}
    num_clients = len(client_gradients)
    
    for name in client_gradients[0].keys():
        stacked = torch.stack([g[name] for g in client_gradients])
        avg_gradient[name] = torch.mean(stacked, dim=0)
    
    return avg_gradient


def compute_gradient_norm(gradients: Dict[str, torch.Tensor]) -> float:
    """
    Compute the L2 norm of gradients.
    
    Args:
        gradients: Dictionary mapping parameter names to gradient tensors
        
    Returns:
        L2 norm of the gradient
    """
    flat = flatten_gradients(gradients)
    return torch.norm(flat).item()


def clip_gradients(
    gradients: Dict[str, torch.Tensor],
    max_norm: float
) -> Dict[str, torch.Tensor]:
    """
    Clip gradients to maximum norm.
    
    Args:
        gradients: Dictionary mapping parameter names to gradient tensors
        max_norm: Maximum allowed norm
        
    Returns:
        Clipped gradients
    """
    current_norm = compute_gradient_norm(gradients)
    
    if current_norm > max_norm:
        scale = max_norm / current_norm
        return {name: grad * scale for name, grad in gradients.items()}
    
    return gradients
