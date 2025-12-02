"""
Model Aggregation Strategies

Implements various model aggregation methods for federated learning.
"""

import torch
import copy
from typing import List, Dict, Optional, Tuple


def fedavg_aggregate(
    global_model: torch.nn.Module,
    client_models: List[torch.nn.Module],
    client_weights: Optional[List[float]] = None
) -> torch.nn.Module:
    """
    Federated Averaging (FedAvg) aggregation.
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: Optional weights for each client (default: equal weights)
        
    Returns:
        Updated global model
    """
    if not client_models:
        return global_model
    
    num_clients = len(client_models)
    
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]
    
    # Get state dicts
    global_state = global_model.state_dict()
    client_states = [model.state_dict() for model in client_models]
    
    # Aggregate parameters
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for i, client_state in enumerate(client_states):
            global_state[key] += client_weights[i] * client_state[key].float()
    
    global_model.load_state_dict(global_state)
    return global_model


def aggregate_models(
    global_model: torch.nn.Module,
    client_updates: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
    learning_rate: float = 1.0
) -> torch.nn.Module:
    """
    Aggregate client updates into global model.
    
    Args:
        global_model: Global model to update
        client_updates: List of parameter updates from clients
        client_weights: Optional weights for each client
        learning_rate: Server learning rate for update
        
    Returns:
        Updated global model
    """
    if not client_updates:
        return global_model
    
    num_clients = len(client_updates)
    
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]
    
    # Compute weighted average of updates
    avg_update = {}
    for key in client_updates[0].keys():
        avg_update[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
        for i, update in enumerate(client_updates):
            avg_update[key] += client_weights[i] * update[key].float()
    
    # Apply update to global model
    global_state = global_model.state_dict()
    for key in global_state.keys():
        if key in avg_update:
            global_state[key] = (global_state[key].float() + learning_rate * avg_update[key]).to(global_state[key].dtype)
    
    global_model.load_state_dict(global_state)
    return global_model


def fedprox_aggregate(
    global_model: torch.nn.Module,
    client_models: List[torch.nn.Module],
    client_weights: Optional[List[float]] = None,
    mu: float = 0.01
) -> torch.nn.Module:
    """
    FedProx aggregation with proximal term consideration.
    
    Note: The proximal term is applied during client training, not aggregation.
    This function performs standard FedAvg aggregation.
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: Optional weights for each client
        mu: Proximal term coefficient (for reference)
        
    Returns:
        Updated global model
    """
    return fedavg_aggregate(global_model, client_models, client_weights)


def weighted_aggregate_by_data_size(
    global_model: torch.nn.Module,
    client_models: List[torch.nn.Module],
    data_sizes: List[int]
) -> torch.nn.Module:
    """
    Aggregate models weighted by client data sizes.
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        data_sizes: List of data sizes for each client
        
    Returns:
        Updated global model
    """
    total_data = sum(data_sizes)
    weights = [size / total_data for size in data_sizes]
    return fedavg_aggregate(global_model, client_models, weights)


def compute_model_divergence(
    model1: torch.nn.Module,
    model2: torch.nn.Module
) -> float:
    """
    Compute parameter divergence between two models.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        L2 norm of parameter difference
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    total_diff = 0.0
    for key in state1.keys():
        diff = state1[key].float() - state2[key].float()
        total_diff += torch.sum(diff ** 2).item()
    
    return total_diff ** 0.5


def momentum_aggregate(
    global_model: torch.nn.Module,
    client_models: List[torch.nn.Module],
    momentum_buffer: Optional[Dict[str, torch.Tensor]] = None,
    momentum: float = 0.9,
    client_weights: Optional[List[float]] = None
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """
    Aggregation with momentum for faster convergence.
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        momentum_buffer: Previous momentum buffer
        momentum: Momentum coefficient
        client_weights: Optional weights for each client
        
    Returns:
        Tuple of (updated model, updated momentum buffer)
    """
    if not client_models:
        return global_model, momentum_buffer or {}
    
    num_clients = len(client_models)
    
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]
    
    global_state = global_model.state_dict()
    client_states = [model.state_dict() for model in client_models]
    
    # Initialize momentum buffer if needed
    if momentum_buffer is None:
        momentum_buffer = {key: torch.zeros_like(global_state[key]) for key in global_state.keys()}
    
    # Compute weighted average of client parameters
    for key in global_state.keys():
        avg_param = torch.zeros_like(global_state[key], dtype=torch.float32)
        for i, client_state in enumerate(client_states):
            avg_param += client_weights[i] * client_state[key].float()
        
        # Compute update
        update = avg_param - global_state[key].float()
        
        # Apply momentum
        momentum_buffer[key] = momentum * momentum_buffer[key] + update
        global_state[key] = global_state[key].float() + momentum_buffer[key]
    
    global_model.load_state_dict(global_state)
    return global_model, momentum_buffer
