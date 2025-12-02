"""
Federated Server Implementation

Implements the federated learning server with gradient similarity-based client selection
and meta-learning enhancement.
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .client import FederatedClient
from .aggregation import fedavg_aggregate, aggregate_models
from .gradient_similarity import (
    compute_gradient_similarity,
    select_clients_by_similarity,
    compute_similarity_matrix
)


class FederatedServer:
    """
    Federated learning server with gradient similarity-based selection.
    
    Coordinates federated learning across multiple clients using
    gradient similarity for intelligent client selection.
    
    Args:
        model: Global model architecture
        num_clients: Total number of clients
        selection_strategy: Client selection strategy ('random', 'gradient_similarity', 'diverse')
        client_fraction: Fraction of clients to select per round
        learning_rate: Server learning rate
        meta_learning_rate: Meta-learning rate for client adaptation
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 5,
        selection_strategy: str = 'gradient_similarity',
        client_fraction: float = 0.4,
        learning_rate: float = 1.0,
        meta_learning_rate: float = 0.01,
        device: str = 'cpu',
        **kwargs
    ):
        self.global_model = model.to(device)
        self.num_clients = num_clients
        self.selection_strategy = selection_strategy
        self.client_fraction = client_fraction
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.device = device
        
        # Additional configuration
        self.gradient_similarity_threshold = kwargs.get('gradient_similarity_threshold', 0.7)
        self.batch_size = kwargs.get('batch_size', 32)
        
        # Client management
        self.clients: List[FederatedClient] = []
        
        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'selected_clients': []
        }
        
        # Gradient storage for similarity computation
        self.client_gradients: List[Dict[str, torch.Tensor]] = []
    
    def register_clients(self, clients: List[FederatedClient]):
        """Register federated learning clients."""
        self.clients = clients
        self.num_clients = len(clients)
    
    def create_clients(
        self,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        client_learning_rate: float = 0.01
    ):
        """
        Create clients from data splits.
        
        Args:
            train_data: List of (features, labels) tuples for each client
            test_data: Optional list of test data for each client
            client_learning_rate: Learning rate for client training
        """
        self.clients = []
        
        for i, data in enumerate(train_data):
            test = test_data[i] if test_data else None
            client = FederatedClient(
                client_id=i,
                model=copy.deepcopy(self.global_model),
                train_data=data,
                test_data=test,
                learning_rate=client_learning_rate,
                meta_learning_rate=self.meta_learning_rate,
                device=self.device
            )
            self.clients.append(client)
        
        self.num_clients = len(self.clients)
    
    def select_clients(self, num_select: Optional[int] = None) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            num_select: Number of clients to select
            
        Returns:
            List of selected client indices
        """
        if num_select is None:
            num_select = max(1, int(self.client_fraction * self.num_clients))
        
        if self.selection_strategy == 'random':
            return list(np.random.choice(self.num_clients, num_select, replace=False))
        
        elif self.selection_strategy == 'gradient_similarity':
            if not self.client_gradients:
                # First round: random selection
                return list(np.random.choice(self.num_clients, num_select, replace=False))
            
            return select_clients_by_similarity(
                self.client_gradients,
                global_gradient=None,
                num_select=num_select,
                strategy='diverse',
                similarity_threshold=self.gradient_similarity_threshold
            )
        
        elif self.selection_strategy == 'diverse':
            if not self.client_gradients:
                return list(np.random.choice(self.num_clients, num_select, replace=False))
            
            return select_clients_by_similarity(
                self.client_gradients,
                num_select=num_select,
                strategy='diverse'
            )
        
        else:
            return list(np.random.choice(self.num_clients, num_select, replace=False))
    
    def broadcast_model(self, client_indices: Optional[List[int]] = None):
        """
        Broadcast global model to clients.
        
        Args:
            client_indices: Indices of clients to broadcast to (default: all)
        """
        global_params = self.global_model.state_dict()
        
        if client_indices is None:
            client_indices = range(self.num_clients)
        
        for idx in client_indices:
            self.clients[idx].set_model_parameters(copy.deepcopy(global_params))
    
    def aggregate_updates(
        self,
        client_indices: List[int],
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ):
        """
        Aggregate client updates into global model.
        
        Args:
            client_indices: Indices of participating clients
            client_updates: Updates from each client
            client_weights: Optional weights for aggregation
        """
        if client_weights is None:
            # Weight by data size
            client_weights = [self.clients[idx].data_size for idx in client_indices]
        
        self.global_model = aggregate_models(
            self.global_model,
            client_updates,
            client_weights,
            self.learning_rate
        )
    
    def train_round(
        self,
        num_local_epochs: int = 5,
        use_meta_learning: bool = True
    ) -> Dict[str, float]:
        """
        Execute one round of federated training.
        
        Args:
            num_local_epochs: Number of local training epochs
            use_meta_learning: Whether to use meta-learning
            
        Returns:
            Dictionary of metrics for this round
        """
        # Select clients
        selected_indices = self.select_clients()
        
        # Broadcast global model to selected clients
        self.broadcast_model(selected_indices)
        
        # Collect updates from selected clients
        client_updates = []
        client_gradients = []
        
        for idx in selected_indices:
            # Local training
            update = self.clients[idx].local_train(
                num_epochs=num_local_epochs,
                batch_size=self.batch_size,
                use_meta_learning=use_meta_learning
            )
            client_updates.append(update)
            
            # Compute gradients for similarity (for next round selection)
            gradients = self.clients[idx].compute_gradients()
            client_gradients.append(gradients)
        
        # Update stored gradients
        new_client_gradients = [None] * self.num_clients
        for i in range(self.num_clients):
            if i in selected_indices:
                idx = selected_indices.index(i)
                new_client_gradients[i] = client_gradients[idx]
            else:
                # Broadcast model to get gradient
                self.clients[i].set_model_parameters(
                    copy.deepcopy(self.global_model.state_dict())
                )
                new_client_gradients[i] = self.clients[i].compute_gradients()
        self.client_gradients = new_client_gradients
        
        # Aggregate updates
        self.aggregate_updates(selected_indices, client_updates)
        
        # Compute metrics
        metrics = self._compute_round_metrics(selected_indices)
        metrics['selected_clients'] = selected_indices
        
        return metrics
    
    def train(
        self,
        num_rounds: int = 100,
        local_epochs: int = 5,
        use_meta_learning: bool = True,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Run federated training for specified number of rounds.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local epochs per round
            use_meta_learning: Whether to use meta-learning
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        iterator = tqdm(range(num_rounds)) if verbose else range(num_rounds)
        
        for round_num in iterator:
            metrics = self.train_round(local_epochs, use_meta_learning)
            
            # Update history
            self.history['round'].append(round_num)
            self.history['train_loss'].append(metrics.get('train_loss', 0))
            self.history['train_accuracy'].append(metrics.get('train_accuracy', 0))
            self.history['test_loss'].append(metrics.get('test_loss', 0))
            self.history['test_accuracy'].append(metrics.get('test_accuracy', 0))
            self.history['selected_clients'].append(metrics['selected_clients'])
            
            if verbose:
                iterator.set_description(
                    f"Round {round_num}: Loss={metrics.get('test_loss', 0):.4f}, "
                    f"Acc={metrics.get('test_accuracy', 0):.4f}"
                )
        
        return self.history
    
    def _compute_round_metrics(self, selected_indices: List[int]) -> Dict[str, float]:
        """Compute metrics for the current round."""
        # Evaluate on all clients
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        # Broadcast updated model to all clients for evaluation
        self.broadcast_model()
        
        for idx in range(self.num_clients):
            client_metrics = self.clients[idx].evaluate()
            weight = self.clients[idx].data_size
            total_loss += client_metrics['loss'] * weight
            total_acc += client_metrics['accuracy'] * weight
            total_samples += weight
        
        return {
            'train_loss': total_loss / total_samples if total_samples > 0 else 0,
            'train_accuracy': total_acc / total_samples if total_samples > 0 else 0,
            'test_loss': total_loss / total_samples if total_samples > 0 else 0,
            'test_accuracy': total_acc / total_samples if total_samples > 0 else 0
        }
    
    def evaluate_global(self, test_loader) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.global_model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        }
    
    def personalize_clients(self, num_epochs: int = 3):
        """
        Personalize models for each client through fine-tuning.
        
        Args:
            num_epochs: Number of fine-tuning epochs
        """
        self.broadcast_model()
        
        for client in self.clients:
            client.fine_tune(num_epochs)
    
    def save_model(self, path: str):
        """Save global model to file."""
        torch.save(self.global_model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load global model from file."""
        self.global_model.load_state_dict(torch.load(path))
    
    def get_client_similarity_matrix(self) -> np.ndarray:
        """
        Get pairwise similarity matrix between clients.
        
        Returns:
            Similarity matrix of shape (num_clients, num_clients)
        """
        if not self.client_gradients:
            return np.eye(self.num_clients)
        
        return compute_similarity_matrix(self.client_gradients)
