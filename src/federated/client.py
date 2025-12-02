"""
Federated Client Implementation

Implements the federated learning client with meta-learning support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import Dict, Optional, Tuple, List


class FederatedClient:
    """
    Federated learning client with meta-learning capabilities.
    
    Supports local training with optional meta-learning enhancement
    for personalized federated learning.
    
    Args:
        client_id: Unique identifier for this client
        model: Neural network model
        train_data: Training data tuple (features, labels)
        test_data: Optional test data tuple
        learning_rate: Learning rate for local optimization
        meta_learning_rate: Learning rate for meta-learning adaptation
        device: Device to use for training ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        learning_rate: float = 0.01,
        meta_learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        
        # Create data loaders
        self.train_loader = self._create_data_loader(train_data)
        self.test_loader = self._create_data_loader(test_data) if test_data else None
        
        self.data_size = len(train_data[0])
        
        # Store personalized model parameters
        self.personalized_params = None
        
        # Meta-learning state
        self.meta_gradients = None
    
    def _create_data_loader(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int = 32
    ) -> DataLoader:
        """Create a DataLoader from data tuple."""
        dataset = TensorDataset(
            data[0].to(self.device),
            data[1].to(self.device)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update local model with given parameters."""
        self.model.load_state_dict(parameters)
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return copy.deepcopy(self.model.state_dict())
    
    def local_train(
        self,
        num_epochs: int = 5,
        batch_size: int = 32,
        use_meta_learning: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local training on client data.
        
        Args:
            num_epochs: Number of local training epochs
            batch_size: Batch size for training
            use_meta_learning: Whether to use meta-learning adaptation
            
        Returns:
            Gradients/updates from local training
        """
        self.model.train()
        
        if use_meta_learning:
            return self._meta_train(num_epochs, batch_size)
        else:
            return self._standard_train(num_epochs, batch_size)
    
    def _standard_train(
        self,
        num_epochs: int,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Standard local training."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        initial_params = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Compute parameter updates
        final_params = self.model.state_dict()
        updates = {}
        for key in initial_params.keys():
            updates[key] = final_params[key] - initial_params[key]
        
        return updates
    
    def _meta_train(
        self,
        num_epochs: int,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Meta-learning enhanced training.
        
        Implements MAML-inspired adaptation for personalized federated learning.
        """
        criterion = nn.CrossEntropyLoss()
        
        # Save initial parameters
        initial_params = copy.deepcopy(self.model.state_dict())
        
        # Inner loop: Task-specific adaptation
        fast_weights = copy.deepcopy(initial_params)
        
        for epoch in range(num_epochs):
            # Create optimizer with current fast weights
            self.model.load_state_dict(fast_weights)
            
            for batch_x, batch_y in self.train_loader:
                # Compute gradients
                self.model.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Update fast weights
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            fast_weights[name] = fast_weights[name] - self.meta_learning_rate * param.grad
        
        # Store meta gradients for similarity computation
        self.meta_gradients = {}
        for key in initial_params.keys():
            self.meta_gradients[key] = initial_params[key] - fast_weights[key]
        
        # Load fast weights as new model parameters
        self.model.load_state_dict(fast_weights)
        
        return self.meta_gradients
    
    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Compute gradients on local data without updating model.
        
        Returns:
            Dictionary of gradients for each parameter
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Zero out existing gradients
        self.model.zero_grad()
        
        # Compute gradients over entire local dataset
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in self.train_loader:
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone() / num_batches
        
        return gradients
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on local test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.test_loader is None:
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'client_id': self.client_id
        }
    
    def save_personalized_model(self):
        """Save current model as personalized model."""
        self.personalized_params = copy.deepcopy(self.model.state_dict())
    
    def load_personalized_model(self):
        """Load personalized model parameters."""
        if self.personalized_params is not None:
            self.model.load_state_dict(self.personalized_params)
    
    def fine_tune(
        self,
        num_epochs: int = 3,
        learning_rate: Optional[float] = None
    ):
        """
        Fine-tune model on local data for personalization.
        
        Args:
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning (default: half of training lr)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate / 2
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.save_personalized_model()
