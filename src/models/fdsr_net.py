"""
FDSR-Net: Feature-wise Dense Sparse Residual Network

A lightweight neural network architecture optimized for medical tabular data
in federated learning settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DenseBlock(nn.Module):
    """Dense block with feature-wise connections."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class SparseResidualConnection(nn.Module):
    """Sparse residual connection for efficient gradient flow."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(SparseResidualConnection, self).__init__()
        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            residual = self.projection(residual)
        return x + residual


class FDSRNet(nn.Module):
    """
    Feature-wise Dense Sparse Residual Network (FDSR-Net).
    
    A lightweight neural network designed for liver cancer recurrence prediction
    in federated learning settings.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes (default: 2 for binary classification)
        dropout_rate: Dropout probability for regularization
        use_residual: Whether to use sparse residual connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        use_residual: bool = True
    ):
        super(FDSRNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # Input layer
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.dense_blocks.append(
                DenseBlock(prev_dim, hidden_dim, dropout_rate)
            )
            if use_residual:
                self.residual_connections.append(
                    SparseResidualConnection(prev_dim, hidden_dim)
                )
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # Input normalization
        x = self.input_bn(x)
        
        # Pass through dense blocks with optional residual connections
        for i, dense_block in enumerate(self.dense_blocks):
            residual = x
            x = dense_block(x)
            if self.use_residual:
                x = self.residual_connections[i](x, residual)
        
        # Output layer
        out = self.output_layer(x)
        return out
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with softmax probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate embeddings before the output layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Embedding tensor
        """
        x = self.input_bn(x)
        for i, dense_block in enumerate(self.dense_blocks):
            residual = x
            x = dense_block(x)
            if self.use_residual:
                x = self.residual_connections[i](x, residual)
        return x
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fdsr_net(
    input_dim: int,
    num_classes: int = 2,
    model_size: str = 'small'
) -> FDSRNet:
    """
    Factory function to create FDSR-Net with predefined configurations.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        model_size: Model size configuration ('small', 'medium', 'large')
        
    Returns:
        FDSRNet model instance
    """
    config = {
        'small': {'hidden_dims': [32, 16], 'dropout_rate': 0.2},
        'medium': {'hidden_dims': [64, 32], 'dropout_rate': 0.3},
        'large': {'hidden_dims': [128, 64, 32], 'dropout_rate': 0.4}
    }
    
    if model_size not in config:
        raise ValueError(f"Invalid model_size: {model_size}. Choose from {list(config.keys())}")
    
    return FDSRNet(
        input_dim=input_dim,
        hidden_dims=config[model_size]['hidden_dims'],
        output_dim=num_classes,
        dropout_rate=config[model_size]['dropout_rate']
    )
