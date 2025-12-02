"""
Data Preprocessing Module

Implements data preprocessing utilities for liver cancer recurrence prediction
in federated learning settings.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(
    file_path: str,
    target_column: str = 'recurrence',
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        target_column: Name of target column
        feature_columns: Optional list of feature columns to use
        
    Returns:
        Tuple of (features, labels)
    """
    df = pd.read_csv(file_path)
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    return X, y


def preprocess_features(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    imputer: Optional[SimpleImputer] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler, SimpleImputer]:
    """
    Preprocess features with imputation and scaling.
    
    Args:
        X: Feature array
        scaler: Optional pre-fitted scaler
        imputer: Optional pre-fitted imputer
        fit: Whether to fit transformers
        
    Returns:
        Tuple of (processed features, scaler, imputer)
    """
    # Handle missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
    
    if fit:
        X = imputer.fit_transform(X)
    else:
        X = imputer.transform(X)
    
    # Standardize features
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, scaler, imputer


def encode_labels(
    y: np.ndarray,
    encoder: Optional[LabelEncoder] = None,
    fit: bool = True
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode categorical labels.
    
    Args:
        y: Label array
        encoder: Optional pre-fitted encoder
        fit: Whether to fit encoder
        
    Returns:
        Tuple of (encoded labels, encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
    
    if fit:
        y = encoder.fit_transform(y)
    else:
        y = encoder.transform(y)
    
    return y, encoder


def split_data_federated(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 5,
    split_strategy: str = 'iid',
    alpha: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], 
           List[Tuple[np.ndarray, np.ndarray]],
           Tuple[np.ndarray, np.ndarray]]:
    """
    Split data for federated learning.
    
    Args:
        X: Feature array
        y: Label array
        num_clients: Number of clients
        split_strategy: Strategy for splitting ('iid', 'non_iid', 'dirichlet')
        alpha: Concentration parameter for Dirichlet distribution
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (client_train_data, client_test_data, global_test_data)
    """
    np.random.seed(random_state)
    
    # Hold out global test set
    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if split_strategy == 'iid':
        client_train, client_test = _iid_split(X_main, y_main, num_clients, random_state)
    elif split_strategy == 'non_iid':
        client_train, client_test = _non_iid_split(X_main, y_main, num_clients, random_state)
    elif split_strategy == 'dirichlet':
        client_train, client_test = _dirichlet_split(X_main, y_main, num_clients, alpha, random_state)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    global_test = (X_test, y_test)
    
    return client_train, client_test, global_test


def _iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    random_state: int
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """IID data split: each client gets random samples."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    split_indices = np.array_split(indices, num_clients)
    
    client_train = []
    client_test = []
    
    for client_indices in split_indices:
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        # Split client data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=random_state
        )
        
        client_train.append((X_train, y_train))
        client_test.append((X_test, y_test))
    
    return client_train, client_test


def _non_iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    random_state: int
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Non-IID split: each client gets samples from limited classes."""
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # Each client gets 2 classes (for binary classification, this means imbalanced)
    classes_per_client = max(1, num_classes // 2)
    
    client_train = []
    client_test = []
    
    for i in range(num_clients):
        # Assign classes to this client
        client_classes = unique_classes[i % num_classes: i % num_classes + classes_per_client]
        if len(client_classes) < classes_per_client:
            client_classes = np.concatenate([
                client_classes, 
                unique_classes[:classes_per_client - len(client_classes)]
            ])
        
        # Get indices for these classes
        mask = np.isin(y, client_classes)
        client_indices = np.where(mask)[0]
        
        # Sample subset for this client
        np.random.seed(random_state + i)
        sample_size = min(len(client_indices), len(X) // num_clients)
        sampled_indices = np.random.choice(client_indices, sample_size, replace=False)
        
        X_client = X[sampled_indices]
        y_client = y[sampled_indices]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=random_state + i
        )
        
        client_train.append((X_train, y_train))
        client_test.append((X_test, y_test))
    
    return client_train, client_test


def _dirichlet_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    random_state: int
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Dirichlet distribution split for varying non-IIDness."""
    np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    n_samples = len(X)
    
    # Generate Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    
    for c in unique_classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        
        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Adjust for rounding errors
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        # Assign indices to clients
        start = 0
        for client_id, prop in enumerate(proportions):
            client_indices[client_id].extend(class_indices[start:start + prop])
            start += prop
    
    client_train = []
    client_test = []
    
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            # Fallback: give some random samples
            indices = np.random.choice(n_samples, n_samples // num_clients, replace=False)
        
        X_client = X[indices]
        y_client = y[indices]
        
        # Split into train/test
        if len(np.unique(y_client)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, random_state=random_state + i, stratify=y_client
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, random_state=random_state + i
            )
        
        client_train.append((X_train, y_train))
        client_test.append((X_test, y_test))
    
    return client_train, client_test


def create_data_loaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        train_data: Tuple of (features, labels) for training
        test_data: Optional tuple for testing
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    X_train, y_train = train_data
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_loader = None
    if test_data is not None:
        X_test, y_test = test_data
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def convert_to_tensors(
    data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        data: Tuple of (features, labels)
        
    Returns:
        Tuple of (feature tensor, label tensor)
    """
    X, y = data
    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 30,
    num_classes: int = 2,
    class_balance: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        num_classes: Number of classes
        class_balance: Proportion of positive class
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(num_samples, num_features)
    
    # Generate labels with specified class balance
    y = np.random.binomial(1, class_balance, num_samples)
    
    # Add some structure: make features slightly predictive of labels
    for i in range(min(5, num_features)):
        X[:, i] += y * np.random.uniform(0.5, 1.5)
    
    return X, y
