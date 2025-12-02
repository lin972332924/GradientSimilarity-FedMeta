"""GradientSimilarity-FedMeta: Data Processing Module"""

from .preprocessing import (
    load_data,
    preprocess_features,
    split_data_federated,
    create_data_loaders
)

__all__ = [
    'load_data',
    'preprocess_features',
    'split_data_federated',
    'create_data_loaders'
]
