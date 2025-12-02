"""GradientSimilarity-FedMeta: Federated Learning Module"""

from .server import FederatedServer
from .client import FederatedClient
from .aggregation import aggregate_models, fedavg_aggregate
from .gradient_similarity import compute_gradient_similarity, select_clients_by_similarity

__all__ = [
    'FederatedServer',
    'FederatedClient',
    'aggregate_models',
    'fedavg_aggregate',
    'compute_gradient_similarity',
    'select_clients_by_similarity'
]
