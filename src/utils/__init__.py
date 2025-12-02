"""GradientSimilarity-FedMeta: Utilities Module"""

from .metrics import (
    compute_accuracy,
    compute_f1_score,
    compute_auc,
    compute_precision_recall,
    evaluate_model
)
from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_client_similarity
)

__all__ = [
    'compute_accuracy',
    'compute_f1_score',
    'compute_auc',
    'compute_precision_recall',
    'evaluate_model',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_client_similarity'
]
