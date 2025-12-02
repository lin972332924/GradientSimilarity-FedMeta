# GradientSimilarity-FedMeta

A Meta Learning Enhanced Gradient Similarity Driven Personalized Federated Learning Method with a Lightweight Neural Network for Multi-Centre Liver Cancer Recurrence Prediction

## Overview

This repository contains the implementation of a personalized federated learning framework that combines meta-learning with gradient similarity-based client selection for liver cancer recurrence prediction across multiple medical centres. The method leverages a lightweight neural network architecture (FDSR-Net) to enable efficient training while maintaining privacy across distributed healthcare datasets.

### Key Features

- **Personalized Federated Learning**: Adaptive model personalization for heterogeneous client data distributions
- **Meta-Learning Enhancement**: MAML-inspired optimization for fast adaptation to local client data
- **Gradient Similarity-Based Selection**: Intelligent client selection based on gradient cosine similarity
- **Lightweight Neural Network**: FDSR-Net architecture optimized for medical tabular data
- **Privacy-Preserving**: No raw patient data sharing between medical centres

## Repository Structure

```
GradientSimilarity-FedMeta/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── __init__.py
│   │   └── fdsr_net.py      # FDSR-Net lightweight model
│   ├── federated/           # Federated learning components
│   │   ├── __init__.py
│   │   ├── server.py        # Federated server implementation
│   │   ├── client.py        # Federated client implementation
│   │   ├── aggregation.py   # Model aggregation strategies
│   │   └── gradient_similarity.py  # Gradient similarity computation
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── visualization.py # Result visualization
│   └── data/                # Data processing
│       ├── __init__.py
│       └── preprocessing.py # Data preprocessing utilities
├── data/
│   ├── raw/                 # Raw data placeholder
│   └── processed/           # Processed data placeholder
├── experiments/             # Experiment configurations and results
├── docs/                    # Additional documentation
├── requirements.txt         # Python dependencies
├── LICENSE                  # License file
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lin972332924/GradientSimilarity-FedMeta.git
cd GradientSimilarity-FedMeta
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.models.fdsr_net import FDSRNet
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient

# Initialize the model
model = FDSRNet(input_dim=30, hidden_dims=[64, 32], output_dim=2)

# Create federated server with gradient similarity selection
server = FederatedServer(
    model=model,
    num_clients=5,
    selection_strategy='gradient_similarity',
    meta_learning_rate=0.01
)

# Run federated training
server.train(num_rounds=100, local_epochs=5)
```

### Training with Custom Configuration

```python
from src.federated.server import FederatedServer

config = {
    'num_rounds': 100,
    'local_epochs': 5,
    'learning_rate': 0.001,
    'meta_learning_rate': 0.01,
    'batch_size': 32,
    'gradient_similarity_threshold': 0.7,
    'client_fraction': 0.4
}

server = FederatedServer(model=model, **config)
results = server.train()
```

## Method Description

### FDSR-Net Architecture

The lightweight neural network (FDSR-Net) is specifically designed for medical tabular data with:
- Feature-wise dense connections for capturing complex feature interactions
- Sparse residual connections for efficient gradient flow
- Dropout regularization for preventing overfitting on limited medical data

### Gradient Similarity-Based Client Selection

The gradient similarity mechanism selects clients with complementary gradient directions:

```
similarity(g_i, g_j) = cos(g_i, g_j) = (g_i · g_j) / (||g_i|| ||g_j||)
```

Clients are selected to maximize diversity while ensuring model convergence.

### Meta-Learning Enhancement

The meta-learning component enables fast adaptation through:
1. Meta-initialization of global model parameters
2. Client-specific fine-tuning with few-shot updates
3. Gradient-based meta-optimization across clients

## Experimental Results

Results will be published upon paper acceptance. The method has been evaluated on:
- Multi-centre liver cancer recurrence datasets
- Comparison with FedAvg, FedProx, Per-FedAvg, and other baselines
- Analysis of privacy-utility trade-offs

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gradientsimilarity_fedmeta2024,
  title={A Meta Learning Enhanced Gradient Similarity Driven Personalized Federated Learning Method with a Lightweight Neural Network for Multi Centre Liver Cancer Recurrence Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and collaboration opportunities, please open an issue or contact the authors.

## Acknowledgements

- This research was supported by [Funding Sources]
- We thank the medical centres for providing the anonymized datasets