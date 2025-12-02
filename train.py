#!/usr/bin/env python
"""
Example Training Script

Demonstrates how to use the GradientSimilarity-FedMeta framework
for federated learning with gradient similarity-based client selection.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models import FDSRNet
from src.federated import FederatedServer, FederatedClient
from src.data.preprocessing import (
    generate_synthetic_data,
    preprocess_features,
    split_data_federated,
    convert_to_tensors
)
from src.utils.visualization import plot_training_history, plot_client_similarity


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GradientSimilarity-FedMeta model'
    )
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples (for synthetic data)')
    parser.add_argument('--num_features', type=int, default=30,
                        help='Number of input features')
    
    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local training epochs')
    parser.add_argument('--client_fraction', type=float, default=0.6,
                        help='Fraction of clients to select per round')
    
    # Model parameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[64, 32],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Client learning rate')
    parser.add_argument('--meta_learning_rate', type=float, default=0.001,
                        help='Meta-learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Selection strategy
    parser.add_argument('--selection_strategy', type=str, default='gradient_similarity',
                        choices=['random', 'gradient_similarity', 'diverse'],
                        help='Client selection strategy')
    
    # Other parameters
    parser.add_argument('--use_meta_learning', action='store_true', default=True,
                        help='Use meta-learning enhancement')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Directory for output files')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GradientSimilarity-FedMeta Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Number of clients: {args.num_clients}")
    print(f"  - Number of rounds: {args.num_rounds}")
    print(f"  - Selection strategy: {args.selection_strategy}")
    print(f"  - Meta-learning: {args.use_meta_learning}")
    print("=" * 60)
    
    # Generate synthetic data (replace with real data loading)
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(
        num_samples=args.num_samples,
        num_features=args.num_features,
        num_classes=2,
        random_state=args.seed
    )
    
    # Preprocess features
    X, scaler, imputer = preprocess_features(X)
    
    # Split data for federated learning
    print(f"Splitting data across {args.num_clients} clients...")
    client_train, client_test, global_test = split_data_federated(
        X, y,
        num_clients=args.num_clients,
        split_strategy='iid',
        random_state=args.seed
    )
    
    # Convert to tensors
    client_train_tensors = [convert_to_tensors(data) for data in client_train]
    client_test_tensors = [convert_to_tensors(data) for data in client_test]
    
    # Create model
    print("\nInitializing FDSR-Net model...")
    model = FDSRNet(
        input_dim=args.num_features,
        hidden_dims=args.hidden_dims,
        output_dim=2,
        dropout_rate=args.dropout_rate
    )
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create federated server
    print("\nSetting up federated server...")
    server = FederatedServer(
        model=model,
        num_clients=args.num_clients,
        selection_strategy=args.selection_strategy,
        client_fraction=args.client_fraction,
        learning_rate=1.0,
        meta_learning_rate=args.meta_learning_rate,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Create clients
    server.create_clients(
        train_data=client_train_tensors,
        test_data=client_test_tensors,
        client_learning_rate=args.learning_rate
    )
    
    # Train
    print(f"\nStarting federated training for {args.num_rounds} rounds...")
    history = server.train(
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        use_meta_learning=args.use_meta_learning,
        verbose=True
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    final_acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0
    final_loss = history['test_loss'][-1] if history['test_loss'] else 0
    
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"Final Test Loss: {final_loss:.4f}")
    
    # Personalize clients
    print("\nPersonalizing client models...")
    server.personalize_clients(num_epochs=3)
    
    # Evaluate personalized models
    print("\nEvaluating personalized models:")
    for i, client in enumerate(server.clients):
        metrics = client.evaluate()
        print(f"  Client {i}: Accuracy = {metrics['accuracy']:.4f}")
    
    # Save results
    print(f"\nSaving results to {output_dir}")
    server.save_model(output_dir / 'global_model.pth')
    
    # Plot training history
    try:
        fig = plot_training_history(
            history,
            metrics=['loss', 'accuracy'],
            save_path=output_dir / 'training_history.png',
            title='Federated Training History'
        )
        print("Training history plot saved.")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    # Plot client similarity matrix
    try:
        similarity_matrix = server.get_client_similarity_matrix()
        fig = plot_client_similarity(
            similarity_matrix,
            save_path=output_dir / 'client_similarity.png',
            title='Client Gradient Similarity'
        )
        print("Client similarity plot saved.")
    except Exception as e:
        print(f"Could not save similarity plot: {e}")
    
    print("\nTraining completed successfully!")
    return history


if __name__ == '__main__':
    main()
