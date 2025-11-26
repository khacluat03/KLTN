"""
Training script for SASRec model.

Usage:
    python scripts/train_sasrec.py --dataset ml-100k --epochs 50
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macrec.systems.methods.sasrec import SASRec, SASRecTrainer, save_sasrec
from loguru import logger


def load_and_preprocess_data(data_path: str, max_len: int = 50):
    """Load and preprocess data into sequences."""
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        data = data.sort_values(['user_id', 'timestamp'])
    else:
        data = data.sort_values(['user_id'])
    
    # Create item ID mapping (1-indexed, 0 is padding)
    unique_items = sorted(data['item_id'].unique())
    item_id_map = {item: idx + 1 for idx, item in enumerate(unique_items)}
    n_items = len(unique_items)
    
    logger.info(f"Number of items: {n_items}")
    logger.info(f"Number of users: {data['user_id'].nunique()}")
    logger.info(f"Number of interactions: {len(data)}")
    
    # Group by user to create sequences
    user_sequences = {}
    sequences = []
    
    for user_id, group in data.groupby('user_id'):
        # Map item IDs
        seq = [item_id_map[item] for item in group['item_id'].values]

        # Skip users with too few interactions
        if len(seq) < 2:
            continue

        # Truncate sequences longer than max_len
        if len(seq) > max_len:
            seq = seq[-max_len:]  # Keep last max_len items

        # Ensure we still have at least 2 items after truncation
        if len(seq) >= 2:
            user_sequences[user_id] = seq
            sequences.append(seq)
    
    logger.info(f"Created {len(sequences)} sequences")
    logger.info(f"Average sequence length: {np.mean([len(s) for s in sequences]):.2f}")
    
    return sequences, item_id_map, user_sequences, n_items, unique_items


def train_sasrec(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 128,
    hidden_size: int = 50,
    n_blocks: int = 2,
    n_heads: int = 1,
    dropout: float = 0.2,
    max_len: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """Train SASRec model."""
    
    # Load data
    sequences, item_id_map, user_sequences, n_items, item_ids = load_and_preprocess_data(
        data_path, max_len
    )
    
    # Initialize model
    logger.info("Initializing SASRec model...")
    model = SASRec(
        n_items=n_items,
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        max_len=max_len,
        device=device
    )
    
    # Initialize trainer
    trainer = SASRecTrainer(
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_neg_samples=1
    )
    
    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        loss = trainer.train_epoch(sequences, list(range(1, n_items + 1)))
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    # Save model
    logger.info(f"Saving model to {output_path}...")
    save_sasrec(model, output_path, item_id_map, user_sequences)
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train SASRec model')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size')
    parser.add_argument('--n_blocks', type=int, default=2, help='Number of attention blocks')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Paths
    data_path = f"data/{args.dataset}/all.csv"
    output_path = args.output if args.output else f"saved_models/sasrec_{args.dataset}.pkl"
    
    # Train
    train_sasrec(
        data_path=data_path,
        output_path=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
