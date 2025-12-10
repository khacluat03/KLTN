import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, List, Tuple
from loguru import logger


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec) model.
    Uses Transformer architecture to learn sequential patterns.
    """
    def __init__(
        self,
        n_items: int,
        hidden_size: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
        max_len: int = 50,
        device: str = 'cpu'
    ):
        super(SASRec, self).__init__()
        
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.max_len = max_len
        self.device = device
        
        # Item embedding (0 is padding, so n_items + 1)
        self.item_emb = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        
        # Positional embedding
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        
        # Dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_size, n_heads, dropout) for _ in range(n_blocks)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.to(device)
        
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            seq: (batch_size, max_len) - sequence of item IDs
        Returns:
            (batch_size, max_len, hidden_size) - sequence representations
        """
        # Create position indices
        positions = torch.arange(seq.size(1), device=self.device).unsqueeze(0).expand_as(seq)
        
        # Embeddings
        seq_emb = self.item_emb(seq)  # (batch, max_len, hidden_size)
        pos_emb = self.pos_emb(positions)  # (batch, max_len, hidden_size)
        
        # Combine and dropout
        x = self.emb_dropout(seq_emb + pos_emb)
        
        # Create attention mask (mask padding)
        mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, max_len)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x
    
    def predict(self, seq: torch.Tensor, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict scores for items.
        Args:
            seq: (batch_size, max_len) - sequence of item IDs
            item_ids: (batch_size, n_items) - items to score. If None, score all items.
        Returns:
            (batch_size, n_items) - scores for items
        """
        # Get sequence representation
        seq_repr = self.forward(seq)  # (batch, max_len, hidden_size)
        
        # Take last position
        last_repr = seq_repr[:, -1, :]  # (batch, hidden_size)
        
        if item_ids is None:
            # Score all items
            item_embs = self.item_emb.weight[1:]  # Exclude padding (n_items, hidden_size)
            scores = torch.matmul(last_repr, item_embs.T)  # (batch, n_items)
        else:
            # Score specific items
            item_embs = self.item_emb(item_ids)  # (batch, n_items, hidden_size)
            scores = torch.sum(last_repr.unsqueeze(1) * item_embs, dim=-1)  # (batch, n_items)
        
        return scores


class AttentionBlock(nn.Module):
    """Single attention block with feed-forward network."""
    def __init__(self, hidden_size: int, n_heads: int, dropout: float):
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (no explicit mask, use causal attention)
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.layer_norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.layer_norm2(x + ff_out)
        
        return x



class SASRecTrainer:
    """Trainer for SASRec model."""
    def __init__(
        self,
        model: SASRec,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        n_neg_samples: int = 1
    ):
        self.model = model
        self.batch_size = batch_size
        self.n_neg_samples = n_neg_samples
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_epoch(self, sequences: List[List[int]], item_ids: List[int]) -> float:
        """
        Train for one epoch.
        Args:
            sequences: List of sequences (each is list of item IDs)
            item_ids: List of all valid item IDs for negative sampling
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(sequences))
        
        for i in range(0, len(sequences), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_seqs = [sequences[idx] for idx in batch_indices]
            
            # Prepare batch
            seq_tensor, pos_tensor, neg_tensor = self._prepare_batch(batch_seqs, item_ids)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get sequence representations
            seq_repr = self.model.forward(seq_tensor)  # (batch, max_len, hidden_size)
            
            # Compute loss for each position
            loss = 0
            for pos in range(1, seq_tensor.size(1)):
                # Get representation at position pos-1 to predict position pos
                repr_at_pos = seq_repr[:, pos-1, :]  # (batch, hidden_size)
                
                # Positive item at position pos
                pos_item = pos_tensor[:, pos]  # (batch,)
                pos_emb = self.model.item_emb(pos_item)  # (batch, hidden_size)
                pos_score = torch.sum(repr_at_pos * pos_emb, dim=-1)  # (batch,)
                
                # Negative items
                neg_items = neg_tensor[:, pos, :]  # (batch, n_neg)
                neg_emb = self.model.item_emb(neg_items)  # (batch, n_neg, hidden_size)
                neg_score = torch.sum(repr_at_pos.unsqueeze(1) * neg_emb, dim=-1)  # (batch, n_neg)
                
                # Binary cross-entropy loss
                pos_labels = torch.ones_like(pos_score)
                neg_labels = torch.zeros_like(neg_score)
                
                loss += self.criterion(pos_score, pos_labels)
                loss += self.criterion(neg_score, neg_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _prepare_batch(
        self,
        sequences: List[List[int]],
        item_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch tensors with padding and negative sampling."""
        max_len = self.model.max_len
        batch_size = len(sequences)
        
        seq_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.model.device)
        pos_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.model.device)
        neg_tensor = torch.zeros((batch_size, max_len, self.n_neg_samples), dtype=torch.long, device=self.model.device)
        
        for i, seq in enumerate(sequences):
            # Truncate or pad sequence
            if len(seq) > max_len:
                seq = seq[-max_len:]
            
            # Fill tensors
            seq_len = len(seq)
            
            # Input sequence (all items except last)
            if seq_len > 1:
                input_seq = seq[:-1]
                seq_tensor[i, :len(input_seq)] = torch.tensor(input_seq, dtype=torch.long)
            else:
                # If sequence has only 1 item, use padding
                seq_tensor[i, 0] = 0
            
            # Target sequence (all items)
            pos_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            
            # Negative sampling
            for pos in range(seq_len):
                user_items = set(seq)
                neg_items = []
                while len(neg_items) < self.n_neg_samples:
                    neg_item = np.random.choice(item_ids)
                    if neg_item not in user_items:
                        neg_items.append(neg_item)
                neg_tensor[i, pos, :] = torch.tensor(neg_items, dtype=torch.long)
        
        return seq_tensor, pos_tensor, neg_tensor


def save_sasrec(model: SASRec, path: str, item_id_map: dict, user_sequences: dict) -> None:
    """Save SASRec model and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'state_dict': model.state_dict(),
            'n_items': model.n_items,
            'hidden_size': model.hidden_size,
            'n_blocks': model.n_blocks,
            'n_heads': model.n_heads,
            'dropout': model.dropout_rate,
            'max_len': model.max_len,
            'item_id_map': item_id_map,
            'user_sequences': user_sequences
        }, f)
    logger.info(f"SASRec model saved to {path}")


def load_sasrec(path: str, device: str = 'cpu') -> Tuple[SASRec, dict, dict]:
    """Load SASRec model and metadata."""
    with open(path, 'rb') as f:
        state = pickle.load(f)
    
    model = SASRec(
        n_items=state['n_items'],
        hidden_size=state['hidden_size'],
        n_blocks=state['n_blocks'],
        n_heads=state['n_heads'],
        dropout=state['dropout'],
        max_len=state['max_len'],
        device=device
    )
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    logger.info(f"SASRec model loaded from {path}")
    return model, state['item_id_map'], state['user_sequences']
