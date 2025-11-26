from typing import Optional, List
from loguru import logger
import torch
import os

from macrec.tools.base import Tool


class SequentialPredictor(Tool):
    """
    Tool to predict next items using SASRec model.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(config_path, *args, **kwargs)
        
        self.model_path = self.config.get('model_path', 'saved_models/sasrec_ml100k.pkl')
        self.max_len = self.config.get('max_len', 50)
        self.device = self.config.get('device', 'cpu')
        
        self.model = None
        self.item_id_map = None
        self.user_sequences = None
        self.reverse_item_map = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(f"SASRec model not found at {self.model_path}. Please train the model first.")
    
    def _load_model(self):
        """Load trained SASRec model."""
        from macrec.systems.methods.sasrec import load_sasrec
        logger.info(f"Loading SASRec model from {self.model_path}...")
        self.model, self.item_id_map, self.user_sequences = load_sasrec(
            self.model_path, device=self.device
        )
        # Create reverse mapping (internal_id -> original_item_id)
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        logger.info("SASRec model loaded successfully")
    
    def reset(self) -> None:
        """Reset the tool state."""
        pass
    
    def predict_next(self, user_id: int, top_k: int = 5) -> List[int]:
        """
        Predict next K items for a user based on their sequence.
        
        Args:
            user_id: User ID
            top_k: Number of items to return
            
        Returns:
            List of item IDs (original IDs, not internal)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        # Get user sequence
        if user_id not in self.user_sequences:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        sequence = self.user_sequences[user_id]
        
        # Prepare sequence tensor
        seq_tensor = self._prepare_sequence(sequence)
        
        # Predict
        with torch.no_grad():
            scores = self.model.predict(seq_tensor)  # (1, n_items)
            scores = scores.squeeze(0)  # (n_items,)
            
            # Get top-K
            top_k_indices = torch.topk(scores, k=min(top_k, len(scores))).indices
            
            # Convert internal IDs back to original item IDs
            top_k_items = [self.reverse_item_map[idx.item() + 1] for idx in top_k_indices]
        
        return top_k_items
    
    def predict_next_with_scores(self, user_id: int, top_k: int = 5) -> List[tuple]:
        """
        Predict next K items with scores.
        
        Returns:
            List of (item_id, score) tuples
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        if user_id not in self.user_sequences:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        sequence = self.user_sequences[user_id]
        seq_tensor = self._prepare_sequence(sequence)
        
        with torch.no_grad():
            scores = self.model.predict(seq_tensor).squeeze(0)
            top_k_values, top_k_indices = torch.topk(scores, k=min(top_k, len(scores)))
            
            results = [
                (self.reverse_item_map[idx.item() + 1], score.item())
                for idx, score in zip(top_k_indices, top_k_values)
            ]
        
        return results
    
    def _prepare_sequence(self, sequence: List[int]) -> torch.Tensor:
        """Prepare sequence tensor for model input."""
        # Truncate if too long
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
        
        # Pad if too short
        padded = [0] * (self.max_len - len(sequence)) + sequence
        
        # Convert to tensor
        seq_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
        
        return seq_tensor
