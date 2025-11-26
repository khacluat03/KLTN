from typing import Optional
from loguru import logger
import pandas as pd
import os

from macrec.tools.base import Tool
from macrec.tools.collaborative_filtering import CollaborativeFiltering

class RatingPredictor(Tool):
    """
    Tool to predict ratings using Collaborative Filtering (User-based or Item-based).
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(config_path, *args, **kwargs)
        
        self.cf_tool = CollaborativeFiltering(config_path, *args, **kwargs)
        
        self.cf_type = self.config.get('cf_type', 'user_based') # user_based or item_based
        self.similarity_method = self.config.get('similarity_method', 'pearson') # pearson or cosine
        self.k_neighbors = self.config.get('k_neighbors', 50)

    def reset(self) -> None:
        """Reset the tool state."""
        self.cf_tool.reset()

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair"""
        if self.cf_type == 'user_based':
            return self.cf_tool.predict_rating_user_based(
                user_id=user_id, 
                item_id=item_id, 
                method=self.similarity_method, 
                k=self.k_neighbors
            )
        elif self.cf_type == 'item_based':
            return self.cf_tool.predict_rating_item_based(
                user_id=user_id, 
                item_id=item_id, 
                method=self.similarity_method, 
                k=self.k_neighbors
            )
        else:
            logger.warning(f"Unknown cf_type: {self.cf_type}. Defaulting to user_based.")
            return self.cf_tool.predict_rating_user_based(
                user_id=user_id, 
                item_id=item_id, 
                method=self.similarity_method, 
                k=self.k_neighbors
            )
        
    def predict_batch(self, user_id: int, item_ids: list[int]) -> dict[int, float]:
        """Predict ratings for a list of items"""
        results = {}
        for item_id in item_ids:
            results[item_id] = self.predict(user_id, item_id)
        return results
