import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, Tuple, List
from loguru import logger
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class MatrixFactorization:
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.1, n_epochs: int = 20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_ids: Optional[List[int]] = None
        self.item_ids: Optional[List[int]] = None
        self.user_id_map: Optional[dict] = None
        self.item_id_map: Optional[dict] = None
        self.global_mean: float = 0.0
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the Matrix Factorization model using SVD.
        Data should have columns: user_id, item_id, rating
        """
        logger.info("Training Matrix Factorization model...")
        
        # Create mappings
        self.user_ids = sorted(data['user_id'].unique())
        self.item_ids = sorted(data['item_id'].unique())
        self.user_id_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        self.global_mean = data['rating'].mean()
        
        # Create dense matrix for easier mean calculation
        R = np.zeros((n_users, n_items))
        for _, row in data.iterrows():
            u_idx = self.user_id_map[row['user_id']]
            i_idx = self.item_id_map[row['item_id']]
            R[u_idx, i_idx] = row['rating']
        
        # Calculate user means (only for rated items)
        user_means = np.zeros(n_users)
        for u_idx in range(n_users):
            rated_items = R[u_idx] > 0
            if rated_items.sum() > 0:
                user_means[u_idx] = R[u_idx][rated_items].mean()
            else:
                user_means[u_idx] = self.global_mean
        
        # Store user means for prediction
        self.user_means = user_means
        
        # Center ratings by subtracting user means
        R_centered = R.copy()
        for u_idx in range(n_users):
            rated_items = R[u_idx] > 0
            R_centered[u_idx][rated_items] -= user_means[u_idx]
        
        # Convert to sparse for SVD
        rows, cols = np.nonzero(R)
        centered_ratings = R_centered[rows, cols]
        R_sparse_centered = csr_matrix((centered_ratings, (rows, cols)), shape=(n_users, n_items))
        
        # Perform SVD on centered ratings
        k = min(self.n_factors, n_users - 1, n_items - 1)
        U, sigma, Vt = svds(R_sparse_centered, k=k)
        
        # Construct factors
        # U: (n_users, k), sigma: (k,), Vt: (k, n_items)
        sigma_diag = np.diag(np.sqrt(sigma))
        self.user_factors = np.dot(U, sigma_diag)
        self.item_factors = np.dot(sigma_diag, Vt).T  # Transpose to get (n_items, k)
        
        logger.info(f"Model trained. User factors: {self.user_factors.shape}, Item factors: {self.item_factors.shape}")

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
            
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            # Cold start: return global mean
            return self.global_mean
            
        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[item_id]
        
        # Predict centered rating
        pred_centered = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        
        # Add user mean back
        pred = pred_centered + self.user_means[u_idx]
        
        # Clip to valid range
        return float(np.clip(pred, 1.0, 5.0))

    def save(self, path: str) -> None:
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'global_mean': self.global_mean,
                'user_means': self.user_means
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.user_factors = state['user_factors']
            self.item_factors = state['item_factors']
            self.user_ids = state['user_ids']
            self.item_ids = state['item_ids']
            self.user_id_map = state['user_id_map']
            self.item_id_map = state['item_id_map']
            self.global_mean = state['global_mean']
            self.user_means = state.get('user_means', np.zeros(len(self.user_ids)))
        logger.info(f"Model loaded from {path}")
