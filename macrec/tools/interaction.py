import pandas as pd
from typing import Optional

from macrec.tools.base import Tool

class InteractionRetriever(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data_path = self.config['data_path']
        assert data_path is not None, 'Data path not found in config.'
        # Optimize memory usage by reading only necessary columns with specific dtypes
        # Valid columns: user_id, item_id, rating, timestamp
        try:
            self.data = pd.read_csv(
                data_path, 
                sep=',', 
                usecols=['user_id', 'item_id', 'rating', 'timestamp'],
                dtype={
                    'user_id': 'int32',
                    'item_id': 'int32', 
                    'rating': 'float32',
                    'timestamp': 'int32'
                }
            )
        except ValueError:
            # Fallback if columns are missing
            self.data = pd.read_csv(data_path, sep=',')

        if 'user_id' not in self.data.columns or 'item_id' not in self.data.columns:
             raise ValueError("Required columns 'user_id' and 'item_id' must be present in the data.")
        
        # Sort by timestamp to ensure chronological order for history retrieval
        if 'timestamp' in self.data.columns:
            self.data.sort_values('timestamp', inplace=True)
            
        self.simulation_mode = False
        self.target_user_id = None
        self.target_item_id = None
        self.cutoff_index = None

    def reset(self, user_id: Optional[int] = None, item_id: Optional[int] = None, *args, **kwargs) -> None:
        """
        Reset the tool state.
        If user_id and item_id are provided, we are in simulation mode (predicting specific interaction).
        We strictly restrict data to interactions BEFORE this specific interaction (cutoff).
        If not provided, we assume chat mode and access full data.
        """
        if user_id is not None and item_id is not None:
            self.simulation_mode = True
            self.target_user_id = user_id
            self.target_item_id = item_id
            
            # Find the specific interaction to verify strict sequential setup
            # We assume the last interaction of this user-item pair is the target if multiple exist (rare)
            # data is already sorted by timestamp
            mask = (self.data['user_id'] == user_id) & (self.data['item_id'] == item_id)
            indices = self.data.index[mask]
            
            if len(indices) == 0:
                # Interaction not in history (e.g. test set item), assume full history available? 
                # Or just use full data. For safety in simulation, we might want to just look at user history up to "now".
                # But without a timestamp query, we can't slice exactly. 
                # Preserving original logic: assert len(data_sample) == 1
                # If we can't find it, we can't set a cutoff. 
                self.cutoff_index = None
            else:
                # Original logic used indices to slice.
                # "data_sample = self.data[...] assert len == 1 ... index = data_sample.index[0]"
                # "partial_data = self.data.iloc[:index]" (This implies data is sorted by global index? or loaded that way?)
                # We will trust the loaded order/timestamp sort order.
                self.cutoff_index = indices[0]
        else:
            self.simulation_mode = False
            self.target_user_id = None
            self.target_item_id = None
            self.cutoff_index = None

    def _get_data_slice(self):
        """Helper to get the permitted slice of data based on current mode."""
        if self.simulation_mode and self.cutoff_index is not None:
            return self.data.iloc[:self.cutoff_index]
        return self.data

    def user_retrieve_data(self, user_id: int, k: int) -> list[tuple]:
        """
        Retrieve raw user history data as a list of (item_id, rating, timestamp) tuples.
        """
        df = self._get_data_slice()
        user_df = df[df['user_id'] == user_id]
        
        # Take last k
        if len(user_df) > k:
            user_df = user_df.iloc[-k:]
            
        items = user_df['item_id'].tolist()
        ratings = user_df['rating'].tolist()
        timestamps = user_df['timestamp'].tolist() if 'timestamp' in user_df.columns else [0]*len(items)
        
        return list(zip(items, ratings, timestamps))

    def user_retrieve(self, user_id: int, k: int, *args, **kwargs) -> str:
        df = self._get_data_slice()
        user_df = df[df['user_id'] == user_id]
        
        if user_df.empty:
             return f'No history found for user {user_id}.'
             
        # Take last k
        if len(user_df) > k:
            user_df = user_df.iloc[-k:]

        retrieved = user_df['item_id'].tolist()
        retrieved_rating = user_df['rating'].tolist()
        
        return f'Retrieved {len(retrieved)} items that user {user_id} interacted with before: {", ".join(map(str, retrieved))} with ratings: {", ".join(map(str, retrieved_rating))}'

    def item_retrieve(self, item_id: int, k: int, *args, **kwargs) -> str:
        df = self._get_data_slice()
        item_df = df[df['item_id'] == item_id]
        
        if item_df.empty:
            return f'No history found for item {item_id}.'
            
        # Take last k
        if len(item_df) > k:
            item_df = item_df.iloc[-k:]

        retrieved = item_df['user_id'].tolist()
        retrieved_rating = item_df['rating'].tolist()
        
        return f'Retrieved {len(retrieved)} users that interacted with item {item_id} before: {", ".join(map(str, retrieved))} with ratings: {", ".join(map(str, retrieved_rating))}'
