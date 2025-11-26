"""
Collaborative Filtering Tool
Implement các phương pháp Collaborative Filtering:
- User-based CF với Pearson Correlation
- User-based CF với Cosine Similarity
- Item-based CF với Pearson Correlation
- Item-based CF với Cosine Similarity
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from loguru import logger

from macrec.tools.base import Tool


class CollaborativeFiltering(Tool):
    """
    Tool thực hiện Collaborative Filtering với các phương pháp:
    - Pearson Correlation Coefficient
    - Cosine Similarity
    """
    
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(config_path, *args, **kwargs)
        
        # Load dữ liệu interaction
        data_path = self.config.get('data_path')
        if data_path:
            self.data = pd.read_csv(data_path, sep=',')
            assert 'user_id' in self.data.columns, 'user_id not found in data.'
            assert 'item_id' in self.data.columns, 'item_id not found in data.'
            assert 'rating' in self.data.columns, 'rating not found in data.'
        else:
            self.data = None
        
        # Tạo user-item rating matrix
        self.rating_matrix: Optional[pd.DataFrame] = None
        self.user_means: Optional[pd.Series] = None
        self.item_means: Optional[pd.Series] = None
        
        # Similarity matrices (cache)
        self.user_similarity_pearson: Optional[pd.DataFrame] = None
        self.user_similarity_cosine: Optional[pd.DataFrame] = None
        self.item_similarity_pearson: Optional[pd.DataFrame] = None
        self.item_similarity_cosine: Optional[pd.DataFrame] = None
        
        # Config
        self.min_common_items = self.config.get('min_common_items', 1)  # Số item tối thiểu chung giữa 2 users
        self.min_common_users = self.config.get('min_common_users', 1)  # Số user tối thiểu chung giữa 2 items
        self.k_neighbors = self.config.get('k_neighbors', 50)  # Số neighbors để dự đoán

        # Model paths for pre-computed similarities
        self.model_base_path = self.config.get('model_path', 'saved_models')

        # Lazy load rating matrix
        self._matrix_built = False

        # Try to load pre-computed similarity matrices
        self._load_pretrained_models()

    def _load_pretrained_models(self) -> None:
        """Load pre-computed similarity matrices if available."""
        import pickle
        import os

        model_paths = {
            'user_similarity_pearson': os.path.join(self.model_base_path, 'cf', 'user_sim_pearson.pkl'),
            'user_similarity_cosine': os.path.join(self.model_base_path, 'cf', 'user_sim_cosine.pkl'),
            'item_similarity_pearson': os.path.join(self.model_base_path, 'cf', 'item_sim_pearson.pkl'),
            'item_similarity_cosine': os.path.join(self.model_base_path, 'cf', 'item_sim_cosine.pkl'),
            'cf_metadata': os.path.join(self.model_base_path, 'cf', 'cf_metadata.pkl'),
        }

        for attr_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading pre-computed {attr_name} from {model_path}")
                    with open(model_path, 'rb') as f:
                        if attr_name == 'cf_metadata':
                            # Special handling for metadata
                            metadata = pickle.load(f)
                            self.rating_matrix = metadata.get('rating_matrix')
                            self.user_means = metadata.get('user_means')
                            self.item_means = metadata.get('item_means')
                            self.data = metadata.get('data')
                            # Mark matrix as built to skip lazy loading
                            self._matrix_built = True
                            logger.info("Successfully loaded metadata and rating matrix")
                        else:
                            setattr(self, attr_name, pickle.load(f))
                            logger.info(f"Successfully loaded {attr_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {attr_name} from {model_path}: {e}")
            else:
                logger.debug(f"Pre-computed {attr_name} not found at {model_path}")

    def reset(self, *args, **kwargs) -> None:
        """Reset tool state"""
        # Có thể clear cache nếu cần
        pass
    
    def _ensure_matrix_built(self) -> None:
        """Ensure rating matrix is built (lazy loading)"""
        if not self._matrix_built:
            self._build_rating_matrix()
            self._matrix_built = True

    def _build_rating_matrix(self) -> None:
        """Xây dựng user-item rating matrix"""
        if self.data is None:
            logger.warning("No data available to build rating matrix")
            return

        # Tạo pivot table: users x items
        self.rating_matrix = self.data.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0  # 0 nghĩa là chưa đánh giá
        )

        # Tính mean rating cho mỗi user và item
        self.user_means = self.rating_matrix.mean(axis=1)
        self.item_means = self.rating_matrix.mean(axis=0)

        logger.info(f"Rating matrix built: {self.rating_matrix.shape[0]} users x {self.rating_matrix.shape[1]} items")
    
    def pearson_correlation(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Tính Pearson Correlation Coefficient giữa 2 vectors
        
        Công thức: r = Σ((x_i - x̄)(y_i - ȳ)) / sqrt(Σ(x_i - x̄)² * Σ(y_i - ȳ)²)
        
        Args:
            vec1: Vector 1 (ratings của user/item 1)
            vec2: Vector 2 (ratings của user/item 2)
            
        Returns:
            Pearson correlation coefficient (-1 đến 1)
        """
        # Chỉ tính trên các items/users mà cả 2 đều có rating (khác 0)
        common_mask = (vec1 != 0) & (vec2 != 0)
        
        if common_mask.sum() < self.min_common_items:
            return 0.0
        
        vec1_common = vec1[common_mask]
        vec2_common = vec2[common_mask]
        
        # Tính mean
        mean1 = vec1_common.mean()
        mean2 = vec2_common.mean()
        
        # Tính numerator và denominator
        numerator = ((vec1_common - mean1) * (vec2_common - mean2)).sum()
        denominator1 = ((vec1_common - mean1) ** 2).sum()
        denominator2 = ((vec2_common - mean2) ** 2).sum()
        
        denominator = np.sqrt(denominator1 * denominator2)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        
        # Đảm bảo giá trị trong khoảng [-1, 1]
        return np.clip(correlation, -1.0, 1.0)
    
    def cosine_similarity(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Tính Cosine Similarity giữa 2 vectors
        
        Công thức: cos(θ) = (A · B) / (||A|| * ||B||)
        
        Args:
            vec1: Vector 1 (ratings của user/item 1)
            vec2: Vector 2 (ratings của user/item 2)
            
        Returns:
            Cosine similarity (0 đến 1)
        """
        # Chỉ tính trên các items/users mà cả 2 đều có rating (khác 0)
        common_mask = (vec1 != 0) & (vec2 != 0)
        
        if common_mask.sum() < self.min_common_items:
            return 0.0
        
        vec1_common = vec1[common_mask]
        vec2_common = vec2[common_mask]
        
        # Dot product
        dot_product = (vec1_common * vec2_common).sum()
        
        # Norms
        norm1 = np.sqrt((vec1_common ** 2).sum())
        norm2 = np.sqrt((vec2_common ** 2).sum())
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Đảm bảo giá trị trong khoảng [0, 1]
        return np.clip(similarity, 0.0, 1.0)
    
    def compute_user_similarity_pearson(self) -> pd.DataFrame:
        """Tính user-user similarity matrix sử dụng Pearson Correlation"""
        # Return pre-computed if available
        if self.user_similarity_pearson is not None:
            return self.user_similarity_pearson

        self._ensure_matrix_built()
        
        logger.info("Computing user-user similarity matrix using Pearson Correlation...")
        n_users = len(self.rating_matrix)
        similarity_matrix = pd.DataFrame(
            index=self.rating_matrix.index,
            columns=self.rating_matrix.index,
            dtype=float
        )
        
        # Tính similarity cho từng cặp users
        for i, user1 in enumerate(self.rating_matrix.index):
            if i % 100 == 0:
                logger.debug(f"Processing user {i+1}/{n_users}")
            for user2 in self.rating_matrix.index:
                if user1 == user2:
                    similarity_matrix.loc[user1, user2] = 1.0
                elif pd.isna(similarity_matrix.loc[user2, user1]):
                    # Chưa tính, tính mới
                    sim = self.pearson_correlation(
                        self.rating_matrix.loc[user1],
                        self.rating_matrix.loc[user2]
                    )
                    similarity_matrix.loc[user1, user2] = sim
                    similarity_matrix.loc[user2, user1] = sim  # Đối xứng
                else:
                    # Đã tính rồi, lấy giá trị đối xứng
                    similarity_matrix.loc[user1, user2] = similarity_matrix.loc[user2, user1]
        
        self.user_similarity_pearson = similarity_matrix
        logger.info("User similarity matrix (Pearson) computed")
        return similarity_matrix
    
    def compute_user_similarity_cosine(self) -> pd.DataFrame:
        """Tính user-user similarity matrix sử dụng Cosine Similarity"""
        # Return pre-computed if available
        if self.user_similarity_cosine is not None:
            return self.user_similarity_cosine

        self._ensure_matrix_built()
        
        logger.info("Computing user-user similarity matrix using Cosine Similarity...")
        n_users = len(self.rating_matrix)
        similarity_matrix = pd.DataFrame(
            index=self.rating_matrix.index,
            columns=self.rating_matrix.index,
            dtype=float
        )
        
        # Tính similarity cho từng cặp users
        for i, user1 in enumerate(self.rating_matrix.index):
            if i % 100 == 0:
                logger.debug(f"Processing user {i+1}/{n_users}")
            for user2 in self.rating_matrix.index:
                if user1 == user2:
                    similarity_matrix.loc[user1, user2] = 1.0
                elif pd.isna(similarity_matrix.loc[user2, user1]):
                    # Chưa tính, tính mới
                    sim = self.cosine_similarity(
                        self.rating_matrix.loc[user1],
                        self.rating_matrix.loc[user2]
                    )
                    similarity_matrix.loc[user1, user2] = sim
                    similarity_matrix.loc[user2, user1] = sim  # Đối xứng
                else:
                    # Đã tính rồi, lấy giá trị đối xứng
                    similarity_matrix.loc[user1, user2] = similarity_matrix.loc[user2, user1]
        
        self.user_similarity_cosine = similarity_matrix
        logger.info("User similarity matrix (Cosine) computed")
        return similarity_matrix
    
    def compute_item_similarity_pearson(self) -> pd.DataFrame:
        """Tính item-item similarity matrix sử dụng Pearson Correlation"""
        # Return pre-computed if available
        if self.item_similarity_pearson is not None:
            return self.item_similarity_pearson

        self._ensure_matrix_built()
        
        logger.info("Computing item-item similarity matrix using Pearson Correlation...")
        # Transpose để có item x user matrix
        item_matrix = self.rating_matrix.T
        n_items = len(item_matrix)
        similarity_matrix = pd.DataFrame(
            index=item_matrix.index,
            columns=item_matrix.index,
            dtype=float
        )
        
        # Tính similarity cho từng cặp items
        for i, item1 in enumerate(item_matrix.index):
            if i % 100 == 0:
                logger.debug(f"Processing item {i+1}/{n_items}")
            for item2 in item_matrix.index:
                if item1 == item2:
                    similarity_matrix.loc[item1, item2] = 1.0
                elif pd.isna(similarity_matrix.loc[item2, item1]):
                    # Chưa tính, tính mới
                    sim = self.pearson_correlation(
                        item_matrix.loc[item1],
                        item_matrix.loc[item2]
                    )
                    similarity_matrix.loc[item1, item2] = sim
                    similarity_matrix.loc[item2, item1] = sim  # Đối xứng
                else:
                    # Đã tính rồi, lấy giá trị đối xứng
                    similarity_matrix.loc[item1, item2] = similarity_matrix.loc[item2, item1]
        
        self.item_similarity_pearson = similarity_matrix
        logger.info("Item similarity matrix (Pearson) computed")
        return similarity_matrix
    
    def compute_item_similarity_cosine(self) -> pd.DataFrame:
        """Tính item-item similarity matrix sử dụng Cosine Similarity"""
        # Return pre-computed if available
        if self.item_similarity_cosine is not None:
            return self.item_similarity_cosine

        self._ensure_matrix_built()
        
        logger.info("Computing item-item similarity matrix using Cosine Similarity...")
        # Transpose để có item x user matrix
        item_matrix = self.rating_matrix.T
        n_items = len(item_matrix)
        similarity_matrix = pd.DataFrame(
            index=item_matrix.index,
            columns=item_matrix.index,
            dtype=float
        )
        
        # Tính similarity cho từng cặp items
        for i, item1 in enumerate(item_matrix.index):
            if i % 100 == 0:
                logger.debug(f"Processing item {i+1}/{n_items}")
            for item2 in item_matrix.index:
                if item1 == item2:
                    similarity_matrix.loc[item1, item2] = 1.0
                elif pd.isna(similarity_matrix.loc[item2, item1]):
                    # Chưa tính, tính mới
                    sim = self.cosine_similarity(
                        item_matrix.loc[item1],
                        item_matrix.loc[item2]
                    )
                    similarity_matrix.loc[item1, item2] = sim
                    similarity_matrix.loc[item2, item1] = sim  # Đối xứng
                else:
                    # Đã tính rồi, lấy giá trị đối xứng
                    similarity_matrix.loc[item1, item2] = similarity_matrix.loc[item2, item1]
        
        self.item_similarity_cosine = similarity_matrix
        logger.info("Item similarity matrix (Cosine) computed")
        return similarity_matrix
    
    def predict_rating_user_based(
        self,
        user_id: int,
        item_id: int,
        method: str = 'pearson',
        k: Optional[int] = None
    ) -> float:
        """
        Dự đoán rating của user cho item sử dụng User-based Collaborative Filtering

        Công thức: r_ui = r̄_u + Σ(sim(u,v) * (r_vi - r̄_v)) / Σ|sim(u,v)|

        Args:
            user_id: ID của user
            item_id: ID của item
            method: 'pearson' hoặc 'cosine'
            k: Số neighbors (None = dùng tất cả)

        Returns:
            Predicted rating
        """
        self._ensure_matrix_built()
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Please load data first.")
        
        if user_id not in self.rating_matrix.index:
            return self.user_means.mean() if self.user_means is not None else 0.0
        
        if item_id not in self.rating_matrix.columns:
            return self.item_means.mean() if self.item_means is not None else 0.0
        
        # Kiểm tra xem user đã rate item này chưa
        if self.rating_matrix.loc[user_id, item_id] != 0:
            return self.rating_matrix.loc[user_id, item_id]
        
        # Lấy similarity matrix
        if method == 'pearson':
            similarity_matrix = self.compute_user_similarity_pearson()
        elif method == 'cosine':
            similarity_matrix = self.compute_user_similarity_cosine()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'cosine'")
        
        # Lấy các users đã rate item này
        item_ratings = self.rating_matrix.loc[:, item_id]
        rated_users = item_ratings[item_ratings != 0].index
        
        if len(rated_users) == 0:
            return self.user_means.loc[user_id] if user_id in self.user_means.index else 0.0
        
        # Lấy similarities với các users đã rate
        user_similarities = similarity_matrix.loc[user_id, rated_users]
        
        # Lọc bỏ các similarities <= 0 (chỉ dùng positive correlations)
        positive_mask = user_similarities > 0
        if positive_mask.sum() == 0:
            return self.user_means.loc[user_id] if user_id in self.user_means.index else 0.0

        # Filter both similarities and rated_users using positive_mask
        positive_users = user_similarities[positive_mask].index
        user_similarities = user_similarities.loc[positive_users]
        rated_users = positive_users
        
        # Chọn top k neighbors
        if k is not None and k < len(rated_users):
            # Sort by similarity and take top k
            sorted_indices = user_similarities.nlargest(k).index
            user_similarities = user_similarities.loc[sorted_indices]
            rated_users = sorted_indices  # rated_users should match user_similarities index
        
        # Tính predicted rating
        user_mean = self.user_means.loc[user_id]
        numerator = 0.0
        denominator = 0.0
        
        for other_user in rated_users:
            sim = user_similarities.loc[other_user]
            other_user_mean = self.user_means.loc[other_user]
            rating = self.rating_matrix.loc[other_user, item_id]
            
            numerator += sim * (rating - other_user_mean)
            denominator += abs(sim)
        
        if denominator == 0:
            return user_mean
        
        predicted_rating = user_mean + (numerator / denominator)
        
        # Clip về khoảng rating hợp lệ (thường là 1-5)
        return np.clip(predicted_rating, 1.0, 5.0)
    
    def predict_rating_item_based(
        self,
        user_id: int,
        item_id: int,
        method: str = 'pearson',
        k: Optional[int] = None
    ) -> float:
        """
        Dự đoán rating của user cho item sử dụng Item-based Collaborative Filtering

        Công thức: r_ui = r̄_i + Σ(sim(i,j) * (r_uj - r̄_j)) / Σ|sim(i,j)|

        Args:
            user_id: ID của user
            item_id: ID của item
            method: 'pearson' hoặc 'cosine'
            k: Số neighbors (None = dùng tất cả)

        Returns:
            Predicted rating
        """
        self._ensure_matrix_built()
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Please load data first.")
        
        if user_id not in self.rating_matrix.index:
            return self.user_means.mean() if self.user_means is not None else 0.0
        
        if item_id not in self.rating_matrix.columns:
            return self.item_means.mean() if self.item_means is not None else 0.0
        
        # Kiểm tra xem user đã rate item này chưa
        if self.rating_matrix.loc[user_id, item_id] != 0:
            return self.rating_matrix.loc[user_id, item_id]
        
        # Lấy similarity matrix
        if method == 'pearson':
            similarity_matrix = self.compute_item_similarity_pearson()
        elif method == 'cosine':
            similarity_matrix = self.compute_item_similarity_cosine()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'cosine'")
        
        # Lấy các items mà user đã rate
        user_ratings = self.rating_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings != 0].index
        
        if len(rated_items) == 0:
            return self.item_means.loc[item_id] if item_id in self.item_means.index else 0.0
        
        # Lấy similarities với các items đã rate
        item_similarities = similarity_matrix.loc[item_id, rated_items]
        
        # Lọc bỏ các similarities <= 0 (chỉ dùng positive correlations)
        positive_mask = item_similarities > 0
        if positive_mask.sum() == 0:
            return self.item_means.loc[item_id] if item_id in self.item_means.index else 0.0
        
        item_similarities = item_similarities[positive_mask]
        rated_items = rated_items[positive_mask]
        
        # Chọn top k neighbors
        if k is not None and k < len(rated_items):
            top_k_indices = item_similarities.nlargest(k).index
            item_similarities = item_similarities[top_k_indices]
            rated_items = rated_items[top_k_indices]
        
        # Tính predicted rating
        item_mean = self.item_means.loc[item_id]
        numerator = 0.0
        denominator = 0.0
        
        for other_item in rated_items:
            sim = item_similarities.loc[other_item]
            other_item_mean = self.item_means.loc[other_item]
            rating = self.rating_matrix.loc[user_id, other_item]
            
            numerator += sim * (rating - other_item_mean)
            denominator += abs(sim)
        
        if denominator == 0:
            return item_mean
        
        predicted_rating = item_mean + (numerator / denominator)
        
        # Clip về khoảng rating hợp lệ (thường là 1-5)
        return np.clip(predicted_rating, 1.0, 5.0)
    
    def recommend_items_user_based(
        self,
        user_id: int,
        n_items: int = 10,
        method: str = 'pearson',
        k: Optional[int] = None,
        candidates: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend items cho user sử dụng User-based CF

        Args:
            user_id: ID của user
            n_items: Số items muốn recommend
            method: 'pearson' hoặc 'cosine'
            k: Số neighbors
            candidates: Optional list of candidate item_ids để filter. Nếu None, dùng tất cả unrated items

        Returns:
            List of (item_id, predicted_rating) tuples, sorted by rating descending
        """
        self._ensure_matrix_built()
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Please load data first.")

        # Check if user exists in rating matrix
        if user_id not in self.rating_matrix.index:
            logger.warning(f"User {user_id} not found in rating matrix. Returning empty recommendations.")
            return []

        # Lấy các items mà user chưa rate
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index

        # Filter theo candidates nếu được cung cấp
        if candidates is not None:
            # Chỉ giữ lại items trong candidates và chưa được user rate
            candidates_set = set(candidates)
            unrated_items = [item for item in unrated_items if item in candidates_set]

        # Nếu có candidates, tối ưu hóa: chỉ tính similarity cho relevant users
        if candidates is not None and len(candidates) > 0:
            predictions = self._recommend_with_candidates_optimization(
                user_id, unrated_items, method, k, n_items
            )
        else:
            # Dự đoán rating cho các items đã filter
            predictions = []
            for item_id in unrated_items:
                pred_rating = self.predict_rating_user_based(user_id, item_id, method, k)
                predictions.append((item_id, pred_rating))

        # Sắp xếp theo rating giảm dần và lấy top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_items]

    def _recommend_with_candidates_optimization(
        self,
        user_id: int,
        unrated_items: List[int],
        method: str,
        k: Optional[int],
        n_items: int
    ) -> List[Tuple[int, float]]:
        """
        Tối ưu hóa CF khi có candidates: chỉ tính similarity cho relevant users
        """
        logger.info(f"Using optimized CF for user {user_id} with {len(unrated_items)} candidates")

        # Tìm users đã rate ít nhất một trong các unrated_items
        relevant_users = set()
        for item_id in unrated_items:
            # Tìm users đã rate item này
            item_ratings = self.rating_matrix[item_id]
            users_who_rated = item_ratings[item_ratings > 0].index
            relevant_users.update(users_who_rated)

        # Loại bỏ user target khỏi relevant_users
        relevant_users.discard(user_id)

        logger.info(f"Found {len(relevant_users)} relevant users for similarity calculation")

        # Nếu không có relevant users, fall back to normal prediction
        if not relevant_users:
            predictions = []
            for item_id in unrated_items:
                pred_rating = self.predict_rating_user_based(user_id, item_id, method, k)
                predictions.append((item_id, pred_rating))
            return predictions

        # Tính similarity chỉ cho relevant users
        user_similarities = {}
        user_ratings = self.rating_matrix.loc[user_id]

        for other_user_id in relevant_users:
            other_ratings = self.rating_matrix.loc[other_user_id]

            # Tính similarity chỉ trên các items mà cả hai user đều đã rate
            common_items = (user_ratings > 0) & (other_ratings > 0)

            if common_items.sum() < 2:  # Cần ít nhất 2 items chung
                continue

            if method == 'pearson':
                # Tính Pearson correlation
                try:
                    sim = user_ratings[common_items].corr(other_ratings[common_items])
                    if not pd.isna(sim):
                        user_similarities[other_user_id] = sim
                except:
                    continue
            elif method == 'cosine':
                # Tính Cosine similarity
                try:
                    sim = self.cosine_similarity(user_ratings, other_ratings)
                    user_similarities[other_user_id] = sim
                except:
                    continue

        # Dự đoán rating sử dụng chỉ relevant neighbors
        predictions = []
        for item_id in unrated_items:
            pred_rating = self._predict_rating_with_similarities(
                user_id, item_id, user_similarities, k
            )
            predictions.append((item_id, pred_rating))

        return predictions

    def _predict_rating_with_similarities(
        self,
        user_id: int,
        item_id: int,
        user_similarities: Dict[int, float],
        k: Optional[int]
    ) -> float:
        """
        Dự đoán rating sử dụng pre-computed similarities
        """
        if not user_similarities:
            return self.rating_matrix.loc[:, item_id].mean()  # Global average

        # Lấy k neighbors gần nhất (hoặc tất cả nếu ít hơn k)
        sorted_neighbors = sorted(
            user_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if k is not None:
            neighbors = sorted_neighbors[:k]
        else:
            neighbors = sorted_neighbors

        # Chỉ lấy neighbors có similarity > 0
        neighbors = [(uid, sim) for uid, sim in neighbors if sim > 0]

        if not neighbors:
            return self.rating_matrix.loc[:, item_id].mean()

        # Dự đoán theo công thức user-based CF
        numerator = 0
        denominator = 0

        for neighbor_id, similarity in neighbors:
            neighbor_rating = self.rating_matrix.loc[neighbor_id, item_id]
            if neighbor_rating > 0:  # Neighbor đã rate item này
                numerator += similarity * neighbor_rating
                denominator += abs(similarity)

        if denominator == 0:
            return self.rating_matrix.loc[:, item_id].mean()

        return numerator / denominator

    def recommend_items_item_based(
        self,
        user_id: int,
        n_items: int = 10,
        method: str = 'pearson',
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend items cho user sử dụng Item-based CF

        Args:
            user_id: ID của user
            n_items: Số items muốn recommend
            method: 'pearson' hoặc 'cosine'
            k: Số neighbors

        Returns:
            List of (item_id, predicted_rating) tuples, sorted by rating descending
        """
        self._ensure_matrix_built()
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Please load data first.")
        
        # Lấy các items mà user chưa rate
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Dự đoán rating cho các items chưa rate
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict_rating_item_based(user_id, item_id, method, k)
            predictions.append((item_id, pred_rating))
        
        # Sắp xếp theo rating giảm dần và lấy top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_items]
    
    def get_similarity_info(
        self,
        entity_id: int,
        entity_type: str = 'user',
        method: str = 'pearson',
        top_k: int = 10
    ) -> str:
        """
        Lấy thông tin về các entities tương tự
        
        Args:
            entity_id: ID của user hoặc item
            entity_type: 'user' hoặc 'item'
            method: 'pearson' hoặc 'cosine'
            top_k: Số entities tương tự nhất
            
        Returns:
            String mô tả các entities tương tự
        """
        if entity_type == 'user':
            if method == 'pearson':
                similarity_matrix = self.compute_user_similarity_pearson()
            else:
                similarity_matrix = self.compute_user_similarity_cosine()
            
            if entity_id not in similarity_matrix.index:
                return f"User {entity_id} not found in similarity matrix"
            
            similarities = similarity_matrix.loc[entity_id].sort_values(ascending=False)
            # Bỏ chính nó (similarity = 1.0)
            similarities = similarities[similarities.index != entity_id]
            top_similar = similarities.head(top_k)
            
            result = f"Top {top_k} users tương tự với user {entity_id} (method: {method}):\n"
            for user_id, sim in top_similar.items():
                result += f"  - User {user_id}: similarity = {sim:.4f}\n"
            
        elif entity_type == 'item':
            if method == 'pearson':
                similarity_matrix = self.compute_item_similarity_pearson()
            else:
                similarity_matrix = self.compute_item_similarity_cosine()
            
            if entity_id not in similarity_matrix.index:
                return f"Item {entity_id} not found in similarity matrix"
            
            similarities = similarity_matrix.loc[entity_id].sort_values(ascending=False)
            # Bỏ chính nó (similarity = 1.0)
            similarities = similarities[similarities.index != entity_id]
            top_similar = similarities.head(top_k)
            
            result = f"Top {top_k} items tương tự với item {entity_id} (method: {method}):\n"
            for item_id, sim in top_similar.items():
                result += f"  - Item {item_id}: similarity = {sim:.4f}\n"
        else:
            return f"Unknown entity_type: {entity_type}. Use 'user' or 'item'"
        
        return result

