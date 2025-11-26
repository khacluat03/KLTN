import pandas as pd

from macrec.tools.base import Tool

class InfoDatabase(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        user_info_path = self.config.get('user_info', None)
        item_info_path = self.config.get('item_info', None)
        if user_info_path is not None:
            self._user_info = pd.read_csv(user_info_path, sep=',')
            assert 'user_id' in self._user_info.columns, 'user_id column not found in user_info.'
        if item_info_path is not None:
            self._item_info = pd.read_csv(item_info_path, sep=',')
            assert 'item_id' in self._item_info.columns, 'item_id column not found in item_info.'

    def reset(self, *args, **kwargs) -> None:
        pass

    def user_info(self, user_id: int) -> str:
        if not hasattr(self, '_user_info'):
            return 'User info database not available.'
        info = self._user_info[self._user_info['user_id'] == user_id]
        if info.empty:
            return f'User {user_id} not found in user info database.'
        assert len(info) == 1, f'Multiple entries found for user {user_id}.'
        if 'user_profile' in self._user_info.columns:
            return info['user_profile'].values[0].replace('\n', '; ')
        else:
            columns = self._user_info.columns
            columns = columns.drop('user_id')
            profile = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'User {user_id} Profile:\n{profile}'

    def item_info(self, item_id: int) -> str:
        if not hasattr(self, '_item_info'):
            return 'Item info database not available.'
        info = self._item_info[self._item_info['item_id'] == item_id]
        if info.empty:
            return f'Item {item_id} not found in item info database.'
        assert len(info) == 1, f'Multiple entries found for item {item_id}.'
        if 'item_attributes' in self._item_info.columns:
            return info['item_attributes'].values[0].replace('\n', '; ')
        else:
            columns = self._item_info.columns
            columns = columns.drop('item_id')
            attributes = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'Item {item_id} Attributes:\n{attributes}'

    def filter_candidates_by_genres(self, preferred_genres: list[str], limit: int = 100) -> list[int]:
        """
        Filter candidate items dựa trên preferred genres

        Args:
            preferred_genres: List of preferred genres (e.g., ['Drama', 'Comedy'])
            limit: Maximum number of candidates to return

        Returns:
            List of item_ids that match preferred genres
        """
        if not hasattr(self, '_item_info'):
            return []

        # Assume genres column exists
        if 'Genres' not in self._item_info.columns:
            # Return random sample if no genres info
            return self._item_info['item_id'].sample(min(limit, len(self._item_info))).tolist()

        candidates = []
        for _, row in self._item_info.iterrows():
            item_genres = str(row['Genres']).split('|')
            # Check if any preferred genre matches
            if any(genre.strip() in preferred_genres for genre in item_genres):
                candidates.append(int(row['item_id']))
                if len(candidates) >= limit:
                    break

        return candidates

    def filter_candidates_by_popularity(self, min_ratings: int = 10, limit: int = 100) -> list[int]:
        """
        Filter candidate items dựa trên popularity (số ratings)

        Args:
            min_ratings: Minimum number of ratings required
            limit: Maximum number of candidates to return

        Returns:
            List of popular item_ids
        """
        if not hasattr(self, '_item_info'):
            return []

        # Need interaction data to count ratings per item
        # For now, return random popular items (can be enhanced with actual rating counts)
        if len(self._item_info) > limit:
            return self._item_info['item_id'].sample(limit).tolist()
        return self._item_info['item_id'].tolist()

    def filter_candidates_content_based(self, user_history: list[int], limit: int = 100,
                                      similarity_threshold: float = 0.3) -> list[int]:
        """
        Content-based filtering: Find items similar to user's historical items

        Args:
            user_history: List of item_ids user has interacted with
            limit: Maximum number of candidates to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of candidate item_ids based on content similarity
        """
        if not hasattr(self, '_item_info') or not user_history:
            return []

        candidates = []
        user_genres = set()

        # Collect all genres from user's history
        for item_id in user_history:
            item_info = self._item_info[self._item_info['item_id'] == item_id]
            if not item_info.empty:
                genres = str(item_info['Genres'].iloc[0]).split('|')
                user_genres.update(g.strip() for g in genres)

        # Find items that share genres with user's history
        for _, row in self._item_info.iterrows():
            item_id = int(row['item_id'])
            if item_id in user_history:  # Skip items user already rated
                continue

            item_genres = set(g.strip() for g in str(row['Genres']).split('|'))
            # Calculate Jaccard similarity
            intersection = len(user_genres.intersection(item_genres))
            union = len(user_genres.union(item_genres))
            if union > 0:
                similarity = intersection / union
                if similarity >= similarity_threshold:
                    candidates.append(item_id)
                    if len(candidates) >= limit:
                        break

        return candidates

    def filter_candidates_hybrid(self, user_id: int, user_history: list[int] = None,
                               preferred_genres: list[str] = None, limit: int = 100) -> list[int]:
        """
        Hybrid filtering combining multiple approaches

        Args:
            user_id: User ID for personalization
            user_history: User's interaction history
            preferred_genres: User's preferred genres
            limit: Maximum candidates to return

        Returns:
            Ranked list of candidate items
        """
        candidates = set()

        # 1. Genre-based filtering
        if preferred_genres:
            genre_candidates = set(self.filter_candidates_by_genres(preferred_genres, limit * 2))
            candidates.update(genre_candidates)

        # 2. Content-based filtering
        if user_history:
            content_candidates = set(self.filter_candidates_content_based(user_history, limit * 2))
            candidates.update(content_candidates)

        # 3. Popularity filtering
        popular_candidates = set(self.filter_candidates_by_popularity(limit=limit * 2))
        candidates.update(popular_candidates)

        # 4. Demographic filtering (if user info available)
        if hasattr(self, '_user_info'):
            user_info = self._user_info[self._user_info['user_id'] == user_id]
            if not user_info.empty:
                # Could add demographic-based filtering here
                pass

        # Convert to list and limit
        candidate_list = list(candidates)
        return candidate_list[:limit] if len(candidate_list) > limit else candidate_list

    def filter_candidates_ml_based(self, user_id: int, user_history: list[int] = None,
                                 limit: int = 100) -> list[int]:
        """
        ML-based filtering using collaborative signals and pattern recognition

        Args:
            user_id: User ID
            user_history: User's rated items
            limit: Maximum candidates to return

        Returns:
            ML-predicted candidate items
        """
        if not user_history:
            return self.filter_candidates_by_popularity(limit=limit)

        candidates = []

        # Simple ML approach: Pattern-based recommendations
        # 1. Find users with similar rating patterns
        # 2. Recommend items that similar users liked but current user hasn't seen

        # This is a simplified version - in production, you could use:
        # - Matrix Factorization (SVD, ALS)
        # - Neural Collaborative Filtering (NCF)
        # - Deep Learning models (Autoencoders, Transformer-based)
        # - Graph Neural Networks (GNN)

        # For now, combine content-based with popularity
        content_candidates = self.filter_candidates_content_based(user_history, limit=limit//2)
        popular_candidates = self.filter_candidates_by_popularity(limit=limit//2)

        # Combine and deduplicate
        all_candidates = list(set(content_candidates + popular_candidates))

        # Simple scoring: prefer content-based matches
        scored_candidates = []
        for item_id in all_candidates:
            score = 1.0 if item_id in content_candidates else 0.5
            scored_candidates.append((item_id, score))

        # Sort by score and return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [item_id for item_id, _ in scored_candidates[:limit]]

        return candidates