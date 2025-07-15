"""
Hierarchical recommender with multi-level recommendations (fast coarse-grained + detailed fine-grained).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class HierarchicalRecommender(BaseRecommender):
    """
    High-performance recommender using hierarchical approach.
    Fast coarse-grained filtering followed by detailed analysis on promising candidates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hierarchical recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Configuration
        hier_config = self.config.get('hierarchical', {})
        self.coarse_method = hier_config.get('coarse_method', 'clustering')  # 'clustering', 'sampling', 'frequency'
        self.fine_method = hier_config.get('fine_method', 'similarity')  # 'similarity', 'variance', 'entropy'
        self.coarse_ratio = hier_config.get('coarse_ratio', 0.3)  # Keep 30% after coarse filtering
        self.min_coarse_candidates = hier_config.get('min_coarse_candidates', 50)
        self.max_coarse_candidates = hier_config.get('max_coarse_candidates', 500)
        self.n_clusters = hier_config.get('n_clusters', 10)
        
        # Internal components
        self.scaler = StandardScaler()
        self.kmeans = None
        self._coarse_cache = {}
        self._fine_cache = {}
        self._last_processed_hash = None
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using hierarchical approach.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # For small datasets, skip hierarchy
        if len(current_results) <= self.min_coarse_candidates:
            return self._apply_fine_grained_analysis(current_results)
        
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(current_results).sum()
        
        # Check cache
        if df_hash == self._last_processed_hash and df_hash in self._fine_cache:
            logger.debug("Using cached hierarchical results")
            return self._fine_cache[df_hash]
        
        # Step 1: Coarse-grained filtering
        coarse_candidates = self._apply_coarse_grained_filtering(current_results)
        
        # Step 2: Fine-grained analysis on candidates
        result_df = self._apply_fine_grained_analysis(coarse_candidates)
        
        # Cache results
        self._fine_cache[df_hash] = result_df
        self._last_processed_hash = df_hash
        
        return result_df
    
    def _apply_coarse_grained_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fast coarse-grained filtering to reduce candidate set.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame with promising candidates
        """
        if self.coarse_method == 'clustering':
            return self._clustering_filtering(df)
        elif self.coarse_method == 'sampling':
            return self._sampling_filtering(df)
        elif self.coarse_method == 'frequency':
            return self._frequency_filtering(df)
        else:
            # Default to clustering
            return self._clustering_filtering(df)
    
    def _clustering_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use clustering for coarse-grained filtering."""
        # Encode data for clustering
        encoded_df = self._encode_for_clustering(df)
        
        if encoded_df.empty or len(encoded_df) < 2:
            return df
        
        # Determine number of clusters
        n_clusters = min(self.n_clusters, len(df) // 2)
        if n_clusters < 2:
            return df
        
        # Perform clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(encoded_df)
            
            # Select representatives from each cluster
            representatives = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 0:
                    # Select multiple representatives per cluster based on cluster size
                    cluster_size = len(cluster_indices)
                    n_representatives = max(1, cluster_size // 10)  # 10% of cluster
                    n_representatives = min(n_representatives, 5)  # Max 5 per cluster
                    
                    if n_representatives >= len(cluster_indices):
                        representatives.extend(cluster_indices)
                    else:
                        # Select diverse representatives within cluster
                        cluster_data = encoded_df.iloc[cluster_indices]
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        
                        # Compute distances to cluster center
                        distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                        
                        # Select closest and farthest points for diversity
                        closest_idx = np.argmin(distances)
                        representatives.append(cluster_indices[closest_idx])
                        
                        if n_representatives > 1:
                            farthest_idx = np.argmax(distances)
                            representatives.append(cluster_indices[farthest_idx])
                        
                        # Add random representatives if needed
                        remaining = n_representatives - 2
                        if remaining > 0:
                            remaining_indices = np.setdiff1d(cluster_indices, 
                                                            [cluster_indices[closest_idx], 
                                                             cluster_indices[farthest_idx]])
                            if len(remaining_indices) > 0:
                                selected = np.random.choice(remaining_indices, 
                                                           size=min(remaining, len(remaining_indices)), 
                                                           replace=False)
                                representatives.extend(selected)
            
            # Ensure we have enough candidates
            if len(representatives) < self.min_coarse_candidates:
                # Add random additional candidates
                all_indices = set(range(len(df)))
                remaining_indices = list(all_indices - set(representatives))
                additional_needed = min(self.min_coarse_candidates - len(representatives), 
                                      len(remaining_indices))
                if additional_needed > 0:
                    additional = np.random.choice(remaining_indices, size=additional_needed, replace=False)
                    representatives.extend(additional)
            
            # Limit to max candidates
            if len(representatives) > self.max_coarse_candidates:
                representatives = np.random.choice(representatives, size=self.max_coarse_candidates, replace=False)
            
            return df.iloc[representatives].copy()
            
        except Exception as e:
            logger.warning(f"Clustering filtering failed: {e}")
            return self._sampling_filtering(df)
    
    def _sampling_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use sampling for coarse-grained filtering."""
        target_size = max(self.min_coarse_candidates, 
                         min(self.max_coarse_candidates, int(len(df) * self.coarse_ratio)))
        
        if target_size >= len(df):
            return df
        
        # Use stratified sampling if possible
        sample_indices = np.random.choice(len(df), size=target_size, replace=False)
        return df.iloc[sample_indices].copy()
    
    def _frequency_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use frequency-based filtering for coarse-grained selection."""
        # Compute simple frequency scores
        scores = np.zeros(len(df))
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < len(df) * 0.5:
                # Use value counts for categorical-like columns
                value_counts = df[col].value_counts()
                for i, value in enumerate(df[col]):
                    scores[i] += value_counts.get(value, 0)
        
        # Select top candidates by frequency
        target_size = max(self.min_coarse_candidates, 
                         min(self.max_coarse_candidates, int(len(df) * self.coarse_ratio)))
        
        if target_size >= len(df):
            return df
        
        top_indices = np.argsort(scores)[-target_size:]
        return df.iloc[top_indices].copy()
    
    def _apply_fine_grained_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply detailed fine-grained analysis on candidate set.
        
        Args:
            df: Candidate DataFrame
            
        Returns:
            DataFrame with final recommendations
        """
        if self.fine_method == 'similarity':
            return self._similarity_analysis(df)
        elif self.fine_method == 'variance':
            return self._variance_analysis(df)
        elif self.fine_method == 'entropy':
            return self._entropy_analysis(df)
        else:
            # Default to similarity
            return self._similarity_analysis(df)
    
    def _similarity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply similarity-based fine-grained analysis."""
        if df.empty or len(df) < 2:
            return self._limit_output(df)
        
        # Encode data
        encoded_df = self._encode_for_clustering(df)
        
        if encoded_df.empty:
            return self._limit_output(df)
        
        # Compute pairwise similarities
        data_matrix = encoded_df.values
        
        # Standardize
        if len(data_matrix) > 1:
            data_matrix = self.scaler.fit_transform(data_matrix)
        
        # Compute similarity scores (mean similarity to all other points)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(data_matrix)
        
        # Remove self-similarity
        np.fill_diagonal(similarity_matrix, 0)
        
        # Compute mean similarity for each row
        similarity_scores = np.mean(similarity_matrix, axis=1)
        
        # Add scores and sort
        df_scored = df.copy()
        df_scored['fine_score'] = similarity_scores
        sorted_df = df_scored.sort_values('fine_score', ascending=False)
        
        return self._limit_output(sorted_df.drop(columns=['fine_score']))
    
    def _variance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply variance-based fine-grained analysis."""
        if df.empty:
            return self._limit_output(df)
        
        # Compute variance scores
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return self._limit_output(df)
        
        # Normalize data
        normalized_data = self.scaler.fit_transform(numeric_df)
        
        # Compute row-wise variance
        variance_scores = np.var(normalized_data, axis=1)
        
        # Add scores and sort
        df_scored = df.copy()
        df_scored['fine_score'] = variance_scores
        sorted_df = df_scored.sort_values('fine_score', ascending=False)
        
        return self._limit_output(sorted_df.drop(columns=['fine_score']))
    
    def _entropy_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply entropy-based fine-grained analysis."""
        if df.empty:
            return self._limit_output(df)
        
        # Discretize data for entropy computation
        df_discrete = df.copy()
        for col in df_discrete.columns:
            if pd.api.types.is_numeric_dtype(df_discrete[col]):
                try:
                    df_discrete[col] = pd.cut(df_discrete[col], bins=5, labels=False, duplicates='drop')
                except ValueError:
                    df_discrete[col] = 0
        
        # Compute entropy scores
        entropy_scores = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df_discrete.iterrows()):
            entropy = 0.0
            for col in df_discrete.columns:
                value = row[col]
                col_values = df_discrete[col].values
                prob = np.sum(col_values == value) / len(col_values)
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            entropy_scores[i] = entropy
        
        # Add scores and sort
        df_scored = df.copy()
        df_scored['fine_score'] = entropy_scores
        sorted_df = df_scored.sort_values('fine_score', ascending=False)
        
        return self._limit_output(sorted_df.drop(columns=['fine_score']))
    
    def _encode_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode DataFrame for clustering operations."""
        if df.empty:
            return pd.DataFrame()
        
        encoded_df = df.copy()
        
        # Encode categorical variables
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object':
                # Simple label encoding
                unique_values = encoded_df[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                encoded_df[col] = encoded_df[col].map(value_map)
        
        # Handle missing values
        encoded_df = encoded_df.fillna(0)
        
        # Ensure all columns are numeric
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').fillna(0)
        
        return encoded_df
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about hierarchical processing."""
        return {
            'coarse_method': self.coarse_method,
            'fine_method': self.fine_method,
            'coarse_ratio': self.coarse_ratio,
            'min_coarse_candidates': self.min_coarse_candidates,
            'max_coarse_candidates': self.max_coarse_candidates,
            'n_clusters': self.n_clusters,
            'coarse_cache_size': len(self._coarse_cache),
            'fine_cache_size': len(self._fine_cache)
        }
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "HierarchicalRecommender"
