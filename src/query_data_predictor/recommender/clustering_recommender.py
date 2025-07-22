"""
Clustering recommender that clusters tuples and returns one tuple from each centroid.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin_min
import warnings

from .base_recommender import BaseRecommender


class ClusteringRecommender(BaseRecommender):
    """
    A clustering recommender that clusters the current query results and returns
    one representative tuple from each cluster centroid.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the clustering recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples by clustering the current results and selecting representative tuples.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
                - n_clusters: Override the number of clusters from config
            
        Returns:
            DataFrame with recommended tuples (one per cluster)
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Get clustering configuration
        clustering_config = self.config.get('clustering', {})
        n_clusters = kwargs.get('n_clusters', clustering_config.get('n_clusters'))
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            target_size = self._determine_output_size(len(current_results))
            n_clusters = min(target_size, len(current_results))
        
        # Ensure we don't have more clusters than data points
        n_clusters = min(n_clusters, len(current_results))
        
        if n_clusters <= 1:
            # If only one cluster, return the first row
            return current_results.head(1)
        
        # Prepare data for clustering
        try:
            encoded_data = self._prepare_data_for_clustering(current_results)
            
            if encoded_data.empty or encoded_data.shape[1] == 0:
                # If we can't encode the data, fall back to random selection
                return self._fallback_selection(current_results, n_clusters)
            
            # Perform clustering
            clusters, kmeans_model = self._perform_clustering(encoded_data, n_clusters)
            
            # Select representative tuples from each cluster
            recommendations = self._select_representatives(
                current_results, encoded_data, clusters, kmeans_model, n_clusters
            )
            
            # Apply final output limiting
            return self._limit_output(recommendations)
            
        except Exception as e:
            # If clustering fails, fall back to random selection
            warnings.warn(f"Clustering failed: {str(e)}. Falling back to random selection.")
            return self._fallback_selection(current_results, n_clusters)
    
    def _prepare_data_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for clustering by encoding categorical variables and scaling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame ready for clustering
        """
        # Create a copy to avoid modifying the original
        prepared_df = df.copy()
        
        # Handle different column types
        encoded_columns = []
        
        for col in prepared_df.columns:
            if prepared_df[col].dtype == 'object':
                # Categorical column - use label encoding
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values
                non_null_mask = prepared_df[col].notna()
                if non_null_mask.any():
                    prepared_df.loc[non_null_mask, col] = self.label_encoders[col].fit_transform(
                        prepared_df.loc[non_null_mask, col].astype(str)
                    )
                    encoded_columns.append(col)
                
            elif np.issubdtype(prepared_df[col].dtype, np.number):
                # Numerical column - keep as is, but handle missing values
                if prepared_df[col].notna().any():
                    encoded_columns.append(col)
        
        # Keep only successfully encoded columns
        if encoded_columns:
            prepared_df = prepared_df[encoded_columns]
            
            # Fill remaining missing values with column means/modes
            for col in prepared_df.columns:
                if prepared_df[col].notna().any():
                    if np.issubdtype(prepared_df[col].dtype, np.number):
                        prepared_df[col].fillna(prepared_df[col].mean(), inplace=True)
                    else:
                        prepared_df[col].fillna(prepared_df[col].mode().iloc[0] if not prepared_df[col].mode().empty else 0, inplace=True)
            
            # Scale the data
            try:
                scaled_data = self.scaler.fit_transform(prepared_df)
                return pd.DataFrame(scaled_data, columns=prepared_df.columns, index=prepared_df.index)
            except Exception:
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _perform_clustering(self, data: pd.DataFrame, n_clusters: int) -> tuple[np.ndarray, KMeans]:
        """
        Perform K-means clustering on the prepared data.
        
        Args:
            data: Prepared DataFrame for clustering
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels array, fitted KMeans model)
        """
        clustering_config = self.config.get('clustering', {})
        
        # Suppress convergence warnings for small datasets
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=clustering_config.get('random_state', 42),
                max_iter=clustering_config.get('max_iter', 300),
                n_init=clustering_config.get('n_init', 10)
            )
            
            clusters = kmeans.fit_predict(data)
            return clusters, kmeans
    
    def _select_representatives(self, original_df: pd.DataFrame, encoded_data: pd.DataFrame, 
                               clusters: np.ndarray, kmeans_model: KMeans, n_clusters: int) -> pd.DataFrame:
        """
        Select representative tuples from each cluster by finding the tuple closest to the centroid.
        
        Args:
            original_df: Original DataFrame
            encoded_data: The encoded data used for clustering
            clusters: Array of cluster labels
            kmeans_model: Fitted KMeans model with centroids
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with representative tuples (closest to centroids)
        """
        representatives = []
        
        for cluster_id in range(n_clusters):
            # Get indices of tuples in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                if len(cluster_indices) == 1:
                    # Only one tuple in cluster, select it
                    representative_idx = cluster_indices[0]
                else:
                    # Find the tuple closest to the cluster centroid
                    cluster_data = encoded_data.iloc[cluster_indices].values
                    centroid = kmeans_model.cluster_centers_[cluster_id].reshape(1, -1)
                    
                    # Find the closest point to the centroid
                    closest_idx_within_cluster, _ = pairwise_distances_argmin_min(
                        centroid, cluster_data, metric='euclidean'
                    )
                    representative_idx = cluster_indices[closest_idx_within_cluster[0]]
                
                representatives.append(original_df.iloc[representative_idx])
        
        if representatives:
            return pd.DataFrame(representatives)
        else:
            return pd.DataFrame()
    
    def _fallback_selection(self, df: pd.DataFrame, n_tuples: int) -> pd.DataFrame:
        """
        Fallback method when clustering fails - returns random selection.
        
        Args:
            df: Input DataFrame
            n_tuples: Number of tuples to select
            
        Returns:
            DataFrame with randomly selected tuples
        """
        n_tuples = min(n_tuples, len(df))
        return df.sample(n=n_tuples, random_state=42)
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "ClusteringRecommender"
