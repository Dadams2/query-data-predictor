import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

class Discretizer:
    def __init__(self, method='equal_width', bins=5, save_path=None, load_path=None):
        """
        A class to discretize numerical columns in a DataFrame.
        
        Parameters:
        method (str): The discretization method ('equal_width', 'equal_freq', or 'kmeans').
        bins (int): The number of bins to create.
        save_path (str): Path to save the binning parameters.
        load_path (str): Path to load existing binning parameters.
        """
        self.method = method
        self.bins = bins
        self.save_path = save_path
        self.discretization_params = {}
        
        if load_path:
            self.load_params(load_path)
    
    def load_params(self, path):
        with open(path, 'rb') as f:
            self.discretization_params = pickle.load(f)
    
    def save_params(self):
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.discretization_params, f)
    
    def discretize_dataframe(self, df):
        """
        Discretizes all float columns in a DataFrame and updates binning parameters iteratively.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: DataFrame with discretized values.
        """
        # Get float columns
        float_columns = df.select_dtypes(include=[np.float64]).columns
        
        for column in float_columns:
            if column in self.discretization_params:
                bins_edges = self.discretization_params[column]
                df[f"{column}_bin"] = np.digitize(df[column], bins_edges, right=False)
            else:
                if self.method == 'equal_width':
                    min_val = df[column].min()
                    max_val = df[column].max()
                    bins_edges = np.linspace(min_val, max_val, self.bins+1)
                    
                    # Digitize using numpy
                    bin_values = np.digitize(df[column], bins_edges, right=False)
                    
                    # Fix values outside the range
                    bin_values = np.where(df[column] < bins_edges[0], 0, bin_values)
                    bin_values = np.where(df[column] > bins_edges[-1], len(bins_edges) - 1, bin_values)
                    
                    df[f"{column}_bin"] = bin_values
                elif self.method == 'equal_freq':
                    # Calculate quantiles for bin edges
                    quantiles = [i/self.bins for i in range(self.bins+1)]
                    bins_edges = [float(df[column].quantile(q)) for q in quantiles]
                    
                    # Remove duplicates while preserving order
                    bins_edges = sorted(set(bins_edges))
                    
                    # Digitize using numpy
                    df[f"{column}_bin"] = np.digitize(df[column], bins_edges, right=False)
                elif self.method == 'kmeans':
                    kmeans = KMeans(n_clusters=self.bins, random_state=42, n_init=10)
                    # Get data as numpy array for kmeans
                    data = df[column].values.reshape(-1, 1)
                    bin_values = kmeans.fit_predict(data)
                    bins_edges = kmeans.cluster_centers_.flatten()
                    
                    df[f"{column}_bin"] = bin_values
                else:
                    raise ValueError("Invalid method. Choose from 'equal_width', 'equal_freq', or 'kmeans'.")
                
                self.discretization_params[column] = bins_edges
            
            # Drop the original column
            df.drop(column, axis=1, inplace=True)
        
        self.save_params()
        return df

    def prepend_column_names(self, df):
        """
        Prepend the column name to all the values in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: DataFrame with column names prepended to the values.
        """
        for column in df.columns:
            df[column] = df[column].apply(lambda x: f"{column}_{x}")
        return df
