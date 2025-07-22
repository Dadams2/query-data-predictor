import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
import random
from datetime import datetime, timedelta
import uuid
import argparse
import time

class SyntheticDataGenerator:
    """
    Generates synthetic data with configurable:
    - Number of records
    - Number and types of features (numeric, categorical)
    - Distributions of features
    - Sparsity (percentage of NULL values)
    - Hierarchical relationships (e.g., geography, product categories)
    """
    
    def __init__(self, config):
        """
        Initialize with configuration parameters
        """
        self.config = config
        self.data = None
        
        # Validate config
        self._validate_config()
        
        # Initialize random seed for reproducibility if provided
        if 'random_seed' in self.config:
            np.random.seed(self.config['random_seed'])
            random.seed(self.config['random_seed'])
    
    def _validate_config(self):
        """Validate the configuration parameters"""
        required_keys = ['num_records', 'numeric_features', 'categorical_features', 'datetime_features']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def generate_data(self):
        """
        Generate synthetic data based on configuration
        """
        num_records = self.config['num_records']
        data = {}
        
        # Generate primary key
        data['id'] = list(range(1, num_records + 1))
        
        # Generate numeric features
        for feature_name, feature_config in self.config['numeric_features'].items():
            data[feature_name] = self._generate_numeric_feature(feature_config, num_records)
        
        # Generate categorical features
        for feature_name, feature_config in self.config['categorical_features'].items():
            data[feature_name] = self._generate_categorical_feature(feature_config, num_records)
        
        # Generate datetime features
        for feature_name, feature_config in self.config['datetime_features'].items():
            data[feature_name] = self._generate_datetime_feature(feature_config, num_records)
        
        # Generate hierarchical features if specified
        if 'hierarchical_features' in self.config:
            for hierarchy_name, hierarchy_config in self.config['hierarchical_features'].items():
                hierarchical_data = self._generate_hierarchical_features(hierarchy_config, num_records)
                for key, values in hierarchical_data.items():
                    data[key] = values
        
        # Apply sparsity
        if 'sparsity' in self.config:
            data = self._apply_sparsity(data)
        
        # Apply correlations if specified
        if 'correlations' in self.config:
            for corr_config in self.config['correlations']:
                feature1 = corr_config['feature1']
                feature2 = corr_config['feature2']
                level = corr_config['level']
                self._add_correlation(data, feature1, feature2, level)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def _generate_numeric_feature(self, feature_config, num_records):
        """Generate numeric feature with specified distribution"""
        distribution = feature_config.get('distribution', 'normal')
        params = feature_config.get('params', {})
        
        if distribution == 'normal':
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return np.random.normal(mean, std, num_records)
        
        elif distribution == 'uniform':
            low = params.get('low', 0)
            high = params.get('high', 1)
            return np.random.uniform(low, high, num_records)
        
        elif distribution == 'exponential':
            scale = params.get('scale', 1)
            return np.random.exponential(scale, num_records)
        
        elif distribution == 'poisson':
            lam = params.get('lambda', 1)
            return np.random.poisson(lam, num_records)
        
        elif distribution == 'binomial':
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return np.random.binomial(n, p, num_records)
        
        elif distribution == 'lognormal':
            mean = params.get('mean', 0)
            sigma = params.get('sigma', 1)
            return np.random.lognormal(mean, sigma, num_records)
        
        elif distribution == 'power_law':
            a = params.get('a', 2)  # Power law exponent
            return np.random.power(a, num_records) * params.get('scale', 100)
        
        elif distribution == 'bimodal':
            means = params.get('means', [0, 10])
            stds = params.get('stds', [1, 1])
            weights = params.get('weights', [0.5, 0.5])
            
            # Generate samples from two normal distributions
            n1 = int(num_records * weights[0])
            n2 = num_records - n1
            samples1 = np.random.normal(means[0], stds[0], n1)
            samples2 = np.random.normal(means[1], stds[1], n2)
            
            # Combine and shuffle
            combined = np.concatenate([samples1, samples2])
            np.random.shuffle(combined)
            return combined
        
        elif distribution == 'custom':
            if 'custom_function' in params:
                return params['custom_function'](num_records)
            else:
                raise ValueError("Custom distribution requires a custom_function parameter")
        
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def _generate_categorical_feature(self, feature_config, num_records):
        """Generate categorical feature with specified distribution"""
        categories = feature_config.get('categories', ['A', 'B', 'C'])
        distribution = feature_config.get('distribution', 'uniform')
        params = feature_config.get('params', {})
        
        if distribution == 'uniform':
            return np.random.choice(categories, num_records)
        
        elif distribution == 'weighted':
            weights = params.get('weights')
            if not weights or len(weights) != len(categories):
                weights = np.ones(len(categories)) / len(categories)
            return np.random.choice(categories, num_records, p=weights)
        
        elif distribution == 'zipf':
            # Zipf distribution for categories (power law)
            a = params.get('a', 1.5)  # Zipf parameter
            weights = 1 / np.arange(1, len(categories) + 1) ** a
            weights = weights / weights.sum()  # Normalize
            return np.random.choice(categories, num_records, p=weights)
        
        else:
            raise ValueError(f"Unsupported distribution for categorical data: {distribution}")
    
    def _generate_datetime_feature(self, feature_config, num_records):
        """Generate datetime feature with specified distribution"""
        start_date = feature_config.get('start_date', '2020-01-01')
        end_date = feature_config.get('end_date', '2023-12-31')
        distribution = feature_config.get('distribution', 'uniform')
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        
        if distribution == 'uniform':
            timestamps = np.random.uniform(start_timestamp, end_timestamp, num_records)
        elif distribution == 'normal':
            # Normal distribution centered at the middle of the range
            mean = (start_timestamp + end_timestamp) / 2
            std = (end_timestamp - start_timestamp) / 6  # 99.7% within range
            timestamps = np.random.normal(mean, std, num_records)
            # Clip to ensure within range
            timestamps = np.clip(timestamps, start_timestamp, end_timestamp)
        elif distribution == 'increasing':
            # More recent dates are more frequent
            alpha = feature_config.get('params', {}).get('alpha', 2)
            x = np.random.power(alpha, num_records)
            timestamps = start_timestamp + x * (end_timestamp - start_timestamp)
        elif distribution == 'decreasing':
            # Older dates are more frequent
            alpha = feature_config.get('params', {}).get('alpha', 2)
            x = 1 - np.random.power(alpha, num_records)
            timestamps = start_timestamp + x * (end_timestamp - start_timestamp)
        elif distribution == 'seasonal':
            # Simple seasonal pattern (e.g., yearly)
            period = feature_config.get('params', {}).get('period', 365.25 * 24 * 3600)  # Default: 1 year in seconds
            amplitude = feature_config.get('params', {}).get('amplitude', 0.5)
            
            # Base uniform distribution
            base = np.random.uniform(0, 1, num_records)
            # Add seasonality
            t = np.linspace(0, 1, num_records)
            seasonal = amplitude * np.sin(2 * np.pi * t)
            combined = base + seasonal
            # Normalize to [0, 1]
            combined = (combined - combined.min()) / (combined.max() - combined.min())
            
            timestamps = start_timestamp + combined * (end_timestamp - start_timestamp)
        else:
            raise ValueError(f"Unsupported distribution for datetime: {distribution}")
        
        return [datetime.fromtimestamp(ts) for ts in timestamps]
    
    def _generate_hierarchical_features(self, hierarchy_config, num_records):
        """Generate hierarchical features like geography or categories"""
        hierarchy_type = hierarchy_config.get('type', 'geography')
        levels = hierarchy_config.get('levels', [])
        
        result = {}
        
        if hierarchy_type == 'geography':
            # Generate geographical hierarchy
            country_list = hierarchy_config.get('countries', ['USA', 'Canada', 'UK', 'Germany', 'France'])
            
            # Generate countries first
            countries = np.random.choice(country_list, num_records, p=hierarchy_config.get('country_weights', None))
            result['country'] = countries
            
            # For each country, generate states/regions
            country_to_regions = hierarchy_config.get('country_to_regions', {
                'USA': ['California', 'Texas', 'New York', 'Florida', 'Illinois'],
                'Canada': ['Ontario', 'Quebec', 'British Columbia', 'Alberta'],
                'UK': ['England', 'Scotland', 'Wales', 'Northern Ireland'],
                'Germany': ['Bavaria', 'Berlin', 'Hesse', 'Saxony'],
                'France': ['Île-de-France', 'Provence', 'Normandy', 'Brittany']
            })

            region_to_cities = hierarchy_config.get('region_to_cities', {
                'California': ['Los Angeles', 'San Francisco', 'San Diego'],
                'Texas': ['Houston', 'Austin', 'Dallas'],
                'New York': ['New York City', 'Buffalo', 'Albany'],
                'Florida': ['Miami', 'Orlando', 'Tampa'],
                'Illinois': ['Chicago', 'Springfield', 'Peoria'],
                'Ontario': ['Toronto', 'Ottawa', 'Mississauga'],
                'Quebec': ['Montreal', 'Quebec City', 'Laval'],
                'British Columbia': ['Vancouver', 'Victoria', 'Kelowna'],
                'Alberta': ['Calgary', 'Edmonton', 'Red Deer'],
                'England': ['London', 'Manchester', 'Birmingham'],
                'Scotland': ['Edinburgh', 'Glasgow', 'Aberdeen'],
                'Wales': ['Cardiff', 'Swansea', 'Newport'],
                'Northern Ireland': ['Belfast', 'Londonderry', 'Lisburn'],
                'Bavaria': ['Munich', 'Nuremberg', 'Augsburg'],
                'Berlin': ['Berlin'],
                'Hesse': ['Frankfurt', 'Wiesbaden', 'Kassel'],
                'Saxony': ['Dresden', 'Leipzig', 'Chemnitz'],
                'Île-de-France': ['Paris', 'Versailles', 'Boulogne-Billancourt'],
                'Provence': ['Marseille', 'Nice', 'Aix-en-Provence'],
                'Normandy': ['Rouen', 'Caen', 'Le Havre'],
                'Brittany': ['Rennes', 'Brest', 'Quimper']
            })

            
            # Generate regions based on countries
            regions = []
            for country in countries:
                if country in country_to_regions:
                    regions.append(np.random.choice(country_to_regions[country]))
                else:
                    regions.append(None)
            result['region'] = regions
            
            # Generate cities based on regions
            cities = []
            for region in regions:
                if region in region_to_cities:
                    cities.append(np.random.choice(region_to_cities[region]))
                else:
                    cities.append(None)
            result['city'] = cities
            
        elif hierarchy_type == 'product':
            # Generate product category hierarchy
            categories = hierarchy_config.get('categories', ['Electronics', 'Clothing', 'Home', 'Food'])
            
            # Generate top-level categories
            top_categories = np.random.choice(categories, num_records, p=hierarchy_config.get('category_weights', None))
            result['category'] = top_categories
            
            # For each category, generate subcategories
            category_to_subcategories = hierarchy_config.get('category_to_subcategories', {
                'Electronics': ['Computers', 'Phones', 'TVs', 'Audio'],
                'Clothing': ['Men', 'Women', 'Children', 'Accessories'],
                'Home': ['Furniture', 'Kitchen', 'Decor', 'Garden'],
                'Food': ['Fresh', 'Frozen', 'Canned', 'Snacks']
            })

            subcategory_to_products = hierarchy_config.get('subcategory_to_products', {
                'Computers': ['Laptop', 'Desktop', 'Tablet', 'Monitor', 'Printer'],
                'Phones': ['Smartphone', 'Feature Phone', 'Charger', 'Headphones'],
                'TVs': ['LED TV', 'OLED TV', 'Smart TV', 'Projector'],
                'Audio': ['Speakers', 'Headphones', 'Earbuds', 'Soundbar'],
                'Men': ['Shirts', 'Pants', 'Jackets', 'Shoes'],
                'Women': ['Dresses', 'Tops', 'Skirts', 'Heels'],
                'Children': ['T-shirts', 'Shorts', 'Sneakers', 'Hats'],
                'Accessories': ['Bags', 'Belts', 'Hats', 'Watches'],
                'Furniture': ['Sofa', 'Bed', 'Chair', 'Table'],
                'Kitchen': ['Cookware', 'Cutlery', 'Dinnerware', 'Appliances'],
                'Decor': ['Lamps', 'Wall Art', 'Rugs', 'Curtains'],
                'Garden': ['Plants', 'Tools', 'Furniture', 'Lights'],
                'Fresh': ['Fruits', 'Vegetables', 'Meat', 'Dairy'],
                'Frozen': ['Ice Cream', 'Frozen Meals', 'Frozen Vegetables', 'Frozen Meat'],
                'Canned': ['Soup', 'Beans', 'Tuna', 'Tomatoes'],
                'Snacks': ['Chips', 'Chocolate', 'Cookies', 'Nuts']
            })
            
            # Generate subcategories based on categories
            subcategories = []
            for category in top_categories:
                if category in category_to_subcategories:
                    subcategories.append(np.random.choice(category_to_subcategories[category]))
                else:
                    subcategories.append(None)
            result['subcategory'] = subcategories
            
            # Generate product types based on subcategories
            products = []
            for subcategory in subcategories:
                if subcategory in subcategory_to_products:
                    products.append(np.random.choice(subcategory_to_products[subcategory]))
                else:
                    products.append(None)
            result['product_type'] = products
        
        return result
    
    def _apply_sparsity(self, data):
        """Apply sparsity to the data by replacing values with appropriate NA values"""
        import numpy as np
        
        sparsity_config = self.config['sparsity']
        global_sparsity = sparsity_config.get('global', 0.0)
        feature_sparsity = sparsity_config.get('features', {})
        
        # Don't apply sparsity to the ID column
        features_to_sparse = [col for col in data.keys() if col != 'id']
        
        # Apply global sparsity
        if global_sparsity > 0:
            for feature in features_to_sparse:
                indices = np.random.choice(len(data[feature]), 
                                        size=int(len(data[feature]) * global_sparsity),
                                        replace=False)
                
                # Convert integer arrays to float before adding NaN
                if hasattr(data[feature], 'dtype') and np.issubdtype(data[feature].dtype, np.integer):
                    data[feature] = data[feature].astype(float)
                
                # For lists containing numeric values, convert to numpy array first
                if isinstance(data[feature], list) and data[feature] and isinstance(data[feature][0], (int, float)):
                    data[feature] = np.array(data[feature], dtype=float)
                    
                # Apply NaN or None based on the data type
                for idx in indices:
                    if isinstance(data[feature], np.ndarray):
                        data[feature][idx] = np.nan
                    else:
                        data[feature][idx] = None
        
        # Apply feature-specific sparsity
        for feature, sparsity_level in feature_sparsity.items():
            if feature in data and sparsity_level > 0:
                indices = np.random.choice(len(data[feature]), 
                                        size=int(len(data[feature]) * sparsity_level),
                                        replace=False)
                
                # Convert integer arrays to float before adding NaN
                if hasattr(data[feature], 'dtype') and np.issubdtype(data[feature].dtype, np.integer):
                    data[feature] = data[feature].astype(float)
                    
                # For lists containing numeric values, convert to numpy array first
                if isinstance(data[feature], list) and data[feature] and isinstance(data[feature][0], (int, float)):
                    data[feature] = np.array(data[feature], dtype=float)
                    
                # Apply NaN or None based on the data type
                for idx in indices:
                    if isinstance(data[feature], np.ndarray):
                        data[feature][idx] = np.nan
                    else:
                        data[feature][idx] = None
        
        return data
    
    def _add_correlation(self, data, feature1, feature2, correlation_level):
        """Add correlation between two numeric features"""
        # Check if both features exist and are numeric
        if feature1 not in data or feature2 not in data:
            raise ValueError(f"Features {feature1} and/or {feature2} not found in data")
        
        # Convert data to numpy arrays for manipulation
        x = np.array(data[feature1], dtype=float)
        y = np.array(data[feature2], dtype=float)
        
        # Handle NaN values if any
        valid_indices = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        if len(x_valid) == 0:
            raise ValueError("No valid data points for correlation")
        
        # Calculate current correlation
        current_corr = np.corrcoef(x_valid, y_valid)[0, 1]
        
        # If current correlation is close to target, return
        if abs(current_corr - correlation_level) < 0.05:
            return
        
        # Apply Cholesky decomposition to achieve desired correlation
        # First, standardize the data
        x_std = (x_valid - np.mean(x_valid)) / np.std(x_valid)
        y_std = (y_valid - np.mean(y_valid)) / np.std(y_valid)
        
        # Create correlation matrix
        corr_matrix = np.array([[1.0, correlation_level], [correlation_level, 1.0]])
        
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate correlated data
        correlated = np.vstack((x_std, np.random.normal(0, 1, len(x_std))))
        correlated = np.dot(L, correlated)
        
        # Replace the second feature with the correlated version
        new_y = correlated[1] * np.std(y_valid) + np.mean(y_valid)
        
        # Put back into the data dictionary
        y[valid_indices] = new_y
        data[feature2] = y.tolist()
        
        # Verify achieved correlation
        achieved_corr = np.corrcoef(x_valid, new_y)[0, 1]
        
        return achieved_corr