#!/usr/bin/env python3
"""
Synthetic Data Generator CLI

A command-line tool for generating synthetic datasets with configurable features,
distributions, and output formats.
"""

import click
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data_generator.data_generator import SyntheticDataGenerator
from synthetic_data_generator.writer import write_to_postgres


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def create_visualizations(df, output_dir):
    """Create and save data visualizations."""
    logger.info("Creating visualizations...")
    
    # Create output directory for plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Price distribution
    plt.figure(figsize=(10, 6))
    if 'price' in df.columns:
        sns.histplot(df['price'].dropna(), bins=30, kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.savefig(plots_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Quantity distribution
    plt.figure(figsize=(10, 6))
    if 'quantity' in df.columns:
        sns.countplot(data=df, x='quantity')
        plt.title('Order Quantities')
        plt.xlabel('Quantity')
        plt.ylabel('Count')
        plt.savefig(plots_dir / 'quantity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Orders over time (if order_date exists)
    if 'order_date' in df.columns:
        plt.figure(figsize=(12, 6))
        df_temp = df.copy()
        df_temp['month'] = pd.to_datetime(df_temp['order_date']).dt.month
        orders_by_month = df_temp.groupby('month').size()
        sns.lineplot(x=orders_by_month.index, y=orders_by_month.values, marker='o')
        plt.title('Orders by Month (Seasonality)')
        plt.xlabel('Month')
        plt.ylabel('Number of Orders')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'orders_by_month.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Correlation heatmap for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Categorical features distribution
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols[:4]:  # Limit to first 4 categorical columns
        if col != 'id':
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'{col.replace("_", " ").title()} Distribution')
            plt.xlabel('Count')
            plt.ylabel(col.replace("_", " ").title())
            plt.tight_layout()
            plt.savefig(plots_dir / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Summary dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price histogram
    if 'price' in df.columns:
        sns.histplot(df['price'].dropna(), ax=axes[0, 0], kde=True)
        axes[0, 0].set_title('Price Distribution')
    
    # Quantity bar plot
    if 'quantity' in df.columns:
        quantity_counts = df['quantity'].value_counts().sort_index()
        axes[0, 1].bar(quantity_counts.index, quantity_counts.values)
        axes[0, 1].set_title('Quantity Distribution')
        axes[0, 1].set_xlabel('Quantity')
        axes[0, 1].set_ylabel('Count')
    
    # Monthly orders
    if 'order_date' in df.columns:
        df_temp = df.copy()
        df_temp['month'] = pd.to_datetime(df_temp['order_date']).dt.month
        monthly_orders = df_temp.groupby('month').size()
        axes[1, 0].plot(monthly_orders.index, monthly_orders.values, marker='o')
        axes[1, 0].set_title('Orders by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Orders')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        axes[1, 1].bar(range(len(missing_data)), missing_data.values)
        axes[1, 1].set_title('Missing Values by Column')
        axes[1, 1].set_xticks(range(len(missing_data)))
        axes[1, 1].set_xticklabels(missing_data.index, rotation=45)
        axes[1, 1].set_ylabel('Missing Count')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                       transform=axes[1, 1].transAxes, 
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=14)
        axes[1, 1].set_title('Missing Values by Column')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {plots_dir}")


def create_summary_report(df, config, output_dir):
    """Create a summary report of the generated dataset."""
    logger.info("Creating summary report...")
    
    report_path = Path(output_dir) / "dataset_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("SYNTHETIC DATASET GENERATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random Seed: {config.get('random_seed', 'Not specified')}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Features: {len(df.columns)}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write("FEATURE TYPES\n")
        f.write("-" * 15 + "\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        f.write(f"Numeric Features ({len(numeric_cols)}): {', '.join(numeric_cols)}\n")
        f.write(f"Categorical Features ({len(categorical_cols)}): {', '.join(categorical_cols)}\n")
        f.write(f"DateTime Features ({len(datetime_cols)}): {', '.join(datetime_cols)}\n\n")
        
        f.write("MISSING VALUES\n")
        f.write("-" * 15 + "\n")
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary) > 0:
            for col, count in missing_summary.items():
                pct = (count / len(df)) * 100
                f.write(f"{col}: {count:,} ({pct:.1f}%)\n")
        else:
            f.write("No missing values in the dataset.\n")
        f.write("\n")
        
        f.write("STATISTICAL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(str(df.describe(include='all')))
        f.write("\n\n")
        
        if len(numeric_cols) > 1:
            f.write("CORRELATIONS\n")
            f.write("-" * 12 + "\n")
            corr_matrix = df[numeric_cols].corr()
            f.write(str(corr_matrix.round(3)))
    
    logger.info(f"Summary report saved to {report_path}")


@click.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Path to YAML configuration file')
@click.option('--output-dir', '-o',
              default='./output',
              help='Output directory for generated files')
@click.option('--csv', '--to-csv',
              is_flag=True,
              help='Output dataset as CSV file')
@click.option('--postgres', '--to-postgres',
              is_flag=True,
              help='Write dataset to PostgreSQL database')
@click.option('--pg-host',
              default='localhost',
              help='PostgreSQL host (default: localhost)')
@click.option('--pg-port',
              default='5432',
              help='PostgreSQL port (default: 5432)')
@click.option('--pg-database',
              default='synthetic_data',
              help='PostgreSQL database name (default: synthetic_data)')
@click.option('--pg-user',
              default='postgres',
              help='PostgreSQL username (default: postgres)')
@click.option('--pg-password',
              prompt=True,
              hide_input=True,
              default='',
              help='PostgreSQL password')
@click.option('--table-name',
              default='synthetic_data',
              help='PostgreSQL table name (default: synthetic_data)')
@click.option('--visualizations/--no-visualizations',
              default=True,
              help='Generate visualizations (default: True)')
@click.option('--summary/--no-summary',
              default=True,
              help='Generate summary report (default: True)')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
def generate_data(config, output_dir, csv, postgres, pg_host, pg_port, 
                 pg_database, pg_user, pg_password, table_name,
                 visualizations, summary, verbose):
    """
    Generate synthetic datasets with configurable features and output formats.
    
    This tool generates synthetic data based on a YAML configuration file and
    outputs the results in various formats including CSV, PostgreSQL, and
    visualizations.
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")
        
        # Load configuration
        config_data = load_config(config)
        
        # Generate data
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator(config_data)
        df = generator.generate_data()
        
        logger.info(f"Generated dataset with {len(df):,} records and {len(df.columns)} features")
        
        # Output to CSV if requested
        if csv:
            csv_path = output_path / f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Dataset saved to CSV: {csv_path}")
        
        # Output to PostgreSQL if requested
        if postgres:
            logger.info("Writing to PostgreSQL database...")
            pg_params = {
                'host': pg_host,
                'port': pg_port,
                'dbname': pg_database,
                'user': pg_user,
                'password': pg_password
            }
            
            success = write_to_postgres(df, table_name, pg_params)
            if success:
                logger.info(f"Dataset successfully written to PostgreSQL table '{table_name}'")
            else:
                logger.error("Failed to write to PostgreSQL database")
        
        # Generate visualizations if requested
        if visualizations:
            create_visualizations(df, output_path)
        
        # Generate summary report if requested
        if summary:
            create_summary_report(df, config_data, output_path)
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    generate_data()
