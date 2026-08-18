"""
Query Data Predictor - A framework for predicting results of successive database queries.
"""

def main():
    from query_data_predictor.cli import main as cli

    return cli()

__all__ = ['main']
