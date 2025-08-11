#!/usr/bin/env python3
"""
Test script for the enhanced experiment system with @k metrics and adaptive sizing.
"""

import logging
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.recommender_experiments import RecommenderExperimentRunner
from query_data_predictor.dataloader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run test experiments with the enhanced system."""
    
    logger.info("Starting enhanced experiment system test")
    
    # Initialize the experiment runner
    runner = RecommenderExperimentRunner(
        config_path="config.yaml",
        dataset_dir="data/datasets",
        output_dir="experiment_data_test",
        enable_full_tuple_storage=True,
        enable_state_tracking=False
    )
    
    # Get available sessions
    sessions = runner.dataloader.get_sessions()
    
    if not sessions:
        logger.error("No sessions found in dataset")
        return
    
    logger.info(f"Found {len(sessions)} sessions in dataset")
    
    # Test with the first session
    test_session = sessions[0]
    logger.info(f"Testing with session {test_session}")
    
    # Run standard enhanced experiment
    logger.info("Running standard enhanced experiment with @k metrics...")
    try:
        standard_results = runner.run_enhanced_experiment(
            session_id=str(test_session),
            max_gap=3,  # Limited for testing
            include_query_text=False,
            store_intermediate_states=False
        )
        
        logger.info(f"Standard experiment completed: {standard_results['successful_experiments']}/{standard_results['total_experiments']} successful")
        
    except Exception as e:
        logger.error(f"Error in standard experiment: {e}", exc_info=True)
    
    # Run adaptive size experiment
    logger.info("Running adaptive size experiment...")
    try:
        adaptive_results = runner.run_adaptive_size_experiment(
            session_id=str(test_session),
            max_gap=3,  # Limited for testing
            include_query_text=False,
            store_intermediate_states=False
        )
        
        logger.info(f"Adaptive size experiment completed: {adaptive_results['successful_experiments']}/{adaptive_results['total_experiments']} successful")
        
    except Exception as e:
        logger.error(f"Error in adaptive size experiment: {e}", exc_info=True)
    
    # Create analysis and visualizations
    logger.info("Creating analysis and visualizations...")
    try:
        from experiments.experiment_analyzer import ExperimentAnalyzer
        
        analyzer = ExperimentAnalyzer(
            experiment_data_dir=runner.collector.base_output_dir,
            include_tuple_analysis=False
        )
        
        # Load results
        results_df = analyzer.load_all_results()
        
        if not results_df.empty:
            logger.info(f"Loaded {len(results_df)} experimental results for analysis")
            
            # Generate standard dashboard
            analyzer.generate_performance_dashboard("test_dashboard.html")
            logger.info("Standard dashboard created")
            
            # Generate enhanced dashboard with @k metrics
            analyzer.generate_enhanced_dashboard_with_k_metrics("test_enhanced_dashboard.html")
            logger.info("Enhanced dashboard with @k metrics created")
            
            # Create statistical summary
            stats = analyzer.create_statistical_summary()
            logger.info(f"Statistical summary created with {len(stats)} sections")
            
        else:
            logger.warning("No results found for analysis")
            
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
    
    logger.info("Enhanced experiment system test completed")


if __name__ == "__main__":
    main()
