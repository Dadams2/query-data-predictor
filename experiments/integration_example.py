"""
Example integration showing how to use the enhanced experimental system
with your existing recommender evaluation code.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from experiments.recommender_experiments import RecommenderExperimentRunner
from experiments.experiment_analyzer import ExperimentAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_enhanced_experiment_example():
    """
    Example of running enhanced experiments and analyzing results.
    This shows the complete workflow from data collection to analysis.
    """
    
    logger.info("=== Enhanced Recommender Experiment Example ===")
    
    # 1. Initialize the enhanced experiment runner
    logger.info("Step 1: Initializing enhanced experiment runner...")
    
    runner = RecommenderExperimentRunner(
        config_path="config.yaml",
        dataset_dir="data/datasets", 
        output_dir="example_experiment_results",
        enable_full_tuple_storage=True,  # Store complete tuple data
        enable_state_tracking=False      # Don't track recommender state for this example
    )
    
    # 2. Run experiments on a sample session
    logger.info("Step 2: Running enhanced experiments...")
    
    # Get available sessions
    sessions = runner.dataloader.get_sessions()
    if not sessions:
        logger.error("No sessions found in dataset")
        return
    
    # Use the first session for this example
    sample_session = sessions[0]
    logger.info(f"Using sample session: {sample_session}")
    
    # Run the enhanced experiment
    experiment_results = runner.run_enhanced_experiment(
        session_id=sample_session,
        max_gap=3,  # Test gaps 1, 2, 3
        include_query_text=True,
        store_intermediate_states=False
    )
    
    # Display experiment summary
    logger.info("Experiment Summary:")
    for key, value in experiment_results.items():
        if key != 'experiment_ids':  # Skip the long list of IDs
            logger.info(f"  {key}: {value}")
    
    # 3. Analyze the results
    logger.info("Step 3: Analyzing experimental results...")
    
    analyzer = ExperimentAnalyzer(
        experiment_data_dir="example_experiment_results",
        include_tuple_analysis=False  # For faster analysis
    )
    
    # Load all results
    results_df = analyzer.load_all_results()
    
    if results_df.empty:
        logger.warning("No results found for analysis")
        return
    
    logger.info(f"Loaded {len(results_df)} experimental results for analysis")
    
    # Generate statistical summary
    stats_summary = analyzer.create_statistical_summary()
    
    logger.info("Statistical Summary:")
    if 'dataset_overview' in stats_summary:
        overview = stats_summary['dataset_overview']
        logger.info(f"  Total experiments: {overview.get('total_experiments', 0)}")
        logger.info(f"  Recommenders: {overview.get('recommenders', [])}")
        logger.info(f"  Gaps tested: {overview.get('gaps_tested', [])}")
    
    if 'performance_statistics' in stats_summary:
        logger.info("  Performance by recommender:")
        perf_stats = stats_summary['performance_statistics']
        if 'eval_overlap_accuracy' in perf_stats:
            for recommender, stats in perf_stats['eval_overlap_accuracy'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    logger.info(f"    {recommender}: {stats['mean']:.4f} ± {stats.get('std', 0):.4f}")
    
    # 4. Generate visualizations and reports
    logger.info("Step 4: Generating visualizations and reports...")
    
    # Create interactive dashboard
    dashboard_file = "example_dashboard.html"
    analyzer.generate_performance_dashboard(dashboard_file)
    logger.info(f"Interactive dashboard saved to: {dashboard_file}")
    
    # Create detailed comparison report
    report_file = "example_comparison_report.html"
    analyzer.create_detailed_comparison_report(report_file)
    logger.info(f"Detailed comparison report saved to: {report_file}")
    
    # Export data for further analysis
    export_dir = "example_analysis_exports"
    analyzer.export_for_further_analysis(export_dir)
    logger.info(f"Analysis exports saved to: {export_dir}/")
    
    # 5. Demonstrate data access patterns
    logger.info("Step 5: Demonstrating data access patterns...")
    
    # Show how to filter results
    clustering_results = analyzer.load_all_results(
        recommender_names=['clustering']
    )
    logger.info(f"Clustering-only results: {len(clustering_results)} experiments")
    
    # Show how to access specific experimental data
    if len(experiment_results.get('experiment_ids', [])) > 0:
        sample_experiment_id = experiment_results['experiment_ids'][0]
        logger.info(f"Sample experiment ID: {sample_experiment_id}")
        
        # The experiment data is stored in structured files
        metadata_file = Path("example_experiment_results") / "metadata" / f"{sample_experiment_id}.json"
        if metadata_file.exists():
            logger.info(f"Experiment metadata available at: {metadata_file}")
    
    logger.info("=== Enhanced Experiment Example Complete ===")
    logger.info("Check the generated files for detailed results and visualizations!")


def compare_old_vs_new_approach():
    """
    Show the difference between the old and new experimental approaches.
    """
    
    logger.info("=== Comparison: Old vs New Approach ===")
    
    # Old approach (simplified)
    logger.info("OLD APPROACH:")
    logger.info("- Results stored in single DataFrame/CSV")
    logger.info("- Limited metadata tracking")
    logger.info("- Basic metrics only")
    logger.info("- Manual analysis required")
    logger.info("- No provenance tracking")
    
    # New approach
    logger.info("\nNEW ENHANCED APPROACH:")
    logger.info("- Structured data storage with full provenance")
    logger.info("- Comprehensive metadata for each experiment")
    logger.info("- Multiple evaluation metrics (accuracy, precision, recall, F1)")
    logger.info("- Automated analysis and visualization")
    logger.info("- Support for stateful recommenders")
    logger.info("- Interactive dashboards and reports")
    logger.info("- Export capabilities for further analysis")
    logger.info("- Error tracking and timeout handling")
    
    # Data structure comparison
    logger.info("\nDATA STRUCTURE COMPARISON:")
    
    logger.info("Old format (simplified):")
    logger.info("  {session_id, current_query, future_query, recommender, accuracy, time}")
    
    logger.info("\nNew format (comprehensive):")
    logger.info("  {")
    logger.info("    experiment_id: unique identifier,")
    logger.info("    metadata: {timestamp, session, recommender, config, status},")
    logger.info("    current_query_context: {position, text, result_size},")
    logger.info("    target_query_context: {position, text, result_size},")
    logger.info("    recommendation_summary: {predicted_count, confidence_info},")
    logger.info("    evaluation: {accuracy, precision, recall, f1, jaccard},")
    logger.info("    provenance: {execution_time, error_info}")
    logger.info("  }")


if __name__ == "__main__":
    # Run the example
    try:
        run_enhanced_experiment_example()
        print("\n" + "="*60)
        compare_old_vs_new_approach()
        
    except Exception as e:
        logger.error(f"Error running example: {e}")
        logger.info("Make sure you have:")
        logger.info("1. The required dependencies installed (see enhanced_requirements.txt)")
        logger.info("2. Your dataset files in the correct location")
        logger.info("3. The config.yaml file in the current directory")
