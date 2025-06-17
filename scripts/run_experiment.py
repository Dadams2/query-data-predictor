#!/usr/bin/env python
"""
Script to run the query prediction experiment.
"""

import os
import sys
import click
import logging
from pathlib import Path

from query_data_predictor.experiment_runner import ExperimentRunner


@click.command()
@click.option(
    "--data-path", "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the dataset directory containing metadata.csv"
)
@click.option(
    "--config-path", "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="config.yaml",
    show_default=True,
    help="Path to the configuration file (YAML or JSON)"
)
@click.option(
    "--session-id", "-s",
    type=str,
    help="Optional session ID to run the experiment for a specific session"
)
@click.option(
    "--log-level", "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Set the logging level"
)
def main(data_path: Path, config_path: Path, session_id: str, log_level: str):
    """Run query prediction experiments."""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"run_experiment_{session_id or 'all'}.log")
        ]
    )
    
    logger = logging.getLogger("run_experiment")
    
    try:
        # Initialize the experiment runner
        logger.info(f"Initializing experiment runner with data path: {data_path}")
        runner = ExperimentRunner(str(data_path), str(config_path))
        
        # Run the experiment
        if session_id:
            logger.info(f"Running experiment for session: {session_id}")
            results = runner.run_experiment_for_session(session_id)
        else:
            logger.info("Running experiment for all sessions")
            results = runner.run_experiment_for_all_sessions()
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
