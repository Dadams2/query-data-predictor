"""
Command-line interface for query_data_predictor using Click.
"""

import click
import logging
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime
from query_data_predictor.config_manager import ConfigManager
from query_data_predictor.experiment_runner import ExperimentRunner
from query_data_predictor.logging_config import setup_logging

logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, verbose):
    """Query Data Predictor - A framework for predicting database query results."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level)
    
    ctx.obj['verbose'] = verbose


@main.command()
@click.option('--config', '-c', 
              type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file (YAML)')
@click.option('--data-path', '-d',
              type=click.Path(exists=True, path_type=Path),
              required=False,
              help='Path to the dataset directory containing metadata.csv')
@click.option('--session-id', '-s',
              help='Specific session ID to run experiment for')
@click.option('--gap', '-g',
              type=int,
              default=1,
              help='Number of queries to skip between prediction pairs (default: 1)')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for results')
@click.pass_context
def run_experiment(ctx, config, data_path, session_id, gap, output):
    """Run prediction experiment on query data."""
    click.echo("Starting query prediction experiment...")

    # Load configuration
    # config_manager = ConfigManager(str(config))
    # config = config_manager.get_config()
    # read in raw config
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    # get values from config but overwrite with args if they exist
    data_path = data_path or config.get('experiment', {}).get('data_path', None)
    sessions = session_id or config.get('experiment', {}).get('sessions', [])
    gap = [gap] or config.get('experiment', {}).get('prediction_gap', [1])
    output_dir = output or config.get('output', {}).get('output_directory', None)

    # resolve output path then add timestamped subdirectory with name
    if output_dir:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = Path.cwd().resolve() / 'results'

    # resolve data path (remove and resolve ../../)
    data_path = Path(data_path).resolve()

    try:
        logger.info(f"Using data path: {data_path}")
        logger.info(f"Using output directory: {output_dir}")
        logger.info(f"Using session IDs: {sessions}")

        runner = ExperimentRunner(
            output_dir=output_dir,
            data_path=data_path,
            sessions=sessions,
            gap=gap,
            config=config
        )

        runner.run_experiment()

        logger.info(f"Completed recommender experiment for all sessions")

    except Exception as e:
        click.echo(f"Error running experiment: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        raise click.ClickException(f"Experiment failed: {e}")


@main.command()
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file to validate')
def validate_config(config):
    """Validate a configuration file."""
    if not config:
        click.echo("No config file specified, using default configuration...")
        config_manager = ConfigManager()
    else:
        click.echo(f"Validating configuration file: {config}")
        config_manager = ConfigManager(str(config))
    
    try:
        config_data = config_manager.get_config()
        click.echo("✓ Configuration is valid")
        
        # Display key configuration settings
        click.echo("\nConfiguration summary:")
        if 'experiment' in config_data:
            exp_config = config_data['experiment']
            click.echo(f"  Experiment: {exp_config.get('name', 'unnamed')}")
            click.echo(f"  Prediction gap: {exp_config.get('prediction_gap', 1)}")
            
        if 'discretization' in config_data:
            disc_config = config_data['discretization']
            click.echo(f"  Discretization: {'enabled' if disc_config.get('enabled') else 'disabled'}")
            if disc_config.get('enabled'):
                click.echo(f"    Method: {disc_config.get('method', 'equal_width')}")
                click.echo(f"    Bins: {disc_config.get('bins', 5)}")
                
        if 'association_rules' in config_data:
            ar_config = config_data['association_rules']
            click.echo(f"  Association rules: {'enabled' if ar_config.get('enabled') else 'disabled'}")
            if ar_config.get('enabled'):
                click.echo(f"    Min support: {ar_config.get('min_support', 0.1)}")
                click.echo(f"    Min threshold: {ar_config.get('min_threshold', 0.7)}")
        
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        raise click.ClickException(f"Invalid configuration: {e}")


@main.command()
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='config.yaml',
              help='Output path for generated config file')
def generate_config(output):
    """Generate a sample configuration file."""
    click.echo(f"Generating sample configuration file: {output}")
    
    sample_config = """# Configuration for query prediction experiment
# Adjust parameters as needed for your specific use case

experiment:
  name: 'sample_experiment'
  prediction_gap: 1
  random_seed: 42
  sessions_limit: null  # null for all sessions, integer for limited number

discretization:
  enabled: true
  method: 'equal_width'  # Options: equal_width, equal_freq, kmeans
  bins: 5
  save_params: true
  params_path: 'discretization_params.pkl'

association_rules:
  enabled: true
  min_support: 0.1
  metric: 'confidence'  # Options: support, confidence, lift, leverage, conviction
  min_threshold: 0.7
  max_len: null  # null for no limit, integer for max rule length

summaries:
  enabled: true
  desired_size: 5  # Number of summaries to generate
  weights: null  # Custom weights for attributes, null for equal weights

interestingness:
  enabled: true
  measures: ["variance", "simpson", "shannon"]

evaluation:
  jaccard_threshold: 0.5
  column_weights: null  # Custom weights for columns, null for equal weights

output:
  results_dir: 'experiment_results'
  save_predictions: true
  save_metrics: true
  save_summaries: true
"""
    
    try:
        output_path = Path(output)
        output_path.write_text(sample_config)
        click.echo(f"✓ Sample configuration saved to: {output_path}")
    except Exception as e:
        click.echo(f"✗ Failed to save configuration: {e}", err=True)
        raise click.ClickException(f"Failed to generate config: {e}")


@main.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata
        version = importlib.metadata.version('query-data-predictor')
        click.echo(f"query-data-predictor version {version}")
    except Exception:
        click.echo("query-data-predictor (development version)")


if __name__ == '__main__':
    main()
