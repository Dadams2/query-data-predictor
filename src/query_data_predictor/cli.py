"""
Command-line interface for query_data_predictor using Click.
"""

import click
import logging
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime
from query_data_predictor.analysis import ResultsAnalyzer
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
    prediction_gap = [gap] if gap else config.get('experiment', {}).get('prediction_gap', [])
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
            gap=prediction_gap,
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
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file (YAML)')
@click.option('--results-path', '-r',
              type=click.Path(exists=True, path_type=Path),
              help='Path to results directory (overrides auto-detection)')
@click.pass_context
def analyze_results(ctx, config, results_path):
    """Analyze experiment results and generate plots/visualizations."""
    click.echo("Starting results analysis...")
    
    try:
        from query_data_predictor.analysis import ResultsAnalyzer
        
        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
        
        # Determine results path
        if results_path:
            # Use provided results path directly
            target_results_dir = Path(results_path).resolve()
            click.echo(f"Using provided results directory: {target_results_dir}")
        else:
            # Auto-detect most recent results directory
            config_output_dir = config_data.get('output', {}).get('output_directory')
            if config_output_dir:
                base_output = Path(config_output_dir).resolve()
            else:
                base_output = Path.cwd().resolve() / 'results'
            
            target_results_dir = _find_most_recent_results_dir(base_output)
            if not target_results_dir:
                raise click.ClickException("No results directory found. Please specify --results-path or run an experiment first.")
            click.echo(f"Auto-detected results directory: {target_results_dir}")
        
        # Initialize and run analyzer
        analyzer = ResultsAnalyzer(
            results_dir=target_results_dir,
            config=config_data
        )
        
        analysis_results = analyzer.analyze()
        
        # Report on generated outputs
        if analysis_results and 'visualizations' in analysis_results:
            viz_results = analysis_results['visualizations']
            if 'plots_created' in viz_results and viz_results['plots_created']:
                click.echo(f"✓ Generated {len(viz_results['plots_created'])} visualization files:")
                for plot_file in viz_results['plots_created']:
                    click.echo(f"  - {plot_file}")
                click.echo(f"✓ Analysis outputs saved to: {viz_results.get('output_directory', 'unknown')}")
            else:
                click.echo("✓ Analysis completed but no visualizations were generated")
        
        click.echo("✓ Analysis completed successfully")
        
    except ImportError:
        click.echo("✗ Analysis module not found. Creating stub implementation...", err=True)
        _stub_analyze_results(results_path or Path.cwd() / 'results')
    except Exception as e:
        click.echo(f"✗ Analysis failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        raise click.ClickException(f"Analysis failed: {e}")


@main.command()
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file (YAML)')
@click.option('--results-path', '-r',
              type=click.Path(exists=True, path_type=Path),
              help='Path to results directory (overrides auto-detection)')
@click.pass_context
def analyze_simple(ctx, config, results_path):
    """Simple analysis across three match scenarios: raw, close, similarity.

    Creates per-session subdirectories `raw`, `close`, and `similarity` under the analysis
    folder and writes plots and CSV summaries for accuracy, precision, recall and F1.
    """
    click.echo("Starting simple results analysis (raw / close / similarity)...")

    try:
        from query_data_predictor.analysis import ResultsAnalyzer

        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}

        # Determine results path
        if results_path:
            target_results_dir = Path(results_path).resolve()
            click.echo(f"Using provided results directory: {target_results_dir}")
        else:
            config_output_dir = config_data.get('output', {}).get('output_directory')
            if config_output_dir:
                base_output = Path(config_output_dir).resolve()
            else:
                base_output = Path.cwd().resolve() / 'results'

            target_results_dir = _find_most_recent_results_dir(base_output)
            if not target_results_dir:
                raise click.ClickException("No results directory found. Please specify --results-path or run an experiment first.")
            click.echo(f"Auto-detected results directory: {target_results_dir}")

        analyzer = ResultsAnalyzer(results_dir=target_results_dir, config=config_data)
        outputs = analyzer.analyze_simple()

        click.echo("✓ Simple analysis completed")
        if outputs and 'output_dirs' in outputs:
            for scenario, path in outputs['output_dirs'].items():
                click.echo(f"  - {scenario}: {path}")

    except Exception as e:
        click.echo(f"✗ Simple analysis failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        raise click.ClickException(f"Simple analysis failed: {e}")


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
def run_and_analyze(ctx, config, data_path, session_id, gap, output):
    """Run prediction experiment and immediately analyze the results."""
    click.echo("Starting experiment and analysis pipeline...")
    
    # First run the experiment
    try:
        ctx.invoke(run_experiment, 
                  config=config, 
                  data_path=data_path, 
                  session_id=session_id, 
                  gap=gap, 
                  output=output)
    except Exception as e:
        click.echo(f"✗ Experiment failed: {e}", err=True)
        raise
    
    # Then run the analysis on the results
    ctx.invoke(analyze_results, config=config)


@main.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata
        version = importlib.metadata.version('query-data-predictor')
        click.echo(f"query-data-predictor version {version}")
    except Exception:
        click.echo("query-data-predictor (development version)")


def _find_most_recent_results_dir(base_path: Path = None) -> Optional[Path]:
    """Find the most recent timestamped results directory."""
    if base_path is None:
        base_path = Path.cwd() / 'results'
    
    if not base_path.exists():
        return None
    
    # Look for timestamped directories matching pattern: {experiment_name}_{YYYYMMDD-HHMMSS}
    timestamped_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and '_' in item.name:
            # Check if the suffix looks like a timestamp
            parts = item.name.split('_')
            if len(parts) >= 2:
                timestamp_part = parts[-1]
                if len(timestamp_part) == 15 and '-' in timestamp_part:  # YYYYMMDD-HHMMSS format
                    try:
                        # Validate timestamp format
                        date_part, time_part = timestamp_part.split('-')
                        if (len(date_part) == 8 and date_part.isdigit() and 
                            len(time_part) == 6 and time_part.isdigit()):
                            timestamped_dirs.append((item, timestamp_part))
                    except ValueError:
                        continue
    
    if not timestamped_dirs:
        return None
    
    # Sort by timestamp and return the most recent
    timestamped_dirs.sort(key=lambda x: x[1], reverse=True)
    return timestamped_dirs[0][0]


def _stub_analyze_results(results_path: Path):
    """Stub implementation for analysis when the module doesn't exist yet."""
    logger.info("=== STUB ANALYSIS IMPLEMENTATION ===")
    logger.info(f"Results path: {results_path}")
    
    if not results_path.exists():
        logger.error(f"Results directory does not exist: {results_path}")
        return
    
    # Create analysis subdirectory
    analysis_dir = results_path / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created analysis directory: {analysis_dir}")
    
    # Find all result files
    result_files = list(results_path.glob("*.json"))
    if not result_files:
        logger.warning(f"No JSON result files found in: {results_path}")
        return
    
    # Parse the filenames to extract session and gap information
    sessions = set()
    gaps = set()
    
    for file_path in result_files:
        # Expected format: {session_id}__gap-{gap_value}.json
        filename = file_path.stem
        if '__gap-' in filename:
            session_part, gap_part = filename.split('__gap-', 1)
            sessions.add(session_part)
            gaps.add(gap_part)
    
    experiment_name = results_path.name
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Sessions found: {sorted(sessions)}")
    logger.info(f"Gaps found: {sorted(gaps)}")
    logger.info(f"Total result files: {len(result_files)}")
    
    # Log some file details
    for file_path in sorted(result_files)[:5]:  # Show first 5 files
        file_size = file_path.stat().st_size
        logger.info(f"  - {file_path.name} ({file_size} bytes)")
    
    if len(result_files) > 5:
        logger.info(f"  ... and {len(result_files) - 5} more files")
    
    # Create a simple summary file in the analysis directory
    summary_file = analysis_dir / 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Analysis Summary for {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total result files: {len(result_files)}\n")
        f.write(f"Sessions: {sorted(sessions)}\n")
        f.write(f"Gaps: {sorted(gaps)}\n")
        f.write(f"Analysis directory: {analysis_dir}\n")
    
    logger.info(f"Summary written to: {summary_file}")
    logger.info("=== END STUB ANALYSIS ===")


if __name__ == '__main__':
    main()
