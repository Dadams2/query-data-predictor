# Multi-Session Recommender Experiments

This document describes the enhanced experiment system for running recommender evaluations across multiple sessions with comprehensive analysis and visualization.

## Overview

The enhanced system extends the original single-session experiment framework to:
- Run experiments across multiple sessions automatically
- Provide intelligent session selection strategies
- Generate comprehensive visualizations and statistical analysis
- Support configurable experiment parameters
- Create publication-ready figures and reports

## Key Components

### 1. Enhanced RecommenderExperimentRunner

The main class now supports:
- **Multi-session experiments**: `run_multi_session_experiments()`
- **Session selection**: `select_sessions_for_experiment()` with multiple strategies
- **Session statistics**: `get_session_statistics()` for informed selection
- **Comprehensive analysis**: `create_comprehensive_analysis_and_visualization()`

### 2. Session Selection Strategies

- **Random**: Random selection of sessions
- **Largest**: Sessions with the most queries
- **Smallest**: Sessions with the fewest queries (but above minimum)
- **Diverse**: Evenly distributed across query count range

### 3. Configuration System

The `multi_session_config.yaml` file allows you to configure:
- Session selection parameters
- Experiment parameters (gaps, query text extraction, etc.)
- Output configuration
- Analysis settings
- Visualization preferences

### 4. Visualization and Analysis

The system integrates with the existing `ExperimentAnalyzer` to create:
- Interactive performance dashboards
- Statistical significance tests
- Publication-ready figures
- Comparative analysis reports
- Data exports for further analysis

## Usage

### Quick Start

```bash
# Run with default settings (10 sessions, diverse selection)
cd experiments
python recommender_experiments.py

# Run single session test for debugging
python recommender_experiments.py single
```

### Advanced Usage

```bash
# Run comprehensive experiments with custom configuration
python run_multi_session_experiments.py --config custom_config.yaml

# Run with specific number of sessions
python run_multi_session_experiments.py --max-sessions 20

# Dry run to see what would be done
python run_multi_session_experiments.py --dry-run

# Verbose logging
python run_multi_session_experiments.py --verbose
```

### Configuration Examples

#### Small-scale testing:
```yaml
experiment_config:
  session_selection:
    min_queries: 2
    max_sessions: 5
    selection_strategy: "random"
  experiment_params:
    max_gap: 3
```

#### Large-scale comprehensive study:
```yaml
experiment_config:
  session_selection:
    min_queries: 5
    max_sessions: 50
    selection_strategy: "diverse"
  experiment_params:
    max_gap: 10
    include_query_text: true
```

## Output Structure

The system creates organized output directories:

```
results/experiment/experiment_results_YYYYMMDD_HHMMSS/
├── experiment_data/           # Raw experimental data
├── analysis/                  # Analysis outputs
│   ├── multi_session_performance_dashboard.html
│   ├── multi_session_comparison_report.html
│   ├── publication_figures/   # Publication-ready figures
│   └── data_exports/         # Data in multiple formats
└── logs/                     # Experiment logs
```

## Key Features

### 1. Robust Error Handling
- Sessions with insufficient queries are automatically skipped
- Failed experiments are logged but don't stop the overall process
- Timeout handling for long-running recommenders

### 2. Comprehensive Metrics
- Overlap accuracy, precision, recall, F1 score
- Execution time tracking
- Statistical significance testing
- Correlation analysis

### 3. Publication-Ready Outputs
- High-resolution figures (300 DPI)
- Multiple output formats (PNG, PDF)
- Professional styling and layout
- Interactive HTML dashboards

### 4. Scalable Architecture
- Configurable memory and timeout limits
- Parallel processing options (future enhancement)
- Efficient data storage and retrieval

## Session Selection Guidelines

### For Development/Testing:
- Use `max_sessions: 5-10`
- Set `min_queries: 2-3`
- Use `selection_strategy: "random"`

### For Comprehensive Studies:
- Use `max_sessions: 20-50`
- Set `min_queries: 5+`
- Use `selection_strategy: "diverse"`

### For Performance Comparisons:
- Use `max_sessions: 10-20`
- Set `min_queries: 3-5`
- Use `selection_strategy: "largest"`

## Analysis and Visualization

The system automatically generates:

1. **Performance Dashboard**: Interactive HTML dashboard with multiple views
2. **Statistical Summary**: Comprehensive statistical analysis with significance tests
3. **Publication Figures**: High-quality figures for papers/presentations
4. **Comparison Report**: Detailed HTML report with recommendations
5. **Data Exports**: CSV, Parquet, and JSON formats for external analysis

## Best Practices

1. **Start Small**: Begin with 5-10 sessions to validate the setup
2. **Monitor Resources**: Watch memory usage with large datasets
3. **Use Dry Run**: Test configuration with `--dry-run` before full runs
4. **Configure Timeouts**: Adjust timeout values based on your data size
5. **Review Session Stats**: Check session statistics before running experiments

## Troubleshooting

### Common Issues:

1. **No suitable sessions found**: Lower `min_queries` or check your dataset
2. **Memory issues**: Reduce `max_sessions` or disable tuple storage
3. **Timeout errors**: Increase timeout values in configuration
4. **Visualization errors**: Check that required packages are installed

### Debug Mode:
```bash
python run_multi_session_experiments.py --verbose --max-sessions 2 --dry-run
```

## Future Enhancements

- Parallel session processing
- Real-time progress monitoring
- Integration with experiment tracking systems
- Advanced statistical analysis methods
- Custom metric definitions
