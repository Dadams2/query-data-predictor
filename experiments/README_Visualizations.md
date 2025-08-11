# Recommender Experiment Visualizations

This document explains how to use the comprehensive visualization features added to the recommender experiments system.

## Overview

The visualization system provides multiple ways to analyze and visualize recommender system performance:

1. **Built-in visualizations** - Integrated into the main experiment runner
2. **Standalone visualizer** - Can visualize existing CSV result files
3. **Interactive dashboard** - HTML dashboard with key metrics
4. **Multiple plot types** - Box plots, heatmaps, distributions, correlations, etc.

## Features

### Visualization Types

1. **Accuracy Comparison**
   - Box plots showing accuracy distribution by recommender
   - Bar plots with confidence intervals
   - Statistical significance testing

2. **Gap Analysis**
   - Line plots showing how accuracy changes with query gap
   - Heatmaps of accuracy vs gap for each recommender
   - Trend analysis across different gap sizes

3. **Result Size Analysis**
   - Violin plots of accuracy by result set size bins
   - Scatter plots with trend lines
   - Performance analysis for different data sizes

4. **Execution Time Analysis**
   - Box plots of execution times by recommender
   - Scatter plots of time vs result size
   - Efficiency metrics (accuracy per second)
   - Time distribution histograms

5. **Performance Heatmaps**
   - Multi-dimensional performance comparison
   - Accuracy, timing, and count matrices
   - Easy identification of best-performing combinations

6. **Distribution Analysis**
   - Histograms and KDE plots of accuracy distributions
   - Cumulative distribution functions
   - Statistical summary visualizations

7. **Correlation Analysis**
   - Correlation matrices of numerical variables
   - Scatter plots showing relationships
   - Feature interaction analysis

8. **Error Analysis**
   - Error rates by recommender and conditions
   - Success rates across different scenarios
   - Failure pattern identification

## Usage

### Method 1: Built-in Visualizations

The main experiment runner now automatically creates visualizations:

```python
from experiments.recommender_experiments import RecommenderExperimentRunner

# Run experiments
runner = RecommenderExperimentRunner()
results_df = runner.run_experiment("session_id", max_gap=5)

# Create visualizations
runner.create_visualizations(
    output_dir="experiment_results_20250701", 
    save_pdf=True  # Saves all plots in one PDF file
)

# Create interactive dashboard
runner.create_summary_dashboard("experiment_results_20250701/dashboard.html")
```

### Method 2: Standalone Visualizer

Use the standalone script to visualize existing results:

```bash
# Basic usage
python experiments/visualize_results.py results.csv

# With custom output directory
python experiments/visualize_results.py results.csv --output-dir my_visualizations

# PDF only (no individual PNG files)
python experiments/visualize_results.py results.csv --no-individual
```

Or use it programmatically:

```python
from experiments.visualize_results import ResultsVisualizer

visualizer = ResultsVisualizer("experiment_results.csv")
visualizer.create_all_visualizations(
    output_dir="visualizations",
    save_individual=True
)
```

### Method 3: Demo Script

Run the demo to see all features in action:

```bash
python experiments/demo_visualizations.py
```

## Output Files

The visualization system creates several types of output:

### PDF Report
- `all_plots_TIMESTAMP.pdf` - Single PDF with all visualizations
- Perfect for presentations and reports

### Individual PNG Files
- `accuracy_comparison_TIMESTAMP.png`
- `gap_analysis_TIMESTAMP.png`
- `size_analysis_TIMESTAMP.png`
- `performance_heatmap_TIMESTAMP.png`
- `distributions_TIMESTAMP.png`
- `timing_analysis_TIMESTAMP.png` (if timing data available)
- High-resolution (300 DPI) for publications

### Interactive Dashboard
- `dashboard.html` - Interactive HTML dashboard
- Summary statistics and key metrics
- Can be opened in any web browser

### Summary Report
- `summary_report_TIMESTAMP.txt` - Text summary of results
- Best performing recommenders
- Statistical summaries

## Customization

### Plot Styling

The visualizations use seaborn's "whitegrid" style by default. You can customize this by modifying the style settings in the code:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Custom styling
sns.set_style("darkgrid")  # or "white", "dark", "ticks"
plt.rcParams['figure.figsize'] = (16, 10)  # Larger figures
plt.rcParams['figure.dpi'] = 150  # Higher resolution
```

### Color Palettes

Seaborn automatically chooses appropriate color palettes, but you can customize them:

```python
# Set custom color palette
sns.set_palette("husl")  # or "Set2", "viridis", etc.
```

### Adding New Visualizations

To add new visualization types, extend the visualization methods in `RecommenderExperimentRunner`:

```python
def _plot_custom_analysis(self, df: pd.DataFrame, pdf: PdfPages = None, 
                         output_path: Path = None, timestamp: str = None):
    """Create custom analysis plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Your custom visualization code here
    # ...
    
    plt.tight_layout()
    self._save_plot(fig, pdf, output_path, f'custom_analysis_{timestamp}')
```

Then add it to the `_create_all_plots` method.

## Dependencies

The visualization features require:
- `matplotlib >= 3.10.1`
- `seaborn >= 0.13.2`
- `pandas >= 2.2.3`
- `numpy`

These are already included in the project dependencies.

## Examples

### Analyzing Performance by Gap

```python
# Run experiment with multiple gaps
results_df = runner.run_experiment("session_id", max_gap=7)

# The gap analysis plots will show:
# - How accuracy degrades with larger gaps
# - Which recommenders maintain performance better
# - Optimal gap sizes for each recommender
```

### Comparing Recommender Efficiency

```python
# The timing analysis plots will show:
# - Which recommenders are fastest
# - How execution time scales with data size
# - Efficiency metrics (accuracy per second)
```

### Identifying Optimal Conditions

```python
# The heatmaps will help identify:
# - Best recommender for each gap size
# - Optimal result set sizes
# - Conditions where each recommender excels
```

## Troubleshooting

### Common Issues

1. **No valid results to visualize**
   - Check that your experiment generated successful results
   - Verify that overlap_accuracy values are not all NaN

2. **Memory issues with large datasets**
   - Use smaller gap ranges for initial exploration
   - Consider sampling your data for visualization

3. **Missing timing data**
   - Some visualizations require execution_time column
   - This is automatically collected in newer experiments

4. **Display issues**
   - If running on a headless server, visualizations will save to files
   - PDF output always works regardless of display availability

### Performance Tips

- Use `save_pdf=True` for comprehensive reports
- Set `save_individual=False` if you only need the PDF
- For large datasets, consider creating visualizations in batches

## Integration with Jupyter Notebooks

The visualization functions work well in Jupyter notebooks:

```python
# In a Jupyter cell
%matplotlib inline
runner.create_visualizations(save_pdf=False)  # Display inline
```

## Future Enhancements

Potential future additions:
- Interactive plots using Plotly
- Statistical significance testing
- Animated plots showing performance over time
- Integration with MLflow or similar experiment tracking tools
