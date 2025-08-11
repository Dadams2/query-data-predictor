# Publication-Ready Visualizations for ExperimentAnalyzer

This document describes the comprehensive publication-ready visualization capabilities added to the `ExperimentAnalyzer` class.

## Overview

The `ExperimentAnalyzer` now includes a complete suite of publication-quality visualization tools that generate professional figures suitable for academic papers, conference presentations, and technical reports. All visualizations follow publication standards with proper fonts, sizing, statistical annotations, and high-resolution output.

## Quick Start

```python
from experiments.experiment_analyzer import ExperimentAnalyzer

# Initialize analyzer
analyzer = ExperimentAnalyzer("path/to/experiment/results")

# Load experimental data
results_df = analyzer.load_all_results()

# Create all publication visualizations
saved_files = analyzer.create_publication_visualizations(
    output_dir="results/experiment/experiment_results_analysis/analysis",
    save_pdf=True,        # Combined PDF
    save_individual=True, # Individual PNG files  
    dpi=300,             # Publication quality
    figsize=(12, 8)      # Standard size
)

# Creates interactive dashboard too
print("Generated files:", saved_files)
```

## Available Visualizations

### 1. Accuracy Comparison (`accuracy_comparison.png`)
**Purpose**: Comprehensive comparison of recommender system accuracy across multiple dimensions.

**Contents**:
- **Panel A**: Box plot comparison with statistical significance annotations
- **Panel B**: Violin plots showing distribution shapes and quartiles
- **Panel C**: Mean accuracy with 95% confidence intervals and value labels
- **Panel D**: Cumulative distribution functions for detailed comparison

**Use Cases**: Primary comparison figure for papers, showing both central tendency and variability.

### 2. Gap Analysis (`gap_analysis.png`)
**Purpose**: Analysis of how query gap (temporal distance) affects recommender performance.

**Contents**:
- **Panel A**: Line plots showing accuracy trends across gap values
- **Panel B**: Heatmap of mean accuracy by recommender × gap
- **Panel C**: Box plots showing accuracy distribution for each gap
- **Panel D**: Effect size analysis comparing to baseline gap

**Use Cases**: Understanding temporal effects, optimizing recommendation timing.

### 3. Result Size Analysis (`result_size_analysis.png`)
**Purpose**: Impact of predicted result set size on accuracy and system behavior.

**Contents**:
- **Panel A**: Accuracy by result size categories (Tiny/Small/Medium/Large/Huge)
- **Panel B**: Scatter plot of accuracy vs predicted count (log scale)
- **Panel C**: Result size distribution by recommender (violin plots)
- **Panel D**: Success rate heatmap by size category

**Use Cases**: Scalability analysis, understanding performance across different query types.

### 4. Execution Time Analysis (`execution_time_analysis.png`)
**Purpose**: Performance timing analysis and efficiency metrics.

**Contents**:
- **Panel A**: Execution time distribution by recommender (log scale)
- **Panel B**: Time vs result size relationship (log-log plot)
- **Panel C**: Efficiency analysis (accuracy per second)
- **Panel D**: Time distribution histograms with density curves

**Use Cases**: Performance optimization, computational cost analysis.

### 5. Performance Heatmap (`performance_heatmap.png`)
**Purpose**: Multi-dimensional performance visualization for pattern identification.

**Contents**:
- **Panel A**: Mean accuracy heatmap by recommender × gap
- **Panel B**: Experiment count heatmap (coverage visualization)
- **Panel C**: Accuracy variability (standard deviation) heatmap
- **Panel D**: Performance by result size category

**Use Cases**: Quick identification of optimal conditions, experimental coverage assessment.

### 6. Distribution Analysis (`distribution_analysis.png`)
**Purpose**: Statistical distribution analysis and normality assessment.

**Contents**:
- **Panel A**: Histograms with kernel density estimation
- **Panel B**: Q-Q plots for normality checking
- **Panel C**: Box-Cox transformed data (if applicable)
- **Panel D**: Empirical cumulative distribution comparison

**Use Cases**: Statistical method validation, assumption checking for parametric tests.

### 7. Correlation Analysis (`correlation_analysis.png`)
**Purpose**: Relationship analysis between experimental variables.

**Contents**:
- **Panel A**: Correlation matrix with hierarchical clustering
- **Panel B**: Key relationship scatter plots
- **Panel C**: Scatter plot matrix for main variables
- **Panel D**: Feature importance analysis (if sklearn available)

**Use Cases**: Understanding variable relationships, feature selection for modeling.

### 8. Error Analysis (`error_analysis.png`)
**Purpose**: Failure pattern analysis and error condition identification.

**Contents**:
- **Panel A**: Success rate by recommender with value labels
- **Panel B**: Error rate heatmap by gap (if available)
- **Panel C**: Overall status distribution pie chart
- **Panel D**: Time-to-failure analysis comparing successful vs failed runs

**Use Cases**: System reliability assessment, failure mode identification.

### 9. Temporal Analysis (`temporal_analysis.png`)
**Purpose**: Time-based analysis of experimental runs and performance trends.

**Contents**:
- **Panel A**: Experiment frequency over time
- **Panel B**: Performance trends over time
- **Panel C**: Execution time trends (log scale)
- **Panel D**: Daily experiment summary with dual y-axes

**Use Cases**: System stability analysis, performance degradation detection.

### 10. Statistical Summary (`statistical_summary_plot.png`)
**Purpose**: Comprehensive statistical analysis and hypothesis testing.

**Contents**:
- **Panel A**: Summary statistics table
- **Panel B**: Effect sizes (Cohen's d) between recommenders
- **Panel C**: 95% confidence intervals with value labels
- **Panel D**: Statistical power analysis simulation

**Use Cases**: Formal statistical reporting, effect size communication.

## Output Formats

### Individual PNG Files
- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with white background
- **Size**: Customizable (default 12×8 inches)
- **Use**: Journal submissions, presentations

### Combined PDF
- **File**: `publication_figures.pdf`
- **Contents**: All visualizations in a single document
- **Use**: Reports, supplementary materials

### Interactive Dashboard
- **File**: `visualization_dashboard.html`
- **Contents**: Overview of all visualizations with descriptions
- **Features**: Direct links to individual files, usage guide
- **Use**: Quick review, sharing with collaborators

## Customization Options

### Figure Size and Resolution
```python
analyzer.create_publication_visualizations(
    figsize=(10, 6),  # Width × Height in inches
    dpi=600,          # Higher resolution for journals
)
```

### Output Control
```python
analyzer.create_publication_visualizations(
    save_pdf=False,        # Skip combined PDF
    save_individual=True,  # Only individual files
    output_dir="results/experiment/experiment_results_analysis/analysis/custom_figs"   # Custom directory
)
```

### Style Customization
```python
# Access the style settings
analyzer._set_publication_style()

# Modify matplotlib settings
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14  # Larger font
plt.rcParams['figure.dpi'] = 150  # Higher display resolution
```

## Publication Standards

All visualizations follow these standards:

### Typography
- **Font**: Times New Roman (serif) for publication compatibility
- **Sizes**: Title 16pt, axes labels 12pt, tick labels 10pt
- **Weight**: Bold titles, normal text

### Layout
- **Margins**: Tight bounding boxes for space efficiency
- **Grid**: Subtle grids (30% alpha) for readability
- **Colors**: Color-blind friendly palettes (Seaborn 'husl')
- **Background**: White for publication requirements

### Statistical Elements
- **Significance**: Automatic p-value annotations (* ** ***)
- **Confidence Intervals**: 95% CI bars with caps
- **Effect Sizes**: Cohen's d calculations and interpretation
- **Sample Sizes**: Reported in figure captions or annotations

## Usage Examples

### Basic Publication Workflow
```python
# 1. Load and analyze data
analyzer = ExperimentAnalyzer("experiment_results")
results_df = analyzer.load_all_results()

# 2. Create all visualizations
saved_files = analyzer.create_publication_visualizations(
    output_dir="results/experiment/experiment_results_analysis/analysis/paper_figures",
    dpi=300
)

# 3. Generate reports
analyzer.create_detailed_comparison_report("results/experiment/experiment_results_analysis/analysis/stats_report.html")
```

### Conference Presentation

```python
# Larger figures for projection
analyzer.create_publication_visualizations(
    figsize=(14, 10),  # Bigger for visibility
    output_dir="results/experiment/experiment_results_analysis/analysis/presentation_figures"
)
```

### Journal Submission

```python
# High resolution, specific format
analyzer.create_publication_visualizations(
    dpi=600,                    # Journal requirement
    save_pdf=False,             # Individual files only
    figsize=(8.5, 11),         # Letter size
    output_dir="results/experiment/experiment_results_analysis/analysis/journal_figs"
)
```

### Custom Analysis

```python
# Create individual visualizations
fig = analyzer._create_accuracy_comparison(figsize=(10, 6))
fig.suptitle('My Custom Title')
fig.savefig('results/experiment/experiment_results_analysis/analysis/custom_accuracy.png', dpi=300, bbox_inches='tight')
plt.close(fig)
```

## Dependencies

Required packages (already in `pyproject.toml`):
```
matplotlib >= 3.10.1
seaborn >= 0.13.2
pandas >= 2.2.3
numpy >= 1.24.0
scipy >= 1.10.0  # For statistical tests
plotly >= 5.0.0  # For interactive features
```

Optional for enhanced features:
```
scikit-learn  # For feature importance analysis
```

## Integration with Existing Workflow

The publication visualizations integrate seamlessly with existing `ExperimentAnalyzer` functionality:

```python
# Standard workflow
analyzer = ExperimentAnalyzer("results")
results_df = analyzer.load_all_results()

# Existing features still work
analyzer.generate_performance_dashboard("results/experiment/experiment_results_analysis/analysis/dashboard.html")
stats = analyzer.create_statistical_summary()

# New: Add publication visualizations
viz_files = analyzer.create_publication_visualizations()

# Everything is compatible
```

## Troubleshooting

### Common Issues

**Issue**: "No data loaded" error
**Solution**: Call `load_all_results()` before creating visualizations

**Issue**: Empty visualizations  
**Solution**: Check that required columns exist (e.g., `eval_overlap_accuracy`)

**Issue**: LaTeX rendering errors
**Solution**: Set `plt.rcParams['text.usetex'] = False` in `_set_publication_style()`

**Issue**: Out of memory for large datasets
**Solution**: Filter data before visualization or increase system memory

### Performance Tips

- Use `include_tuple_analysis=False` for faster loading
- Filter to specific sessions/recommenders if dataset is large
- Close figures explicitly with `plt.close(fig)` to free memory
- Use `dpi=150` for drafts, `dpi=300` for final figures

## Examples Directory

See these files for complete examples:

- `demo_publication_visualizations.py` - Complete workflow demonstration
- `experiment_analyzer.py` - Main implementation with docstrings
- Example output in `results/experiment/experiment_results_analysis/analysis/` (after running demo)

## Citation

When using these visualizations in publications, consider citing:

```text
The experimental analysis and visualizations were generated using a custom
ExperimentAnalyzer system with publication-ready visualization capabilities,
implementing statistical best practices for recommender system evaluation.
```

## Future Enhancements

Planned additions:

- [ ] Animation support for temporal data
- [ ] Interactive plotly versions of all static plots  
- [ ] LaTeX table generation for statistics
- [ ] Automatic figure caption generation
- [ ] Integration with academic plotting libraries (matplotlib + seaborn + plotly)
- [ ] Support for multi-language figure labels
