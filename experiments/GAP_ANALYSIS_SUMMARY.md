# Gap Analysis Visualizations Summary

## Overview
The experiment analyzer now provides comprehensive gap analysis visualizations for all performance metrics. Each metric has its own dedicated gap analysis chart showing how query gap affects recommender performance.

## Available Gap Analysis Charts

### 1. General Gap Analysis (`gap_analysis.png`)
- **Method**: `_create_gap_analysis()`
- **Description**: Multi-metric overview showing how query gap affects multiple performance metrics
- **Features**:
  - Line plots with error bars for primary metric vs gap
  - Heatmap of metric performance by recommender and gap
  - Multi-metric comparison for baseline gap
  - Effect size analysis for F1 score (or first available metric)

### 2. Precision Gap Analysis (`precision_gap_analysis.png`)
- **Method**: `_create_precision_gap_analysis()`
- **Metric**: Precision
- **Features**:
  - Line plot of precision vs query gap with error bars
  - Heatmap showing precision across recommenders and gaps
  - Box plots showing precision distribution by gap
  - Effect size analysis comparing gaps to baseline

### 3. Recall Gap Analysis (`recall_gap_analysis.png`)
- **Method**: `_create_recall_gap_analysis()`
- **Metric**: Recall
- **Features**:
  - Line plot of recall vs query gap with error bars
  - Heatmap showing recall across recommenders and gaps
  - Box plots showing recall distribution by gap
  - Effect size analysis comparing gaps to baseline

### 4. F1 Score Gap Analysis (`f1_gap_analysis.png`)
- **Method**: `_create_f1_gap_analysis()`
- **Metric**: F1 Score
- **Features**:
  - Line plot of F1 score vs query gap with error bars
  - Heatmap showing F1 score across recommenders and gaps
  - Box plots showing F1 score distribution by gap
  - Effect size analysis comparing gaps to baseline

### 5. ROC-AUC Gap Analysis (`roc_auc_gap_analysis.png`)
- **Method**: `_create_roc_auc_gap_analysis()`
- **Metric**: ROC-AUC
- **Features**:
  - Line plot of ROC-AUC vs query gap with error bars
  - Heatmap showing ROC-AUC across recommenders and gaps
  - Box plots showing ROC-AUC distribution by gap
  - Effect size analysis comparing gaps to baseline

## Visualization Structure

Each individual gap analysis chart contains 4 subplots:

### A) Metric vs Query Gap (Top Left)
- Line plot with error bars
- Shows mean metric value for each gap
- Different lines for each recommender system
- Error bars show standard deviation

### B) Metric Heatmap (Top Right)
- Heatmap visualization
- Rows: Recommender systems
- Columns: Query gap values
- Color intensity: Mean metric value
- Annotations show exact values

### C) Distribution by Gap (Bottom Left)
- Box plots showing metric distribution
- X-axis: Query gap values
- Y-axis: Metric values
- Different colors for each recommender
- Shows quartiles, outliers, and medians

### D) Gap Effect Size (Bottom Right)
- Effect size heatmap
- Compares each gap to baseline (minimum gap)
- Effect size = (gap_mean - baseline_mean) / baseline_std
- Positive values: improvement over baseline
- Negative values: degradation from baseline
- Color scale: Red (negative) to Blue (positive)

## Usage

### Generating All Gap Analysis Charts
```python
from experiment_analyzer import ExperimentAnalyzer

analyzer = ExperimentAnalyzer('path/to/experiment/data')
analyzer.load_all_results()

# Generate all visualizations including gap analysis charts
saved_files = analyzer.create_publication_visualizations()

# Files will be saved as:
# - gap_analysis.png
# - precision_gap_analysis.png
# - recall_gap_analysis.png
# - f1_gap_analysis.png
# - roc_auc_gap_analysis.png
```

### Generating Individual Charts
```python
# Generate specific gap analysis chart
fig = analyzer._create_precision_gap_analysis()
fig.savefig('custom_precision_gap.png', dpi=300, bbox_inches='tight')
```

## Dashboard Integration

All gap analysis charts are included in the HTML visualization dashboard:
- Interactive preview of each chart
- Download links for high-resolution PNG files
- Descriptions of what each chart shows
- Usage guidelines for publications

## Publication Ready Features

- **High DPI**: 300 DPI resolution for publication quality
- **Professional Styling**: Publication-ready fonts and formatting
- **Statistical Rigor**: Error bars, effect sizes, and significance testing
- **Clear Labels**: Descriptive titles, axis labels, and legends
- **Color Schemes**: Colorblind-friendly palettes
- **Annotations**: Values displayed on heatmaps for precise reading

## Interpretation Guidelines

### Line Plots (Panel A)
- **Steep slopes**: Large sensitivity to query gap
- **Flat lines**: Robust performance across gaps
- **Error bars**: Variability in performance
- **Line separation**: Differences between recommenders

### Heatmaps (Panel B)
- **Color intensity**: Performance level
- **Patterns**: Consistent trends across systems
- **Hotspots**: Best performing conditions
- **Cold spots**: Challenging conditions

### Box Plots (Panel C)
- **Box width**: Distribution spread
- **Median lines**: Central tendency
- **Outliers**: Unusual performance cases
- **Overlap**: Similar performance ranges

### Effect Size Heatmaps (Panel D)
- **Red cells**: Performance degradation
- **Blue cells**: Performance improvement
- **White cells**: No significant change
- **Intensity**: Magnitude of effect

## Data Requirements

For gap analysis visualizations to work:
1. **Required columns**: `meta_gap`, `meta_recommender_name`
2. **Metric columns**: `eval_precision`, `eval_recall`, `eval_f1_score`, `eval_roc_auc`
3. **Minimum data**: At least 2 gap values and 2 recommenders
4. **Statistical validity**: Multiple measurements per condition

## Error Handling

- **Missing metrics**: Charts gracefully handle missing metric columns
- **Insufficient data**: Shows informative message when data is inadequate
- **Edge cases**: Handles single recommenders or gaps appropriately
- **Error logging**: Detailed logs for troubleshooting issues

The gap analysis system provides comprehensive insights into how query gap affects different aspects of recommender performance, enabling researchers to understand robustness, sensitivity, and optimal operating conditions for their systems.
