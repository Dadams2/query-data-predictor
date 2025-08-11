# Enhanced Experimental System for Recommender Evaluation

This directory contains an enhanced experimental data collection and analysis system for evaluating recommender systems.

## Architecture Overview

The system consists of three main components:

1. **Enhanced Experiment Collector** (`enhanced_experiment_collector.py`)
   - Structured data collection with full provenance
   - Support for multiple data formats (JSON, Parquet, Pickle)
   - Comprehensive metadata tracking
   - State management for stateful recommenders

2. **Enhanced Experiment Runner** (`enhanced_recommender_experiments.py`)
   - Integration with existing recommender framework  
   - Enhanced data collection during evaluation
   - Comprehensive metric calculation
   - Error handling and timeout management

3. **Enhanced Analyzer** (`enhanced_experiment_analyzer.py`)
   - Interactive visualizations with Plotly
   - Statistical significance testing
   - Comparative analysis across recommenders
   - Export capabilities for further analysis

## Key Features

### Data Collection
- **Full Provenance**: Every experiment includes complete context
- **Flexible Storage**: JSONL for streaming, Parquet for analysis
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, Jaccard similarity
- **Error Tracking**: Failed experiments are recorded for analysis
- **State Tracking**: Optional recommender state snapshots

### Data Structure
The system stores experimental results in a structured hierarchy:

```
experiment_data/
├── metadata/           # JSON files with experiment metadata
├── results/           # JSONL stream for real-time analysis
├── tuples/           # Parquet files with actual tuple data
├── context/          # Configuration and query context
├── analysis/         # Generated analysis results
└── state_snapshots/  # Recommender state snapshots (optional)
```

### Analysis Capabilities
- **Interactive Dashboards**: Web-based performance visualization
- **Statistical Testing**: ANOVA, t-tests, effect sizes
- **Correlation Analysis**: Relationship between metrics
- **Trend Analysis**: Performance over time and query position
- **Export Options**: CSV, Parquet, R scripts, Jupyter notebooks

## Quick Start

### 1. Basic Usage
```python
from enhanced_recommender_experiments import EnhancedRecommenderExperimentRunner

# Initialize runner
runner = EnhancedRecommenderExperimentRunner(
    output_dir="my_experiment_results",
    enable_full_tuple_storage=True
)

# Run experiments
session_id = "11305"  # Your session ID
results = runner.run_enhanced_experiment(
    session_id=session_id,
    max_gap=5,
    include_query_text=True
)

print(f"Completed {results['successful_experiments']} experiments")
```

### 2. Analysis
```python
from enhanced_experiment_analyzer import EnhancedExperimentAnalyzer

# Initialize analyzer
analyzer = EnhancedExperimentAnalyzer("my_experiment_results")

# Load and analyze results
results_df = analyzer.load_all_results()

# Generate interactive dashboard
analyzer.generate_performance_dashboard("dashboard.html")

# Create detailed report
analyzer.create_detailed_comparison_report("report.html")

# Export for further analysis
analyzer.export_for_further_analysis("exports/")
```

## Data Format Recommendations

Based on your requirements, here are the recommended answers to your questions:

### 1. **Best Data Format**
- **Primary**: **JSONL** (JSON Lines) for real-time streaming and easy querying
- **Secondary**: **Parquet** for efficient analysis and columnar operations
- **Complex Objects**: **Pickle** for recommender states and complex metadata

### 2. **Results Structure**
```python
{
  "experiment_id": "unique_experiment_identifier",
  "metadata": {
    "timestamp": "2025-07-02T18:00:00Z",
    "session_id": "source_session_id", 
    "recommender_name": "clustering",
    "gap": 2,
    "status": "completed"
  },
  "current_query_context": {
    "query_position": 5,
    "query_text": "SELECT ...",
    "result_set_size": 1000
  },
  "target_query_context": {
    "query_position": 7,
    "result_set_size": 800
  },
  "recommendation_summary": {
    "predicted_count": 150,
    "actual_count": 800,
    "confidence_available": true
  },
  "evaluation": {
    "overlap_accuracy": 0.75,
    "precision": 0.82,
    "recall": 0.68,
    "f1_score": 0.74,
    "jaccard_similarity": 0.61
  }
}
```

### 3. **What to Write**
- **Always**: Experiment metadata, evaluation metrics, summary statistics
- **Configurable**: Full tuple data (can be expensive for large result sets)
- **Optional**: Confidence scores, intermediate states, query text
- **Recommended**: Tuple hashes for exact matching without storing full data

### 4. **How Data Should Be Written**
- **Streaming**: Append to JSONL files for real-time monitoring
- **Batch**: Periodic conversion to Parquet for efficient analysis
- **Separation**: Metadata separate from large data objects
- **Indexing**: Use experiment IDs to link related data

### 5. **Query Results Context**
**Yes**, store query results context including:
- Query position in sequence
- Query text (if available)
- Result set size
- Execution metadata
- Hash identifiers for deduplication

## Advanced Features

### Future Recommender Support
The system is designed to support stateful recommenders:

```python
# For recommenders that maintain state
class StatefulRecommender:
    def __init__(self):
        self.query_history = []
        self.learned_patterns = {}
    
    def recommend_tuples(self, current_results, query_position=None, session_context=None):
        # Use accumulated knowledge from previous queries
        # The enhanced system can track this state
        pass
```

### Batch vs Online Evaluation
```python
# Batch evaluation (current approach)
results = runner.run_enhanced_experiment(session_id)

# Online evaluation (for streaming/production)
collector = ExperimentCollector()
for query in query_stream:
    # Collect results as they happen
    experiment_id = collector.collect_experiment(...)
```

### Custom Metrics
```python
# Extend the evaluation system
class CustomEvaluationResult(EvaluationResult):
    custom_metric: float
    domain_specific_score: float

# Use in collection
collector.collect_experiment(
    ...,
    evaluation_result=CustomEvaluationResult(...)
)
```

## Integration with Existing Code

The enhanced system is designed to work alongside your existing `recommender_experiments.py`:

1. **Gradual Migration**: You can run both systems in parallel
2. **Data Bridge**: The `collect_recommendation_experiment()` function bridges old and new formats  
3. **Backward Compatibility**: Existing analysis code can still work with CSV exports

## Performance Considerations

- **Tuple Storage**: Disable for very large result sets to save space
- **Streaming**: Use JSONL for real-time monitoring without memory issues
- **Indexing**: Experiment IDs enable efficient data linking
- **Compression**: Parquet files are automatically compressed

## Next Steps

1. **Test Integration**: Start with a small session to verify the system works
2. **Customize Metrics**: Add domain-specific evaluation metrics
3. **Production Deployment**: Set up automated experiment runs
4. **Analysis Workflows**: Create standard analysis procedures for your team

The enhanced system provides a solid foundation for rigorous recommender evaluation with full experimental provenance and powerful analysis capabilities.
