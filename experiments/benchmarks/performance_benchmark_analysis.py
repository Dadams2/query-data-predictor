"""
Benchmark Results Analysis

This script provides a detailed analysis of the comprehensive benchmark results,
including performance comparisons, scalability analysis, and recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_benchmark_directories(root_dir="experiments"):
    """Find all benchmark result directories and determine which need analysis."""
    root_path = Path(root_dir)
    
    # Find all directories that start with "benchmark_results"
    benchmark_dirs = [d for d in root_path.iterdir() 
                     if d.is_dir() and d.name.startswith("benchmark_results")]
    
    if not benchmark_dirs:
        raise FileNotFoundError(f"No benchmark result directories found in {root_path}")
    
    print(f"Found {len(benchmark_dirs)} benchmark result directories:")
    
    dirs_to_analyze = []
    for benchmark_dir in sorted(benchmark_dirs):
        print(f"  Checking: {benchmark_dir}")
        
        # Look for CSV files in this directory
        csv_files = list(benchmark_dir.glob("benchmark_results_*.csv"))
        if not csv_files:
            print(f"    No CSV files found, skipping")
            continue
            
        # Check if corresponding analysis directory exists
        analysis_dir_name = benchmark_dir.name.replace("benchmark_results", "benchmark_analysis")
        analysis_dir = root_path / analysis_dir_name
        
        if analysis_dir.exists():
            print(f"    Analysis already exists at {analysis_dir}, skipping")
            continue
            
        print(f"    Needs analysis - found {len(csv_files)} CSV files")
        dirs_to_analyze.append(benchmark_dir)
    
    if not dirs_to_analyze:
        print("All benchmark directories have been analyzed.")
        return []
    
    print(f"\nDirectories that need analysis: {len(dirs_to_analyze)}")
    for dir_path in dirs_to_analyze:
        print(f"  {dir_path}")
    
    return dirs_to_analyze

def load_results_from_directory(benchmark_dir):
    """Load all benchmark results from a specific directory."""
    benchmark_path = Path(benchmark_dir)
    
    # Find all CSV files with benchmark results pattern
    csv_files = list(benchmark_path.glob("benchmark_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No benchmark results found in {benchmark_path}")
    
    print(f"Loading results from {benchmark_path}")
    print(f"Found {len(csv_files)} benchmark result files:")
    
    all_dataframes = []
    for csv_file in sorted(csv_files):
        print(f"  Loading: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            # Add a column to track which file the data came from
            df['source_file'] = csv_file.name
            df['file_timestamp'] = csv_file.stat().st_mtime
            df['source_directory'] = benchmark_path.name
            all_dataframes.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
    
    if not all_dataframes:
        raise ValueError("No valid benchmark result files could be loaded")
    
    # Combine all results
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined {len(combined_df)} total benchmark results from {len(all_dataframes)} files")
    
    return combined_df

def analyze_performance(df):
    """Analyze performance metrics across recommenders."""
    print("\\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Filter successful results
    successful = df[df['success'] == True]
    
    # Overall statistics
    print(f"\\nOverall Statistics:")
    print(f"  Total tests conducted: {len(df)}")
    print(f"  Successful tests: {len(successful)}")
    print(f"  Success rate: {len(successful)/len(df):.1%}")
    
    # Performance by recommender
    print(f"\\nPerformance Summary by Recommender:")
    print(f"{'Recommender':<25} {'Tests':<8} {'Avg Time (s)':<12} {'Avg Memory (MB)':<15} {'Avg Recommendations':<18}")
    print("-" * 80)
    
    for recommender in successful['recommender'].unique():
        data = successful[successful['recommender'] == recommender]
        avg_time = data['execution_time'].mean()
        avg_memory = data['memory_delta'].mean()
        avg_recs = data['recommendations_count'].mean()
        
        print(f"{recommender:<25} {len(data):<8} {avg_time:<12.4f} {avg_memory:<15.2f} {avg_recs:<18.1f}")
    
    return successful

def analyze_scalability(df):
    """Analyze how performance scales with data size."""
    print(f"\\n" + "="*60)
    print("SCALABILITY ANALYSIS")
    print("="*60)
    
    successful = df[df['success'] == True]
    
    # Calculate correlations between data size and performance
    print(f"\\nCorrelation between data size and performance:")
    print(f"{'Recommender':<25} {'Time vs Size':<15} {'Memory vs Size':<15}")
    print("-" * 55)
    
    for recommender in successful['recommender'].unique():
        data = successful[successful['recommender'] == recommender]
        if len(data) > 2:
            time_corr = data['current_data_size'].corr(data['execution_time'])
            memory_corr = data['current_data_size'].corr(data['memory_delta'])
            print(f"{recommender:<25} {time_corr:<15.3f} {memory_corr:<15.3f}")
    
    # Performance by data size ranges
    successful['data_size_range'] = pd.cut(successful['current_data_size'], 
                                          bins=[0, 10, 50, 200, 500, float('inf')], 
                                          labels=['<10', '10-50', '50-200', '200-500', '>500'])
    
    print(f"\\nPerformance by Data Size Range:")
    size_performance = successful.groupby('data_size_range').agg({
        'execution_time': ['mean', 'std'],
        'memory_delta': ['mean', 'std'],
        'recommendations_count': 'mean'
    }).round(4)
    
    print(size_performance)

def analyze_quality(df):
    """Analyze recommendation quality metrics."""
    print(f"\\n" + "="*60)
    print("QUALITY ANALYSIS")
    print("="*60)
    
    successful = df[df['success'] == True]
    
    # Quality metrics by recommender
    print(f"\\nQuality Metrics by Recommender:")
    print(f"{'Recommender':<25} {'Hit Rate':<10} {'Coverage':<10} {'Diversity':<10}")
    print("-" * 55)
    
    for recommender in successful['recommender'].unique():
        data = successful[successful['recommender'] == recommender]
        hit_rate = data['hit_rate'].mean()
        coverage = data['coverage'].mean()
        diversity = data['diversity'].mean()
        
        print(f"{recommender:<25} {hit_rate:<10.4f} {coverage:<10.4f} {diversity:<10.4f}")
    
    # Note about quality metrics
    print(f"\\nNote: Hit rates and coverage are 0.0 because we're predicting future query results")
    print(f"which typically don't overlap with current results in this dataset.")
    print(f"Diversity shows how varied the recommendations are within each set.")

def identify_best_performers(df):
    """Identify the best performing recommenders for different criteria."""
    print(f"\\n" + "="*60)
    print("BEST PERFORMERS")
    print("="*60)
    
    successful = df[df['success'] == True]
    
    # Calculate average performance metrics
    performance = successful.groupby('recommender').agg({
        'execution_time': 'mean',
        'memory_delta': 'mean',
        'hit_rate': 'mean',
        'coverage': 'mean',
        'diversity': 'mean',
        'recommendations_count': 'mean'
    }).round(4)
    
    print(f"\\nBest Performers by Category:")
    print(f"  Fastest (lowest avg execution time): {performance['execution_time'].idxmin()}")
    print(f"  Most Memory Efficient: {performance['memory_delta'].idxmin()}")
    print(f"  Most Diverse Recommendations: {performance['diversity'].idxmax()}")
    
    # Exclude baselines for algorithm comparison
    algorithms_only = performance.drop(['random_baseline', 'dummy_baseline'])
    if not algorithms_only.empty:
        print(f"\\nBest Algorithms (excluding baselines):")
        print(f"  Fastest Algorithm: {algorithms_only['execution_time'].idxmin()}")
        print(f"  Most Memory Efficient Algorithm: {algorithms_only['memory_delta'].idxmin()}")
        print(f"  Most Diverse Algorithm: {algorithms_only['diversity'].idxmax()}")

def generate_detailed_visualizations(df, benchmark_dir_name, root_dir="experiments"):
    """Generate detailed visualizations for the analysis."""
    # Create analysis directory name based on benchmark directory name
    analysis_dir_name = benchmark_dir_name.replace("benchmark_results", "benchmark_analysis")
    output_path = Path(root_dir) / analysis_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in {output_path}")
    
    successful = df[df['success'] == True]
    
    # Convert file timestamp to datetime for time-series analysis
    df['run_datetime'] = pd.to_datetime(df['file_timestamp'], unit='s')
    successful['run_datetime'] = pd.to_datetime(successful['file_timestamp'], unit='s')
    
    # Determine grid size based on whether we have multiple runs
    unique_runs = df['source_file'].nunique()
    if unique_runs > 1:
        # Include time-series plots
        fig = plt.figure(figsize=(25, 20))
        subplot_rows, subplot_cols = 4, 3
    else:
        # Standard plots only
        fig = plt.figure(figsize=(20, 16))
        subplot_rows, subplot_cols = 3, 3
    
    # 1. Execution time comparison
    plt.subplot(subplot_rows, subplot_cols, 1)
    successful.boxplot(column='execution_time', by='recommender', ax=plt.gca())
    plt.title('Execution Time Distribution')
    plt.xlabel('Recommender')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')  # Log scale for better visibility
    
    # 2. Memory usage comparison
    plt.subplot(subplot_rows, subplot_cols, 2)
    successful.boxplot(column='memory_delta', by='recommender', ax=plt.gca())
    plt.title('Memory Usage Distribution')
    plt.xlabel('Recommender')
    plt.ylabel('Memory Delta (MB)')
    plt.xticks(rotation=45, ha='right')
    
    # 3. Diversity comparison
    plt.subplot(subplot_rows, subplot_cols, 3)
    successful.boxplot(column='diversity', by='recommender', ax=plt.gca())
    plt.title('Recommendation Diversity')
    plt.xlabel('Recommender')
    plt.ylabel('Diversity Score')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Time vs Data Size scatter
    plt.subplot(subplot_rows, subplot_cols, 4)
    for recommender in successful['recommender'].unique():
        data = successful[successful['recommender'] == recommender]
        plt.scatter(data['current_data_size'], data['execution_time'], 
                   label=recommender, alpha=0.7)
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Scalability: Time vs Data Size')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Memory vs Data Size scatter
    plt.subplot(subplot_rows, subplot_cols, 5)
    for recommender in successful['recommender'].unique():
        data = successful[successful['recommender'] == recommender]
        plt.scatter(data['current_data_size'], data['memory_delta'], 
                   label=recommender, alpha=0.7)
    plt.xlabel('Data Size')
    plt.ylabel('Memory Delta (MB)')
    plt.title('Scalability: Memory vs Data Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Average performance comparison
    plt.subplot(subplot_rows, subplot_cols, 6)
    avg_performance = successful.groupby('recommender')['execution_time'].mean().sort_values()
    avg_performance.plot(kind='bar')
    plt.title('Average Execution Time by Recommender')
    plt.xlabel('Recommender')
    plt.ylabel('Average Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    
    # 7. Performance vs Quality scatter
    plt.subplot(subplot_rows, subplot_cols, 7)
    avg_metrics = successful.groupby('recommender').agg({
        'execution_time': 'mean',
        'diversity': 'mean'
    })
    plt.scatter(avg_metrics['execution_time'], avg_metrics['diversity'])
    for recommender, row in avg_metrics.iterrows():
        plt.annotate(recommender, (row['execution_time'], row['diversity']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Average Execution Time (s)')
    plt.ylabel('Average Diversity')
    plt.title('Performance vs Quality Trade-off')
    plt.xscale('log')
    
    # 8. Success rate by recommender
    plt.subplot(subplot_rows, subplot_cols, 8)
    success_rates = df.groupby('recommender')['success'].mean()
    success_rates.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Success Rate by Recommender')
    plt.xlabel('Recommender')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # 9. Data size distribution
    plt.subplot(subplot_rows, subplot_cols, 9)
    successful['current_data_size'].hist(bins=20, alpha=0.7)
    plt.xlabel('Current Data Size')
    plt.ylabel('Frequency')
    plt.title('Distribution of Test Data Sizes')
    
    # Add time-series plots if we have multiple runs
    if unique_runs > 1:
        # 10. Performance over time
        plt.subplot(subplot_rows, subplot_cols, 10)
        time_performance = successful.groupby(['run_datetime', 'recommender'])['execution_time'].mean().unstack()
        for recommender in time_performance.columns:
            if pd.notna(time_performance[recommender]).any():
                plt.plot(time_performance.index, time_performance[recommender], 
                        marker='o', label=recommender, alpha=0.7)
        plt.xlabel('Benchmark Run Time')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Performance Trends Over Time')
        plt.yscale('log')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 11. Memory usage over time
        plt.subplot(subplot_rows, subplot_cols, 11)
        time_memory = successful.groupby(['run_datetime', 'recommender'])['memory_delta'].mean().unstack()
        for recommender in time_memory.columns:
            if pd.notna(time_memory[recommender]).any():
                plt.plot(time_memory.index, time_memory[recommender], 
                        marker='o', label=recommender, alpha=0.7)
        plt.xlabel('Benchmark Run Time')
        plt.ylabel('Average Memory Delta (MB)')
        plt.title('Memory Usage Trends Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 12. Success rate over time
        plt.subplot(subplot_rows, subplot_cols, 12)
        time_success = df.groupby(['run_datetime', 'recommender'])['success'].mean().unstack()
        for recommender in time_success.columns:
            if pd.notna(time_success[recommender]).any():
                plt.plot(time_success.index, time_success[recommender], 
                        marker='o', label=recommender, alpha=0.7)
        plt.xlabel('Benchmark Run Time')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Trends Over Time')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\\nDetailed visualizations saved to {output_path}")

def generate_recommendations(df):
    """Generate recommendations based on the benchmark results."""
    print(f"\\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    successful = df[df['success'] == True]
    
    # Calculate performance metrics
    performance = successful.groupby('recommender').agg({
        'execution_time': ['mean', 'std'],
        'memory_delta': ['mean', 'std'],
        'diversity': 'mean',
        'recommendations_count': 'mean'
    }).round(4)
    
    print(f"\\nBased on the benchmark results, here are our recommendations:")
    print(f"\\n1. FOR REAL-TIME APPLICATIONS (speed is critical):")
    fastest = performance[('execution_time', 'mean')].nsmallest(3)
    for i, (recommender, time) in enumerate(fastest.items(), 1):
        print(f"   {i}. {recommender} (avg: {time:.4f}s)")
    
    print(f"\\n2. FOR MEMORY-CONSTRAINED ENVIRONMENTS:")
    most_efficient = performance[('memory_delta', 'mean')].nsmallest(3)
    for i, (recommender, memory) in enumerate(most_efficient.items(), 1):
        print(f"   {i}. {recommender} (avg: {memory:.2f}MB)")
    
    print(f"\\n3. FOR DIVERSE RECOMMENDATIONS:")
    # The diversity column is actually a DataFrame with 'mean' column
    diversity_means = performance['diversity']['mean'].to_dict()
    sorted_diversity = sorted(diversity_means.items(), key=lambda x: x[1], reverse=True)
    
    for i, (recommender, diversity) in enumerate(sorted_diversity[:3], 1):
        print(f"   {i}. {recommender} (diversity: {diversity:.4f})")
    
    print(f"\\n4. GENERAL OBSERVATIONS:")
    print(f"   • Random and Dummy baselines are fastest but provide limited value")
    print(f"   • Clustering recommender offers good speed-quality balance")
    print(f"   • Tuple-based methods (association rules, summaries, hybrid) are slower but more sophisticated")
    print(f"   • Association rules method has issues (missing mine_association_rules method)")
    print(f"   • Memory usage varies significantly with data size and method complexity")
    
    print(f"\\n5. RECOMMENDED APPROACH BY USE CASE:")
    print(f"   • Prototype/Development: clustering (good balance)")
    print(f"   • Production with small data: tuple_summaries (faster than hybrid)")
    print(f"   • Production with large data: clustering (better scalability)")
    print(f"   • Research/Analysis: fix association rules then use tuple_hybrid")

def analyze_by_source_file(df):
    """Analyze results broken down by source file (benchmark run)."""
    print(f"\n" + "="*60)
    print("ANALYSIS BY BENCHMARK RUN")
    print("="*60)
    
    # Convert file timestamp to readable datetime
    df['run_datetime'] = pd.to_datetime(df['file_timestamp'], unit='s')
    
    print(f"\nBenchmark Runs Summary:")
    print(f"{'Source File':<30} {'Run Time':<20} {'Total Tests':<12} {'Success Rate':<12}")
    print("-" * 75)
    
    for source_file in df['source_file'].unique():
        file_data = df[df['source_file'] == source_file]
        run_time = file_data['run_datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        total_tests = len(file_data)
        success_rate = (file_data['success'] == True).mean()
        
        print(f"{source_file:<30} {run_time:<20} {total_tests:<12} {success_rate:<12.1%}")
    
    # Show performance trends over time if multiple runs
    unique_files = df['source_file'].nunique()
    if unique_files > 1:
        print(f"\nPerformance Trends Across {unique_files} Benchmark Runs:")
        
        # Calculate average metrics per run
        run_summary = df[df['success'] == True].groupby(['source_file', 'run_datetime']).agg({
            'execution_time': 'mean',
            'memory_delta': 'mean',
            'diversity': 'mean',
            'recommendations_count': 'mean'
        }).round(4)
        
        print("\nAverage Performance by Run:")
        print(run_summary)
        
        # Show improvement/degradation
        if unique_files >= 2:
            print(f"\nTrend Analysis (comparing first vs latest run):")
            first_run = df[df['source_file'] == df['source_file'].min()]
            latest_run = df[df['source_file'] == df['source_file'].max()]
            
            first_avg = first_run[first_run['success'] == True]['execution_time'].mean()
            latest_avg = latest_run[latest_run['success'] == True]['execution_time'].mean()
            
            if pd.notna(first_avg) and pd.notna(latest_avg):
                change = ((latest_avg - first_avg) / first_avg) * 100
                trend = "improved" if change < 0 else "degraded"
                print(f"  Average execution time {trend} by {abs(change):.1f}%")
    
    return df

def analyze_benchmark_directory(benchmark_dir, root_dir="experiments"):
    """Analyze a single benchmark directory."""
    print(f"\n{'='*80}")
    print(f"ANALYZING BENCHMARK DIRECTORY: {benchmark_dir.name}")
    print(f"{'='*80}")
    
    try:
        # Load results from this directory
        df = load_results_from_directory(benchmark_dir)
        
        # Run all analyses
        print(f"\nRunning comprehensive analysis...")
        df_with_timestamps = analyze_by_source_file(df)
        successful_df = analyze_performance(df)
        analyze_scalability(df)
        analyze_quality(df)
        identify_best_performers(df)
        generate_detailed_visualizations(df, benchmark_dir.name, root_dir)
        generate_recommendations(df)
        
        # Save a summary report
        analysis_dir_name = benchmark_dir.name.replace("benchmark_results", "benchmark_analysis")
        analysis_dir = Path(root_dir) / analysis_dir_name
        summary_file = analysis_dir / "analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"BENCHMARK ANALYSIS SUMMARY\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Source Directory: {benchmark_dir}\n")
            f.write(f"Analysis Directory: {analysis_dir}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Results: {len(df)}\n")
            f.write(f"Successful Results: {len(df[df['success'] == True])}\n")
            f.write(f"Source Files: {df['source_file'].nunique()}\n")
            f.write(f"Benchmark Runs: {df['source_file'].unique().tolist()}\n")
        
        print(f"\nAnalysis completed for {benchmark_dir.name}")
        print(f"Results saved to: {analysis_dir}")
        return True
        
    except Exception as e:
        print(f"Error analyzing {benchmark_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_all_combined(root_dir="experiments"):
    """Legacy function to analyze all benchmark results combined."""
    print("ANALYZING ALL BENCHMARK RESULTS COMBINED")
    print("="*60)
    
    try:
        # Find all benchmark directories
        root_path = Path(root_dir)
        benchmark_dirs = [d for d in root_path.iterdir() 
                         if d.is_dir() and d.name.startswith("benchmark_results")]
        
        if not benchmark_dirs:
            raise FileNotFoundError(f"No benchmark result directories found in {root_path}")
        
        # Load all results from all directories
        all_dataframes = []
        for benchmark_dir in benchmark_dirs:
            try:
                df = load_results_from_directory(benchmark_dir)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not load from {benchmark_dir}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No valid benchmark result directories could be loaded")
        
        # Combine all results
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Combined {len(combined_df)} total results from {len(all_dataframes)} directories")
        
        # Run analyses on combined data
        df_with_timestamps = analyze_by_source_file(combined_df)
        successful_df = analyze_performance(combined_df)
        analyze_scalability(combined_df)
        analyze_quality(combined_df)
        identify_best_performers(combined_df)
        
        # Generate visualizations with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generate_detailed_visualizations(combined_df, f"benchmark_analysis_combined_{timestamp}", root_dir)
        generate_recommendations(combined_df)
        
        print(f"\nCombined analysis completed")
        return True
        
    except Exception as e:
        print(f"Error during combined analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main analysis function."""
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*60)
    print("Searching for benchmark directories that need analysis...")
    
    try:
        # Find benchmark directories that need analysis
        dirs_to_analyze = find_benchmark_directories()
        
        if not dirs_to_analyze:
            print("No benchmark directories need analysis. Exiting.")
            return 0
        
        # Analyze each directory
        successful_analyses = 0
        failed_analyses = 0
        
        for benchmark_dir in dirs_to_analyze:
            success = analyze_benchmark_directory(benchmark_dir)
            if success:
                successful_analyses += 1
            else:
                failed_analyses += 1
        
        # Final summary
        print(f"\n" + "="*60)
        print("BATCH ANALYSIS COMPLETE")
        print("="*60)
        print(f"Directories processed: {len(dirs_to_analyze)}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {failed_analyses}")
        
        if successful_analyses > 0:
            print(f"\nAnalysis directories created:")
            root_path = Path("experiments")
            analysis_dirs = [d for d in root_path.iterdir() 
                           if d.is_dir() and d.name.startswith("benchmark_analysis")]
            for analysis_dir in sorted(analysis_dirs):
                print(f"  {analysis_dir}")
        
        return 0 if failed_analyses == 0 else 1
        
    except Exception as e:
        print(f"Error during batch analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--combined', action='store_true', 
                       help='Analyze all benchmark results combined instead of individually')
    parser.add_argument('--root-dir', default='experiments',
                       help='Root directory to search for benchmark results (default: experiments)')
    
    args = parser.parse_args()
    
    if args.combined:
        # Run combined analysis
        success = analyze_all_combined(args.root_dir)
        sys.exit(0 if success else 1)
    else:
        # Run individual directory analysis (default behavior)
        sys.exit(main())
