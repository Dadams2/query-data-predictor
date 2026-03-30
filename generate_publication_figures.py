"""
Generate publication-quality figures for the vision paper.

Produces figures referenced in vision-paper/main.tex:
  1. workload_comparison.pdf  — Hero figure: SIMBA drilldown vs adversarial benchmark
  2. gap_analysis.pdf         — F1 across prediction gaps on adversarial benchmark
  3. overlap_analysis.pdf     — Inter-query overlap distributions
  4. decay_sensitivity.pdf    — Temporal decay parameter sensitivity (theoretical)

Usage:
    source .venv/bin/activate
    python generate_publication_figures.py
"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
from query_data_predictor.analysis import ResultsAnalyzer
warnings.filterwarnings('ignore')

# ── Result directories (update timestamps after new runs) ──────────────────
BENCHMARK_RESULTS_DIR = Path("results/benchmark_mdi/benchmark_mdi_redo_20260312-170746")
BENCHMARK_DIR = Path(
    "results/benchmark_mdi/benchmark_mdi_redo_20260312-170746"
    "/analysis/summary/raw/all_sessions_metrics.csv"
)
BENCHMARK_CONFIG = Path("experiments/configs/benchmark_mdi_vs_baselines.yml")
SIMBA_DRILLDOWN_RESULTS_DIR = Path("results/simba_drilldown/simba_drilldown_20260317-105610")
SIMBA_DRILLDOWN_DIR = Path(
    "results/simba_drilldown/simba_drilldown_20260317-105610"
    "/analysis/summary/raw/all_sessions_metrics.csv"
)
SIMBA_DRILLDOWN_CONFIG = Path("experiments/configs/simba_drilldown.yml")
SDSS_RESULTS_DIR = Path("results/sdss_vs_baselines/sdss_vs_baselines_20260325-171351")
SDSS_DIR = Path(
    "results/sdss_vs_baselines/sdss_vs_baselines_20260325-171351"
    "/analysis/summary/raw/all_sessions_metrics.csv"
)
SDSS_CONFIG = Path("experiments/configs/sdss_vs_baselines.yml")

# ── Shared constants ───────────────────────────────────────────────────────
NAME_MAP = {
    'multidimensional_interestingness': 'MDI (Ours)',
    'random': 'Random',
    'clustering': 'Clustering',
    'frequency': 'Frequency',
    'similarity': 'Similarity',
    'sampling': 'Sampling',
}

# Consistent recommender ordering for all plots
RECOMMENDER_ORDER = [
    'multidimensional_interestingness',
    'random',
    'clustering',
    'frequency',
    'similarity',
    'sampling',
]

# Color palette: MDI gets a distinct color, baselines share a family
COLORS = {
    'multidimensional_interestingness': '#2176AE',  # strong blue
    'random': '#B0B0B0',       # gray
    'clustering': '#E8873C',   # orange
    'frequency': '#57A773',    # green
    'similarity': '#D64045',   # red
    'sampling': '#9B72AA',     # purple
}


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.8,
    })


def load_csv(path: Path) -> pd.DataFrame:
    """Load an analysis CSV, fail loudly if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path.parent.parent.parent.name}/.../{path.name}")
    return df


def ensure_analysis_csv(csv_path: Path, results_dir: Path, config_path: Path) -> pd.DataFrame:
    """
    Load an analysis CSV, generating it from raw result JSONs first if needed.
    """
    if csv_path.exists():
        return load_csv(csv_path)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Neither analysis CSV nor results directory exists: {csv_path} / {results_dir}"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found for analysis: {config_path}")

    print(f"  Analysis CSV missing for {results_dir.name}; generating simple analysis...")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    analyzer = ResultsAnalyzer(results_dir=results_dir, config=config_data)
    analyzer.analyze_simple()

    return load_csv(csv_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Hero — Workload Comparison
# ═══════════════════════════════════════════════════════════════════════════

def generate_workload_comparison(
    workloads: list[tuple[pd.DataFrame, str]],
    output_paths: list[Path],
):
    """
    Three-panel bar chart comparing gap=1 F1 across workloads.
    Arranged for vertical layouts: two panels on top, one centered below.
    """
    fig = plt.figure(figsize=(7.0, 4.9))
    axes = [
        fig.add_axes([0.10, 0.62, 0.33, 0.28]),
        fig.add_axes([0.57, 0.62, 0.33, 0.28]),
        fig.add_axes([0.335, 0.14, 0.33, 0.28]),
    ]

    if len(workloads) != 3:
        raise ValueError("generate_workload_comparison expects exactly three workloads")

    ymax = 0.4
    for df, _ in workloads:
        gap1 = df[df['gap'] == 1]
        if gap1.empty:
            continue
        agg = gap1.groupby('recommender')['f1_score'].agg(['mean', 'std']).reset_index()
        if not agg.empty:
            ymax = max(ymax, float((agg['mean'] + agg['std'].fillna(0)).max()) * 1.1)

    panel_labels = ['A', 'B', 'C']

    for ax, (df, title), panel_label in zip(axes, workloads, panel_labels):
        gap1 = df[df['gap'] == 1]
        agg = gap1.groupby('recommender')['f1_score'].agg(['mean', 'std']).reset_index()
        agg['std'] = agg['std'].fillna(0.0)

        # Order recommenders consistently
        present = [r for r in RECOMMENDER_ORDER if r in agg['recommender'].values]
        agg = agg.set_index('recommender').loc[present].reset_index()

        x = np.arange(len(present))
        ax.bar(
            x, agg['mean'], yerr=agg['std'], capsize=3,
            color=[COLORS[r] for r in present],
            edgecolor='white', linewidth=0.5, width=0.7,
            error_kw={'linewidth': 0.8},
        )

        # Baseline mean line (excluding MDI)
        baseline_mask = agg['recommender'] != 'multidimensional_interestingness'
        baseline_mean = agg.loc[baseline_mask, 'mean'].mean()
        ax.axhline(baseline_mean, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([NAME_MAP.get(r, r) for r in present],
                           rotation=28, ha='right', fontsize=7.5)
        ax.set_title(title, fontsize=9.5, pad=8, fontweight='semibold')
        ax.set_ylim(0, min(1.0, ymax))
        ax.set_yticks(np.arange(0, min(1.0, ymax) + 1e-9, 0.2))
        ax.grid(axis='y', alpha=0.22)
        ax.grid(axis='x', visible=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(
            0.0, 1.06, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left'
        )

    for ax in axes:
        ax.margins(x=0.06)

    axes[0].set_ylabel('F1 Score (gap = 1)')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('F1 Score (gap = 1)')

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════

def generate_gap_analysis(df_bench: pd.DataFrame, output_path: Path):
    """
    Line chart: F1 vs prediction gap on the adversarial benchmark.
    MDI bold/solid, baselines dashed.
    """
    fig, ax = plt.subplots(figsize=(5, 3.2))

    agg = df_bench.groupby(['recommender', 'gap'])['f1_score'].agg(
        ['mean', 'std']
    ).reset_index()

    present = [r for r in RECOMMENDER_ORDER if r in agg['recommender'].values]

    for rec in present:
        subset = agg[agg['recommender'] == rec].sort_values('gap')
        is_mdi = rec == 'multidimensional_interestingness'
        ax.plot(
            subset['gap'], subset['mean'],
            marker='o' if is_mdi else 's',
            linestyle='-' if is_mdi else '--',
            linewidth=2.2 if is_mdi else 1.2,
            markersize=6 if is_mdi else 4,
            color=COLORS[rec],
            label=NAME_MAP.get(rec, rec),
            zorder=10 if is_mdi else 5,
        )
        ax.fill_between(
            subset['gap'],
            subset['mean'] - subset['std'],
            subset['mean'] + subset['std'],
            alpha=0.12 if is_mdi else 0.06,
            color=COLORS[rec],
        )

    ax.set_xlabel('Prediction Gap')
    ax.set_ylabel('F1 Score')
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Overlap Distribution
# ═══════════════════════════════════════════════════════════════════════════

def generate_overlap_analysis(df_simba: pd.DataFrame, df_bench: pd.DataFrame,
                               output_path: Path):
    """
    Box plot comparing inter-query overlap distributions between
    SIMBA drilldown and adversarial benchmark.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Use the 'overlap' column (fraction of recommended in future)
    # averaged per (session, gap, query) across recommenders to get a workload-level overlap
    simba_overlap = df_simba.groupby(['session_id', 'gap', 'query_number'])['overlap'].mean().reset_index()
    simba_overlap['Workload'] = 'SIMBA\nDrilldown'

    bench_overlap = df_bench.groupby(['session_id', 'gap', 'query_number'])['overlap'].mean().reset_index()
    bench_overlap['Workload'] = 'Adversarial\nBenchmark'

    combined = pd.concat([simba_overlap, bench_overlap])

    sns.boxplot(
        data=combined, x='Workload', y='overlap', ax=ax,
        palette=['#2176AE', '#D64045'],
        width=0.5, linewidth=0.8, fliersize=2,
    )

    ax.set_ylabel('Mean Overlap Fraction')
    ax.set_xlabel('')
    ax.set_title('Inter-Query Overlap by Workload Type', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Decay Sensitivity (from real sweep experiments)
# ═══════════════════════════════════════════════════════════════════════════

DECAY_SWEEP_DIR = Path("results/decay_sweep")

def generate_decay_sensitivity(output_path: Path):
    """
    Sensitivity curve for temporal decay parameter from real experiments.
    Loads analysis CSVs from the decay sweep directory.
    """
    decay_rates = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    rate_labels = ["0.00", "0.01", "0.05", "0.10", "0.15", "0.20", "0.30", "0.50"]

    means = []
    stds = []
    available_rates = []

    for rate, label in zip(decay_rates, rate_labels):
        # Find the most recent analysis directory for this decay rate
        rate_dir = DECAY_SWEEP_DIR / f"decay_{label}"
        if not rate_dir.exists():
            print(f"  WARNING: No results for decay_rate={label}, skipping")
            continue

        # Find timestamped subdirectory
        subdirs = sorted(rate_dir.glob(f"decay_sweep_{label}_*"), reverse=True)
        if not subdirs:
            print(f"  WARNING: No timestamped dir for decay_rate={label}, skipping")
            continue

        # Load per-session CSVs and combine
        analysis_dir = subdirs[0] / "analysis"
        session_csvs = list(analysis_dir.glob("session_*/session_data_summary.csv"))
        if not session_csvs:
            print(f"  WARNING: No analysis CSVs for decay_rate={label}, skipping")
            continue

        df = pd.concat([pd.read_csv(c) for c in session_csvs], ignore_index=True)
        # Use gap=1 F1 for MDI (only recommender in these configs)
        gap1 = df[df['gap'] == 1]
        f1_mean = gap1['f1_score'].mean()
        f1_std = gap1['f1_score'].std()

        available_rates.append(rate)
        means.append(f1_mean)
        stds.append(f1_std)
        print(f"  decay_rate={label}: F1={f1_mean:.3f} ± {f1_std:.3f}")

    if len(available_rates) < 2:
        print("  ERROR: Need at least 2 decay rates to plot. Skipping figure.")
        return

    available_rates = np.array(available_rates)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.errorbar(
        available_rates, means, yerr=stds,
        fmt='o-', linewidth=2, markersize=6,
        color='#2176AE', capsize=4, capthick=1.2,
        ecolor='#2176AE', elinewidth=0.8, alpha=0.9,
    )

    # Mark the best decay rate
    best_idx = np.argmax(means)
    best_rate = available_rates[best_idx]
    ax.axvline(x=best_rate, color='#D64045', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Best ($\\lambda_r = {best_rate}$)')

    ax.set_xlabel('Temporal Decay Rate ($\\lambda_r$)')
    ax.set_ylabel('F1 Score (gap = 1)')
    ax.set_xticks(available_rates)
    ax.set_xticklabels([f'{r:.2f}' for r in available_rates], fontsize=8)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--benchmark-csv', type=Path, default=BENCHMARK_DIR)
    parser.add_argument('--simba-csv', type=Path, default=SIMBA_DRILLDOWN_DIR)
    parser.add_argument('--sdss-csv', type=Path, default=SDSS_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--paper-figures-dir', type=Path,
                        default=Path('vision-paper/figures'))
    args = parser.parse_args()

    set_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_bench = ensure_analysis_csv(args.benchmark_csv, BENCHMARK_RESULTS_DIR, BENCHMARK_CONFIG)
    df_simba = ensure_analysis_csv(args.simba_csv, SIMBA_DRILLDOWN_RESULTS_DIR, SIMBA_DRILLDOWN_CONFIG)
    df_sdss = ensure_analysis_csv(args.sdss_csv, SDSS_RESULTS_DIR, SDSS_CONFIG)

    print("\nGenerating figures...")

    # Figure 1: Hero comparison
    generate_workload_comparison(
        [
            (df_simba, 'SIMBA Drilldown'),
            (df_bench, 'Adversarial\nBenchmark'),
            (df_sdss, 'SDSS Archive'),
        ],
        [
            args.output_dir / "workload_comparison.pdf",
            args.paper_figures_dir / "workload_comparison.pdf",
        ],
    )

    # Figure 2: Gap analysis
    generate_gap_analysis(
        df_bench,
        args.output_dir / "gap_analysis.pdf"
    )

    # Figure 3: Overlap distribution
    generate_overlap_analysis(
        df_simba, df_bench,
        args.paper_figures_dir / "overlap_analysis.pdf"
    )

    # Figure 4: Decay sensitivity
    generate_decay_sensitivity(
        args.output_dir / "decay_sensitivity.pdf"
    )

    print(f"\nDone. Figures in {args.output_dir}/ and {args.paper_figures_dir}/")


if __name__ == '__main__':
    main()
