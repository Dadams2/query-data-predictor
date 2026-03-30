"""
Generate publication-quality figures for the vision paper.

This script uses `paper_artifact_manifest.yaml` to find the preferred result
snapshots, and falls back to the latest matching run for each config.

Usage:
    UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.matplotlib XDG_CACHE_HOME=.cache \
      uv run python generate_publication_figures.py
"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from query_data_predictor.paper_artifact import (
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    load_metrics_dataframe,
    resolve_experiment_paths,
)
warnings.filterwarnings('ignore')

FROZEN_DECAY_SWEEP_DIR = Path("previous-results/decay-sweep")
LATEST_DECAY_SWEEP_DIR = Path("results/decay_sweep")

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
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
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


def describe_dataframe(df: pd.DataFrame, label: str, source: Path):
    """Print a small status line for a loaded experiment dataframe."""
    print(f"  Loaded {len(df)} rows for {label} from {source}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Hero — Workload Comparison
# ═══════════════════════════════════════════════════════════════════════════

WORKLOAD_PANEL_SPECS = [
    ("workload_comparison_simba.pdf", "SIMBA Drill-down"),
    ("workload_comparison_adversarial.pdf", "Adversarial Benchmark"),
    ("workload_comparison_sdss.pdf", "SkyServer SQL Logs"),
]


def _workload_gap1_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate gap=1 F1 statistics for a workload."""
    gap1 = df[df['gap'] == 1]
    agg = gap1.groupby('recommender')['f1_score'].agg(['mean', 'std']).reset_index()
    agg['std'] = agg['std'].fillna(0.0)
    present = [r for r in RECOMMENDER_ORDER if r in agg['recommender'].values]
    return agg.set_index('recommender').loc[present].reset_index()


def _add_workload_panel(ax, agg: pd.DataFrame, ymax: float, show_ylabel: bool):
    """Draw one workload panel with no legend and no x-axis text labels."""
    present = agg['recommender'].tolist()
    x = np.arange(len(present))
    ax.bar(
        x, agg['mean'], yerr=agg['std'], capsize=3,
        color=[COLORS[r] for r in present],
        edgecolor='white', linewidth=0.6, width=0.68,
        error_kw={'linewidth': 1.0},
    )

    baseline_mask = agg['recommender'] != 'multidimensional_interestingness'
    baseline_mean = agg.loc[baseline_mask, 'mean'].mean()
    ax.axhline(
        baseline_mean,
        color='gray',
        linestyle=':',
        linewidth=1.0,
        alpha=0.9,
    )

    ax.set_xticks([])
    ax.set_xlim(-0.55, len(present) - 0.45)
    ax.set_ylim(0, min(1.0, ymax))
    ax.set_yticks(np.arange(0, min(1.0, ymax) + 1e-9, 0.1))
    if show_ylabel:
        ax.set_ylabel('F1 score')
    else:
        ax.set_ylabel('')
    ax.grid(axis='y', alpha=0.22)
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _shared_legend_handles():
    """Legend handles shared by all workload panels."""
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[key], edgecolor='white')
        for key in RECOMMENDER_ORDER
    ]
    labels = [NAME_MAP[key] for key in RECOMMENDER_ORDER]
    handles.append(plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0))
    labels.append('Baseline mean')
    return handles, labels


def _save_workload_panel(ax, output_path: Path):
    """Save a single workload panel figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(ax.figure)


def generate_workload_comparison(
    workloads: list[tuple[pd.DataFrame, str]],
    output_dir: Path,
    mirror_output_dir: Path | None = None,
):
    """
    Generate a combined preview, three panel PDFs, and one shared legend PDF.

    The paper can use the per-panel files with LaTeX subfigures, while the
    combined preview remains convenient for artifact browsing.
    """
    if len(workloads) != 3:
        raise ValueError("generate_workload_comparison expects exactly three workloads")

    output_dir.mkdir(parents=True, exist_ok=True)
    if mirror_output_dir is not None:
        mirror_output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [_workload_gap1_summary(df) for df, _ in workloads]
    ymax = 0.4
    for agg in summaries:
        if not agg.empty:
            ymax = max(ymax, float((agg['mean'] + agg['std']).max()) * 1.1)

    # Per-panel outputs used by the LaTeX figure.
    for idx, (filename, _) in enumerate(WORKLOAD_PANEL_SPECS):
        fig, ax = plt.subplots(figsize=(2.7, 2.35))
        _add_workload_panel(ax, summaries[idx], ymax, show_ylabel=(idx == 0))
        fig.tight_layout()
        _save_workload_panel(ax, output_dir / filename)
        if mirror_output_dir is not None:
            fig, ax = plt.subplots(figsize=(2.7, 2.35))
            _add_workload_panel(ax, summaries[idx], ymax, show_ylabel=(idx == 0))
            fig.tight_layout()
            _save_workload_panel(ax, mirror_output_dir / filename)

    legend_handles, legend_labels = _shared_legend_handles()
    for target_dir in [output_dir] + ([mirror_output_dir] if mirror_output_dir is not None else []):
        legend_fig = plt.figure(figsize=(2.1, 2.35))
        legend_fig.legend(
            legend_handles,
            legend_labels,
            loc='center',
            frameon=False,
            ncol=1,
            fontsize=10,
            handlelength=1.7,
            labelspacing=0.7,
        )
        legend_target = target_dir / "workload_comparison_legend.pdf"
        legend_fig.savefig(legend_target)
        print(f"  Saved: {legend_target}")
        plt.close(legend_fig)

    # Combined preview for artifact browsing.
    combined_fig = plt.figure(figsize=(11.8, 2.7))
    gs = combined_fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.92], wspace=0.28)
    for idx in range(3):
        ax = combined_fig.add_subplot(gs[0, idx])
        _add_workload_panel(ax, summaries[idx], ymax, show_ylabel=(idx == 0))
        ax.set_title(WORKLOAD_PANEL_SPECS[idx][1], fontsize=11, pad=8, fontweight='semibold')

    legend_ax = combined_fig.add_subplot(gs[0, 3])
    legend_ax.axis('off')
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc='center',
        frameon=False,
        ncol=1,
        fontsize=10,
        handlelength=1.7,
        labelspacing=0.7,
    )

    combined_fig.tight_layout()
    for target_dir in [output_dir] + ([mirror_output_dir] if mirror_output_dir is not None else []):
        preview_target = target_dir / "workload_comparison.pdf"
        combined_fig.savefig(preview_target)
        print(f"  Saved: {preview_target}")
    plt.close(combined_fig)


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

def resolve_decay_sweep_run(label: str, prefer_frozen: bool = True) -> Path | None:
    """Resolve the frozen or latest result directory for one decay-rate run."""
    frozen_dir = FROZEN_DECAY_SWEEP_DIR / f"decay-{label}"
    latest_parent = LATEST_DECAY_SWEEP_DIR / f"decay_{label}"

    if prefer_frozen and frozen_dir.exists():
        return frozen_dir

    if latest_parent.exists():
        subdirs = sorted(latest_parent.glob(f"decay_sweep_{label}_*"), reverse=True)
        if subdirs:
            return subdirs[0]

    if frozen_dir.exists():
        return frozen_dir

    return None


def generate_decay_sensitivity(output_path: Path, prefer_frozen: bool = True):
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
        run_dir = resolve_decay_sweep_run(label, prefer_frozen=prefer_frozen)
        if run_dir is None:
            print(f"  WARNING: No results for decay_rate={label}, skipping")
            continue

        # Load per-session CSVs and combine
        analysis_dir = run_dir / "analysis"
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
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument('--benchmark-csv', type=Path)
    parser.add_argument('--simba-csv', type=Path)
    parser.add_argument('--sdss-csv', type=Path)
    parser.add_argument('--use-latest-results', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/paper/figures'))
    parser.add_argument('--paper-figures-dir', type=Path, default=None)
    args = parser.parse_args()

    set_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.paper_figures_dir is None:
        detected = Path('vision-paper/figures')
        args.paper_figures_dir = detected if detected.exists() else None
    if args.paper_figures_dir is not None:
        args.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    prefer_frozen = not args.use_latest_results

    benchmark_paths = resolve_experiment_paths(
        manifest, 'adversarial_benchmark', prefer_frozen=prefer_frozen
    )
    simba_paths = resolve_experiment_paths(
        manifest, 'simba_drilldown', prefer_frozen=prefer_frozen
    )
    sdss_paths = resolve_experiment_paths(
        manifest, 'sdss_logs', prefer_frozen=prefer_frozen
    )

    print("Loading data...")
    if args.benchmark_csv:
        df_bench = pd.read_csv(args.benchmark_csv)
        describe_dataframe(df_bench, benchmark_paths.label, args.benchmark_csv)
    else:
        df_bench = load_metrics_dataframe(benchmark_paths)
        describe_dataframe(df_bench, benchmark_paths.label, benchmark_paths.analysis_csv)

    if args.simba_csv:
        df_simba = pd.read_csv(args.simba_csv)
        describe_dataframe(df_simba, simba_paths.label, args.simba_csv)
    else:
        df_simba = load_metrics_dataframe(simba_paths)
        describe_dataframe(df_simba, simba_paths.label, simba_paths.analysis_csv)

    if args.sdss_csv:
        df_sdss = pd.read_csv(args.sdss_csv)
        describe_dataframe(df_sdss, sdss_paths.label, args.sdss_csv)
    else:
        df_sdss = load_metrics_dataframe(sdss_paths)
        describe_dataframe(df_sdss, sdss_paths.label, sdss_paths.analysis_csv)

    print("\nGenerating figures...")

    # Figure 1: Hero comparison
    generate_workload_comparison(
        [
            (df_simba, 'SIMBA Drilldown'),
            (df_bench, 'Adversarial\nBenchmark'),
            (df_sdss, 'SDSS Archive'),
        ],
        args.output_dir,
        mirror_output_dir=args.paper_figures_dir,
    )

    # Figure 2: Gap analysis
    generate_gap_analysis(
        df_bench,
        args.output_dir / "gap_analysis.pdf"
    )
    if args.paper_figures_dir is not None:
        generate_gap_analysis(
            df_bench,
            args.paper_figures_dir / "gap_analysis.pdf"
        )

    # Figure 3: Overlap distribution
    generate_overlap_analysis(
        df_simba, df_bench,
        args.output_dir / "overlap_analysis.pdf"
    )
    if args.paper_figures_dir is not None:
        generate_overlap_analysis(
            df_simba, df_bench,
            args.paper_figures_dir / "overlap_analysis.pdf"
        )

    # Figure 4: Decay sensitivity
    generate_decay_sensitivity(
        args.output_dir / "decay_sensitivity.pdf",
        prefer_frozen=prefer_frozen,
    )
    if args.paper_figures_dir is not None:
        generate_decay_sensitivity(
            args.paper_figures_dir / "decay_sensitivity.pdf",
            prefer_frozen=prefer_frozen,
        )

    if args.paper_figures_dir is not None:
        print(f"\nDone. Figures in {args.output_dir}/ and {args.paper_figures_dir}/")
    else:
        print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == '__main__':
    main()
