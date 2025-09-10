import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

OUTPUT_DIR = os.path.join('outputs', 'plots')

ALG_ORDER = [
    'PEFT', 'IPEFT', 'HEFT', 'HEFT-LA', 'IHEFT', 'DLS'
]
ALG_COLORS: Dict[str, str] = {
    'PEFT': '#1f77b4',   # blue
    'IPEFT': '#17becf',  # cyan
    'HEFT': '#2ca02c',   # green
    'HEFT-LA': '#98df8a',# light green
    'IHEFT': '#ff7f0e',  # orange
    'DLS': '#d62728',    # red
}

METRICS = [
    ('makespan', 'Makespan (lower is better)'),
    ('energy_cost', 'Energy (arbitrary units, lower is better)')
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_alg_order(algs: List[str]) -> List[str]:
    # Keep known order; append any unknowns at the end
    known = [a for a in ALG_ORDER if a in algs]
    unknown = [a for a in algs if a not in ALG_ORDER]
    return known + sorted(unknown)


def plot_metric_by_procs_per_scenario(df: pd.DataFrame, metric: str, ylabel: str) -> List[str]:
    saved = []
    scenarios = sorted(df['scenario'].unique(), key=lambda s: (
        int(s.split('_')[1]) if '_' in s and s.split('_')[1].isdigit() else 0, s
    ))
    for scenario in scenarios:
        sdf = df[df['scenario'] == scenario].copy()
        # Get available procs sorted numerically
        procs = sorted(sdf['procs'].unique())
        algs = normalize_alg_order(sorted(sdf['algorithm'].unique()))

        plt.figure(figsize=(8, 5))
        for alg in algs:
            adf = sdf[sdf['algorithm'] == alg].sort_values('procs')
            if adf.empty:
                continue
            plt.plot(adf['procs'], adf[metric], marker='o', label=alg, color=ALG_COLORS.get(alg))

        plt.title(f'{scenario}: {ylabel} vs Processors')
        plt.xlabel('Processors')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=9)
        fname = os.path.join(OUTPUT_DIR, f'{scenario}_{metric}.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        saved.append(fname)
    return saved


def scatter_makespan_vs_energy(df: pd.DataFrame) -> str:
    plt.figure(figsize=(7, 5))
    algs = normalize_alg_order(sorted(df['algorithm'].unique()))
    for alg in algs:
        adf = df[df['algorithm'] == alg]
        plt.scatter(adf['makespan'], adf['energy_cost'], s=28, alpha=0.8, label=alg, color=ALG_COLORS.get(alg))
    plt.xlabel('Makespan')
    plt.ylabel('Energy')
    plt.title('Makespan vs Energy across all scenarios')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    fname = os.path.join(OUTPUT_DIR, 'makespan_vs_energy_scatter.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def main():
    ensure_dir(OUTPUT_DIR)
    csv_path = os.path.join('outputs', 'aggregate_metrics.csv')
    if not os.path.isfile(csv_path):
        raise SystemExit(f'Aggregate CSV not found at {csv_path}. Run aggregate_all_results.py first.')

    df = pd.read_csv(csv_path)

    generated: List[str] = []
    for metric, ylabel in METRICS:
        generated += plot_metric_by_procs_per_scenario(df, metric, ylabel)

    generated.append(scatter_makespan_vs_energy(df))

    # Also create a compact overview per algorithm: relative makespan to best per (scenario, procs)
    tmp = df.copy()
    tmp['key'] = tmp['scenario'].astype(str) + '_' + tmp['procs'].astype(str)
    best_by_key = tmp.groupby('key')['makespan'].transform('min')
    tmp['rel_makespan'] = tmp['makespan'] / best_by_key

    # Plot average relative makespan per algorithm (lower is better)
    avg_rel = tmp.groupby('algorithm')['rel_makespan'].mean().reset_index()
    avg_rel = avg_rel.sort_values('rel_makespan')

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        avg_rel['algorithm'], avg_rel['rel_makespan'],
        color=[ALG_COLORS.get(a, '#999999') for a in avg_rel['algorithm']]
    )
    plt.axhline(1.0, color='#555', linestyle='--', linewidth=1)
    plt.ylabel('Average Relative Makespan (to best)')
    plt.title('Algorithm competitiveness (avg across all scenarios)')
    for rect in bars:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'avg_relative_makespan.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    generated.append(fname)

    print('Saved plots:')
    for f in generated:
        print(' -', f)


if __name__ == '__main__':
    main()
