"""Generate a derived scenario by inheriting the DAG and top-left blocks from a base scenario.

This script guarantees exact inheritance of the base scenario's DAG and the first
`base_procs` columns/rows of the execution-time, bandwidth and power matrices.
Only the additional processors (columns/rows) are generated deterministically from a seed.

Usage example:
  python graphs/generate_inherited_scenario.py \
    --inherit-prefix graphs/Scenario_3/peft1024_48proc \
    --target-procs 128 \
    --out-prefix graphs/Scenario_3/peft1024_128proc \
    --seed 42

The generated files follow the repository convention:
  <out-prefix>_task_connectivity.csv
  <out-prefix>_task_exe_time.csv
  <out-prefix>_resource_BW.csv
  <out-prefix>_task_power.csv

The script is conservative: it will raise if the target procs is smaller than the base procs
and will validate that the top-left blocks are identical after generation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def copy_connectivity(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = in_path.read_bytes()
    out_path.write_bytes(data)


def extend_exec_or_power(base_df: pd.DataFrame, target_p: int, seed: int, integer: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_tasks = base_df.shape[0]
    base_p = base_df.shape[1]
    cols = [f"P_{i}" for i in range(target_p)]
    out = pd.DataFrame(index=base_df.index, columns=cols)

    # copy base columns exactly
    out.iloc[:, :base_p] = base_df.values

    if target_p > base_p:
        base_mean = base_df.mean(axis=1).to_numpy()
        # generate per-new-processor performance multiplier (deterministic)
        for j in range(base_p, target_p):
            perf = float(rng.uniform(0.75, 1.25))
            # construct column
            colvals = []
            for t in range(n_tasks):
                task_bias = 0.7 if (t % 17) == (j % 17) else 1.0
                v = base_mean[t] * perf * task_bias
                if integer:
                    v = max(1, int(round(v)))
                else:
                    # small gaussian noise to avoid exact duplicates across procs
                    v = max(0.01, float(round(v + rng.normal(0, 0.05), 2)))
                colvals.append(v)
            out.iloc[:, j] = colvals

    # keep dtypes consistent with input where possible
    if integer:
        out = out.astype(int)
    else:
        out = out.astype(float)
    return out


def extend_bw(base_bw: pd.DataFrame, target_p: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_p = base_bw.shape[0]
    names = [f"P_{i}" for i in range(target_p)]
    out = pd.DataFrame(0, index=names, columns=names, dtype=int)

    # copy top-left block exactly
    out.iloc[:base_p, :base_p] = base_bw.values

    # generate new rows/cols by sampling existing rows and applying tiny deterministic perturbation
    flat = base_bw.values.flatten()
    flat_pos = flat[flat > 0]
    mu = int(flat_pos.mean()) if flat_pos.size > 0 else 1

    for i in range(base_p, target_p):
        # pick a template row index deterministically
        template_idx = int(rng.integers(0, base_p))
        template = base_bw.iloc[template_idx].to_numpy().copy()
        # apply small perturbation: with low probability flip 1->2 or 2->1
        newrow = template.copy()
        for j in range(base_p):
            if template[j] == 0:
                newrow[j] = 0
            else:
                if rng.random() < 0.05:
                    newrow[j] = 2 if template[j] == 1 else 1
                else:
                    newrow[j] = template[j]
        out.iloc[i, :base_p] = newrow
        out.iloc[:base_p, i] = newrow

    # fill new-new block
    for i in range(base_p, target_p):
        for j in range(base_p, target_p):
            if i == j:
                out.iat[i, j] = 0
            else:
                # sample around mu with small jitter
                val = int(max(1, round(rng.normal(mu, max(0.5, mu*0.1)))))
                out.iat[i, j] = val

    # ensure diagonal zero
    for i in range(target_p):
        out.iat[i, i] = 0
    return out


def validate_inheritance(base: pd.DataFrame, derived: pd.DataFrame) -> None:
    base_p = base.shape[1]
    left = derived.iloc[:, :base_p]
    # exact equality for integers or strings
    if not left.equals(base):
        # try exact numeric equality as fallback
        if not np.array_equal(left.to_numpy(), base.to_numpy()):
            raise AssertionError("Inheritance validation failed: derived top-left block does not match base exactly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inherit-prefix', required=True, help='prefix of base scenario files (without suffix)')
    ap.add_argument('--target-procs', required=True, type=int)
    ap.add_argument('--out-prefix', required=True, help='output prefix for new scenario files')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    base_prefix = Path(args.inherit_prefix)
    out_prefix = Path(args.out_prefix)
    t = args.target_procs
    seed = args.seed

    base_conn = base_prefix.parent / (base_prefix.name + '_task_connectivity.csv')
    base_exec = base_prefix.parent / (base_prefix.name + '_task_exe_time.csv')
    base_bw = base_prefix.parent / (base_prefix.name + '_resource_BW.csv')
    base_power = base_prefix.parent / (base_prefix.name + '_task_power.csv')

    out_conn = out_prefix.parent / (out_prefix.name + '_task_connectivity.csv')
    out_exec = out_prefix.parent / (out_prefix.name + '_task_exe_time.csv')
    out_bw = out_prefix.parent / (out_prefix.name + '_resource_BW.csv')
    out_power = out_prefix.parent / (out_prefix.name + '_task_power.csv')

    # basic existence checks
    for p in (base_conn, base_exec, base_bw, base_power):
        if not p.exists():
            print(f'Error: base file missing: {p}', file=sys.stderr)
            sys.exit(1)

    # copy connectivity exactly
    copy_connectivity(base_conn, out_conn)

    # read base matrices
    exec_df = read_matrix(base_exec)
    bw_df = read_matrix(base_bw)
    power_df = read_matrix(base_power)

    base_p = exec_df.shape[1]
    if t < base_p:
        raise SystemExit(f'Target procs {t} must be >= base procs {base_p}')

    # extend matrices
    exec_new = extend_exec_or_power(exec_df, t, seed, integer=True)
    power_new = extend_exec_or_power(power_df, t, seed + 1, integer=False)
    bw_new = extend_bw(bw_df, t, seed + 2)

    # validate top-left inheritance
    validate_inheritance(exec_df, exec_new)
    validate_inheritance(power_df, power_new)
    if not np.array_equal(bw_new.values[:base_p, :base_p], bw_df.values):
        raise AssertionError('Inheritance validation failed for bandwidth top-left block')

    # write outputs
    write_matrix(exec_new, out_exec)
    write_matrix(bw_new, out_bw)
    write_matrix(power_new, out_power)

    print('Generated inherited scenario:')
    print(f'  connectivity: {out_conn}')
    print(f'  exec:         {out_exec}')
    print(f'  bw:           {out_bw}')
    print(f'  power:        {out_power}')


if __name__ == '__main__':
    main()
