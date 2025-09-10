"""Generate a PEFT-like heterogeneous DAG with 4096 tasks and 8 processors.

Follows Scenario 1-4 structure; scaled up to 4096 tasks.
- 8 layers to keep per-layer width moderate at this size.
- Forward-only edges; each task chooses 1-3 targets within next 1-2 layers.
- Comm weights: 0 (15% chance) else 5..60 scaled by layer depth.
- Exec times: base 8..40, per-processor perf 0.75..1.25 + mod-17 affinity (0.7x time).
- Power: 1.5 + 0.9 * (1/perf) + noise; +5% if affinity time improvement.

Outputs four CSVs with given prefix: *_task_connectivity.csv, *_task_exe_time.csv,
*_resource_BW.csv, *_task_power.csv.
"""
from __future__ import annotations
import argparse, random, csv
from pathlib import Path
import numpy as np

DEF_TASKS = 4096
DEF_PROCS = 8


def build_dag(n_tasks:int, n_layers:int=8, seed:int=42):
    rng = random.Random(seed)
    base = n_tasks // n_layers
    rem = n_tasks % n_layers
    layers=[]; t=0
    for i in range(n_layers):
        sz = base + (1 if i < rem else 0)
        layers.append(list(range(t, t+sz)))
        t+=sz
    edges=[]
    for li, layer in enumerate(layers[:-1]):
        future = [v for l in layers[li+1: min(li+3, n_layers)] for v in l]
        for u in layer:
            out_deg = rng.randint(1,3)
            targets = rng.sample(future, min(out_deg, len(future)))
            for v in targets:
                if v <= u: continue
                if rng.random() < 0.15:
                    w=0
                else:
                    base_w = rng.randint(5,60)
                    depth_factor = 1 + li/(n_layers-1)*0.6
                    w = int(base_w * depth_factor)
                edges.append((u,v,w))
    return layers, edges


def gen_exec_matrix(n_tasks:int, n_procs:int, seed:int=43):
    rng = np.random.default_rng(seed)
    perf = rng.uniform(0.75,1.25,size=n_procs)
    base_times = rng.integers(8,41,size=n_tasks)
    mat = np.zeros((n_tasks,n_procs), dtype=int)
    for i in range(n_tasks):
        for p in range(n_procs):
            task_bias = 1.0
            if (i % 17) == (p % 17):
                task_bias *= 0.7
            mat[i,p] = max(1, int(round(base_times[i]*perf[p]*task_bias)))
    return mat, perf


def gen_power_matrix(exec_mat, perf_multipliers, seed:int=44):
    rng = np.random.default_rng(seed)
    n_tasks, n_procs = exec_mat.shape
    power = np.zeros_like(exec_mat, dtype=float)
    for p in range(n_procs):
        speed = 1.0 / perf_multipliers[p]
        for t in range(n_tasks):
            base = 1.5 + speed*0.9
            noise = rng.normal(0,0.15)
            if exec_mat[t,p] < exec_mat[t].mean()*0.85:
                base *= 1.05
            power[t,p] = max(0.5, round(base+noise,2))
    return power


def write_connectivity(prefix:str, n_tasks:int, edges):
    path = Path(f"{prefix}_task_connectivity.csv")
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T'] + [f"T{i}" for i in range(n_tasks)])
        mat = [[0]*n_tasks for _ in range(n_tasks)]
        for u,v,wt in edges:
            mat[u][v] = wt
        for i in range(n_tasks):
            w.writerow([f"T{i}"] + mat[i])


def write_exec(prefix:str, exec_mat):
    n_tasks, n_procs = exec_mat.shape
    path = Path(f"{prefix}_task_exe_time.csv")
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TP'] + [f"P_{p}" for p in range(n_procs)])
        for t in range(n_tasks):
            w.writerow([f"T_{t}"] + list(map(int, exec_mat[t])))


def write_bw(prefix:str, n_procs:int, seed:int=45):
    rng = random.Random(seed)
    path = Path(f"{prefix}_resource_BW.csv")
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P'] + [f"P_{p}" for p in range(n_procs)])
        for i in range(n_procs):
            row=[f"P_{i}"]
            for j in range(n_procs):
                if i==j:
                    row.append(0)
                else:
                    row.append(1 if rng.random()<0.9 else 2)
            w.writerow(row)


def write_power(prefix:str, power_mat):
    n_tasks, n_procs = power_mat.shape
    path = Path(f"{prefix}_task_power.csv")
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TP'] + [f"P_{p}" for p in range(n_procs)])
        for t in range(n_tasks):
            w.writerow([f"T_{t}"] + [f"{power_mat[t,p]:.2f}" for p in range(n_procs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=DEF_TASKS)
    ap.add_argument('--procs', type=int, default=DEF_PROCS)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', required=True)
    args = ap.parse_args()

    layers, edges = build_dag(args.tasks, args.layers, seed=args.seed)
    exec_mat, perf = gen_exec_matrix(args.tasks, args.procs, seed=args.seed+1)
    power_mat = gen_power_matrix(exec_mat, perf, seed=args.seed+2)

    write_connectivity(args.prefix, args.tasks, edges)
    write_exec(args.prefix, exec_mat)
    write_bw(args.prefix, args.procs, seed=args.seed+3)
    write_power(args.prefix, power_mat)

    print(f"Wrote PEFT-like scenario to prefix {args.prefix}")
    print(f"Tasks={args.tasks}, Procs={args.procs}, Edges={len(edges)}")

if __name__ == '__main__':
    main()
