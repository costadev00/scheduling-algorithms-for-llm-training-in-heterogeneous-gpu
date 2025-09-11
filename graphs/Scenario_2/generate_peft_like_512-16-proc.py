"""Generate PEFT-like heterogeneous DAG (512 tasks, 16 processors) com herança.

Design goals copied from the small peftgraph (10 tasks, 3 procs):
- Moderate fan-out early then converging toward sinks.
- Varied communication weights (some heavier edges on later dependencies).
- Heterogeneous execution times with per-processor bias (some processors faster on groups of tasks).
- Power (energy rate) matrix correlated loosely with speed (faster processor => slightly higher power sometimes, but not always) to create trade-offs.

Approach:
- Build layered DAG: ~6 layers; random edges forward only with controlled sparsity similar to original (avg out-degree roughly 1-2).
- Add a tail of sinks merging from prior layer.
- Communication weights: small integers (0, 5-60) biased higher toward later layers to mimic original heavier late edges.
- Execution times: base drawn from uniform 8-40 then apply per-processor multiplicative noise so that processors have distinct performance profiles.
- Power: base proportional to (execution time scaling factor)^0.8 + noise.

Outputs four CSVs matching existing format:
  <prefix>_task_connectivity.csv
  <prefix>_task_exe_time.csv
  <prefix>_resource_BW.csv
  <prefix>_task_power.csv

Usage:
  python generate_peft_like_256.py --prefix graphs/peft256

"""
from __future__ import annotations
import argparse, random, csv, sys
from pathlib import Path
import numpy as np

DEF_TASKS = 512
DEF_PROCS = 16

def build_dag(n_tasks:int, n_layers:int=6, seed:int=42):
    rng = random.Random(seed)
    # Allocate tasks to layers roughly evenly
    base = n_tasks // n_layers
    remainder = n_tasks % n_layers
    layers = []
    t = 0
    for i in range(n_layers):
        sz = base + (1 if i < remainder else 0)
        layers.append(list(range(t, t+sz)))
        t += sz
    edges = []  # (u,v,weight)
    for li, layer in enumerate(layers[:-1]):
        next_union = [v for l in layers[li+1: min(li+3, n_layers)] for v in l]  # allow skip one layer occasionally
        for u in layer:
            # choose 1-3 outgoing edges
            out_deg = rng.randint(1,3)
            targets = rng.sample(next_union, min(out_deg, len(next_union)))
            for v in targets:
                if v <= u: # ensure acyclic forward
                    continue
                # communication weight similar scale to original (0 or 5..60)
                if rng.random() < 0.15:
                    w=0
                else:
                    base_w = rng.randint(5,60)
                    # slightly amplify for later layers
                    depth_factor = 1 + li/(n_layers-1)*0.6
                    w = int(base_w * depth_factor)
                edges.append((u,v,w))
    return layers, edges

def gen_exec_matrix(n_tasks:int, n_procs:int, seed:int=43):
    rng = np.random.default_rng(seed)
    # Per-processor performance multiplier (lower => faster)
    perf = rng.uniform(0.75,1.25,size=n_procs)
    base_times = rng.integers(8,41,size=n_tasks)
    mat = np.zeros((n_tasks,n_procs), dtype=int)
    for i in range(n_tasks):
        for p in range(n_procs):
            # mimic some processors much faster for subset: add task-specific variance
            task_bias = 1.0
            if (i % 17) == (p % 17): # occasional affinity
                task_bias *= 0.7
            mat[i,p] = max(1, int(round(base_times[i]*perf[p]*task_bias)))
    return mat, perf

def gen_power_matrix(exec_mat, perf_multipliers, seed:int=44):
    rng = np.random.default_rng(seed)
    n_tasks, n_procs = exec_mat.shape
    power = np.zeros_like(exec_mat, dtype=float)
    # Derive nominal speed: inverse perf multiplier
    for p in range(n_procs):
        speed = 1.0 / perf_multipliers[p]
        for t in range(n_tasks):
            base = 1.5 + speed*0.9  # baseline scaling
            noise = rng.normal(0,0.15)
            # Slight correlation: tasks with affinity speed-ups may draw a bit more power
            if exec_mat[t,p] < exec_mat[t].mean()*0.85:
                base *= 1.05
            power[t,p] = max(0.5, round(base+noise,2))
    return power

def write_connectivity(path_prefix:str, n_tasks:int, edges:list[tuple[int,int,int]]):
    path = Path(f"{path_prefix}_task_connectivity.csv")
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['T'] + [f"T{i}" for i in range(n_tasks)]
        writer.writerow(header)
        # build adjacency matrix with weights
        mat = [[0]*n_tasks for _ in range(n_tasks)]
        for u,v,w in edges:
            mat[u][v] = w
        for i in range(n_tasks):
            writer.writerow([f"T{i}"] + mat[i])

def write_exec(path_prefix:str, exec_mat):
    n_tasks, n_procs = exec_mat.shape
    path = Path(f"{path_prefix}_task_exe_time.csv")
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TP'] + [f"P_{p}" for p in range(n_procs)])
        for t in range(n_tasks):
            writer.writerow([f"T_{t}"] + list(map(int, exec_mat[t])))

def write_bw(path_prefix:str, n_procs:int, seed:int=45):
    rng = random.Random(seed)
    path = Path(f"{path_prefix}_resource_BW.csv")
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['P'] + [f"P_{p}" for p in range(n_procs)])
        for i in range(n_procs):
            row=[f"P_{i}"]
            for j in range(n_procs):
                if i==j:
                    row.append(0)
                else:
                    # Keep simple symmetric bandwidth 0/1 with occasional 2 for heterogeneity
                    val = 1 if rng.random()<0.9 else 2
                    row.append(val)
            writer.writerow(row)

def write_power(path_prefix:str, power_mat):
    n_tasks, n_procs = power_mat.shape
    path = Path(f"{path_prefix}_task_power.csv")
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TP'] + [f"P_{p}" for p in range(n_procs)])
        for t in range(n_tasks):
            writer.writerow([f"T_{t}"] + [f"{power_mat[t,p]:.2f}" for p in range(n_procs)])

def _load_matrix(prefix, suffix):
    p=Path(f"{prefix}_{suffix}.csv")
    if not p.exists(): return None
    import csv as _csv
    with p.open() as f:
        r=_csv.reader(f); header=next(r)
        rows=[row[1:] for row in r]
    if 'exe_time' in suffix:
        return np.array([[int(x) for x in row] for row in rows], dtype=int)
    if 'power' in suffix:
        return np.array([[float(x) for x in row] for row in rows], dtype=float)
    if 'resource_BW' in suffix:
        return np.array([[int(x) for x in row] for row in rows], dtype=int)
    return None

def _load_edges(prefix):
    p=Path(f"{prefix}_task_connectivity.csv")
    if not p.exists(): return None
    import csv as _csv
    with p.open() as f:
        r=_csv.reader(f); header=next(r)
        rows=list(r)
    edges=[]
    for i,row in enumerate(rows):
        for j,val in enumerate(row[1:]):
            v=int(val)
            if v>0: edges.append((i,j,v))
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=DEF_TASKS)
    ap.add_argument('--procs', type=int, default=DEF_PROCS)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--inherit-prefix')
    ap.add_argument('--tolerance', type=float, default=0.0)
    args=ap.parse_args()

    inherit_exec=inherit_power=inherit_bw=inherit_edges=None
    if args.inherit_prefix:
        inherit_exec=_load_matrix(args.inherit_prefix,'task_exe_time')
        inherit_power=_load_matrix(args.inherit_prefix,'task_power')
        inherit_bw=_load_matrix(args.inherit_prefix,'resource_BW')
        inherit_edges=_load_edges(args.inherit_prefix)

    if inherit_edges is not None:
        edges=inherit_edges
        print('Reutilizando conectividade herdada.')
    else:
        _, edges = build_dag(args.tasks, args.layers, seed=args.seed)

    exec_mat, perf = gen_exec_matrix(args.tasks, args.procs, seed=args.seed+1)
    inherited_cols=0
    if inherit_exec is not None:
        if inherit_exec.shape[0]!=args.tasks:
            print('[erro] tasks incompatíveis', file=sys.stderr); sys.exit(1)
        base_cols=inherit_exec.shape[1]
        if base_cols>=args.procs:
            print('[erro] base possui >= colunas que alvo', file=sys.stderr); sys.exit(1)
        if args.tolerance>=0:
            diff=abs(exec_mat[:,:base_cols]-inherit_exec).max()
            if diff>args.tolerance:
                print(f'[info] Substituindo colunas herdadas (desvio={diff})')
        exec_mat[:,:base_cols]=inherit_exec
        inherited_cols=base_cols

    power_mat = gen_power_matrix(exec_mat, perf, seed=args.seed+2)
    if inherit_power is not None and inherit_power.shape[1]==inherited_cols:
        power_mat[:,:inherited_cols]=inherit_power

    rng_bw=random.Random(args.seed+3)
    bw=np.zeros((args.procs,args.procs),dtype=int)
    for i in range(args.procs):
        for j in range(args.procs):
            if i==j: bw[i,j]=0
            else: bw[i,j]=1 if rng_bw.random()<0.9 else 2
    if inherit_bw is not None and inherit_bw.shape[0]==inherited_cols:
        bw[:inherited_cols,:inherited_cols]=inherit_bw

    write_connectivity(args.prefix, args.tasks, edges)
    write_exec(args.prefix, exec_mat)
    # escreve BW
    path_bw=Path(f"{args.prefix}_resource_BW.csv")
    with path_bw.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['P']+[f"P_{p}" for p in range(args.procs)])
        for i in range(args.procs):
            w.writerow([f"P_{i}"]+list(map(int,bw[i])))
    write_power(args.prefix, power_mat)
    print(f"Wrote PEFT-like scenario to prefix {args.prefix} (Inherited={inherited_cols})")
    print(f"Tasks={args.tasks}, Procs={args.procs}, Edges={len(edges)}")

if __name__ == '__main__':
    main()
