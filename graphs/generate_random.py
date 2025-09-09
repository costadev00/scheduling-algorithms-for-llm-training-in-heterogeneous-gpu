"""Generate random DAG workload CSVs compatible with compare_same_dataset.py.

Outputs:
- <prefix>_task_connectivity.csv  (NxN, header T,T_0..; weights=float bytes; acyclic)
- <prefix>_task_exe_time.csv      (N x P, header TP,P_0..; ints)
- <prefix>_task_power.csv         (N x P, header TP,P_0..; floats)
- <prefix>_resource_BW.csv        (P x P, header P,P_0..; symmetric, diag=0)
- <prefix>_task_map.txt           (Index,Label)

Default ranges aim to produce reasonable hetero compute/comm.
"""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path

def _write_csv_square(path:Path, header0:str, n:int, rows:list[list[float]], row_prefix:str):
    path.parent.mkdir(parents=True, exist_ok=True)
    header=[header0] + [f"{row_prefix}_{i}" for i in range(n)]
    with open(path, 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([f"{row_prefix}_{i}"] + [f"{rows[i][j]:.6f}" for j in range(n)])

def _write_exec_or_power(path:Path, n:int, p:int, rows:list[list[float]], as_int:bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    header=['TP'] + [f"P_{i}" for i in range(p)]
    with open(path, 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(header)
        for i in range(n):
            if as_int:
                w.writerow([f"T_{i}"] + [str(int(round(v))) for v in rows[i]])
            else:
                w.writerow([f"T_{i}"] + [f"{v:.3f}" for v in rows[i]])

def _write_map(path:Path, labels:list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',newline='') as f:
        f.write("Index,Label\n")
        for i,lab in enumerate(labels):
            f.write(f"{i},{lab}\n")

def generate_random_dag(n:int, p:int, edge_prob:float, wmin:float, wmax:float, seed:int):
    random.seed(seed)
    # Create a random topological order and add edges i->j only if order[i] < order[j]
    order=list(range(n)); random.shuffle(order)
    pos={order[i]: i for i in range(n)}
    weights=[[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if pos[i] < pos[j]:
                if random.random() < edge_prob:
                    weights[i][j] = random.uniform(wmin, wmax)
    return weights

def generate_exec(n:int, p:int, emin:float, emax:float, seed:int):
    random.seed(seed+17)
    rows=[[0.0 for _ in range(p)] for _ in range(n)]
    # mild heterogeneity across processors
    speed=[1.0 + 0.25*(i/(p-1) if p>1 else 0.0) for i in range(p)]
    for t in range(n):
        base=random.uniform(emin, emax)
        for proc in range(p):
            jitter = 0.85 + 0.3*random.random()
            rows[t][proc] = base * jitter / speed[proc]
    return rows

def generate_power(n:int, p:int, pmin:float, pmax:float, seed:int):
    random.seed(seed+33)
    rows=[[0.0 for _ in range(p)] for _ in range(n)]
    for t in range(n):
        base=random.uniform(pmin, pmax)
        for proc in range(p):
            jitter = 0.9 + 0.2*random.random()
            rows[t][proc] = base * jitter
    return rows

def generate_bw(p:int, bmin:float, bmax:float, seed:int):
    random.seed(seed+49)
    m=[[0.0 for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            v=random.uniform(bmin, bmax)
            m[i][j]=m[j][i]=v
    return m

def main():
    ap=argparse.ArgumentParser(description='Generate random DAG and matrices')
    ap.add_argument('--tasks', type=int, required=True)
    ap.add_argument('--procs', type=int, required=True)
    ap.add_argument('--edge_prob', type=float, default=0.02)
    ap.add_argument('--weight_min', type=float, default=1_000_000.0)
    ap.add_argument('--weight_max', type=float, default=32_000_000.0)
    ap.add_argument('--exec_min', type=float, default=50.0)
    ap.add_argument('--exec_max', type=float, default=200.0)
    ap.add_argument('--power_min', type=float, default=5.0)
    ap.add_argument('--power_max', type=float, default=15.0)
    ap.add_argument('--bw_min', type=float, default=3.0e11)
    ap.add_argument('--bw_max', type=float, default=1.2e12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', type=str, required=True)
    args=ap.parse_args()

    labels=[f"TASK_{i}" for i in range(args.tasks)]
    dag = generate_random_dag(args.tasks, args.procs, args.edge_prob, args.weight_min, args.weight_max, args.seed)
    exec_times = generate_exec(args.tasks, args.procs, args.exec_min, args.exec_max, args.seed)
    power = generate_power(args.tasks, args.procs, args.power_min, args.power_max, args.seed)
    bw = generate_bw(args.procs, args.bw_min, args.bw_max, args.seed)

    pref=Path(args.prefix)
    _write_csv_square(pref.parent / f"{pref.name}_task_connectivity.csv", 'T', args.tasks, dag, 'T')
    _write_exec_or_power(pref.parent / f"{pref.name}_task_exe_time.csv", args.tasks, args.procs, exec_times, as_int=True)
    _write_exec_or_power(pref.parent / f"{pref.name}_task_power.csv", args.tasks, args.procs, power, as_int=False)
    _write_csv_square(pref.parent / f"{pref.name}_resource_BW.csv", 'P', args.procs, bw, 'P')
    _write_map(pref.parent / f"{pref.name}_task_map.txt", labels)

    print({'prefix': str(pref), 'tasks': args.tasks, 'procs': args.procs})

if __name__=='__main__':
    main()
