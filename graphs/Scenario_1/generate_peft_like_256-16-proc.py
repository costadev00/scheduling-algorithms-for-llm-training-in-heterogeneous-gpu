"""Generate a PEFT-like heterogeneous DAG with 256 tasks and 16 processors.

Enhancement: can inherit previously generated matrices (connectivity, exec time,
bandwidth, power) from a smaller number de processadores (ex.: 8) para garantir
comparabilidade entre cenários. Use --inherit-prefix apontando para o prefixo
anterior (sem sufixo _task_exe_time.csv etc.). As primeiras colunas (P_0..P_{k-1})
dos arquivos gerados serão idênticas às herdadas; novas colunas (k..15) são
geradas mantendo o mesmo modelo estocástico.

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
import argparse, random, math, csv, sys
from pathlib import Path
import numpy as np

DEF_TASKS = 256
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

def _load_inherit_exec(path_prefix: str):
    path = Path(f"{path_prefix}_task_exe_time.csv")
    if not path.exists():
        return None
    import numpy as np
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        procs = [h for h in header if h.startswith('P_')]
        rows = []
        for r in reader:
            rows.append([int(x) for x in r[1:]])
    mat = np.array(rows, dtype=int)
    return mat

def _load_inherit_power(path_prefix: str):
    path = Path(f"{path_prefix}_task_power.csv")
    if not path.exists():
        return None
    import numpy as np
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for r in reader:
            rows.append([float(x) for x in r[1:]])
    return np.array(rows, dtype=float)

def _load_inherit_bw(path_prefix: str):
    path = Path(f"{path_prefix}_resource_BW.csv")
    if not path.exists():
        return None
    import numpy as np
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for r in reader:
            rows.append([int(x) for x in r[1:]])
    return np.array(rows, dtype=int)

def _load_inherit_connectivity(path_prefix: str):
    path = Path(f"{path_prefix}_task_connectivity.csv")
    if not path.exists():
        return None
    edges=[]
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        n_tasks = len(header)-1
        rows = list(reader)
        for i,row in enumerate(rows):
            src = i
            weights = row[1:]
            for j,w in enumerate(weights):
                val = int(w)
                if val>0:
                    edges.append((src,j,val))
    return edges, n_tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, default=DEF_TASKS)
    ap.add_argument('--procs', type=int, default=DEF_PROCS)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', required=True, help='Output file prefix, e.g. graphs/Scenario_1/peft256_16proc')
    ap.add_argument('--inherit-prefix', help='Prefix dos arquivos de um cenário menor (ex.: graphs/Scenario_1/peft256_8proc)')
    ap.add_argument('--tolerance', type=float, default=0.0, help='Se >0, valida colunas herdadas vs geradas (diferença máxima permitida)')
    args=ap.parse_args()

    inherit_exec = inherit_power = inherit_bw = inherit_edges = None
    inherit_tasks = None
    if args.inherit_prefix:
        inherit_exec = _load_inherit_exec(args.inherit_prefix)
        inherit_power = _load_inherit_power(args.inherit_prefix)
        inherit_bw = _load_inherit_bw(args.inherit_prefix)
        conn = _load_inherit_connectivity(args.inherit_prefix)
        if conn:
            inherit_edges, inherit_tasks = conn
        if inherit_exec is None:
            print(f"[warn] Não foi possível carregar exec de {args.inherit_prefix}", file=sys.stderr)

    # DAG: reutiliza se herdado e mesmo número de tarefas
    if inherit_edges and inherit_tasks == args.tasks:
        edges = inherit_edges
        print("Reutilizando conectividade herdada.")
    else:
        layers, edges = build_dag(args.tasks, args.layers, seed=args.seed)

    exec_mat, perf = gen_exec_matrix(args.tasks, args.procs, seed=args.seed+1)

    # Se herdando exec, garantir colunas idênticas
    if inherit_exec is not None:
        base_procs = inherit_exec.shape[1]
        if base_procs >= args.procs:
            print(f"[erro] base_procs={base_procs} >= novo procs={args.procs}", file=sys.stderr)
            sys.exit(1)
        if inherit_exec.shape[0] != args.tasks:
            print(f"[erro] Número de tasks diferente na herança ({inherit_exec.shape[0]} vs {args.tasks})", file=sys.stderr)
            sys.exit(1)
        # opcional: verificar diferença vs geração (para debug)
        if args.tolerance >= 0 and args.tolerance < float('inf'):
            import numpy as np
            diff = np.abs(exec_mat[:,:base_procs] - inherit_exec)
            max_diff = diff.max()
            if max_diff > args.tolerance:
                print(f"[info] Substituindo colunas herdadas (desvio gerado={max_diff})")
        exec_mat[:, :base_procs] = inherit_exec
        inherited_cols = base_procs
    else:
        inherited_cols = 0

    # Power completo com exec final (para correlação correta)
    power_mat = gen_power_matrix(exec_mat, perf, seed=args.seed+2)
    if inherit_power is not None:
        if inherit_power.shape[0] == args.tasks and inherit_power.shape[1] == inherited_cols:
            power_mat[:, :inherited_cols] = inherit_power
        else:
            print('[warn] Power herdado incompatível; ignorando.', file=sys.stderr)

    # Bandwidth
    bw_seed = args.seed+3
    rng_bw = random.Random(bw_seed)
    import numpy as np
    bw_mat = np.zeros((args.procs, args.procs), dtype=int)
    for i in range(args.procs):
        for j in range(args.procs):
            if i==j:
                bw_mat[i,j]=0
            else:
                bw_mat[i,j] = 1 if rng_bw.random()<0.9 else 2
    if inherit_bw is not None:
        base = inherit_bw.shape[0]
        if base == inherited_cols and inherit_bw.shape[1] == base:
            bw_mat[:base,:base] = inherit_bw
        else:
            print('[warn] Bandwidth herdado incompatível; ignorando.', file=sys.stderr)

    # Escrita arquivos
    write_connectivity(args.prefix, args.tasks, edges)
    write_exec(args.prefix, exec_mat)
    # sobrescrever write_bw para usar nossa matriz pronta
    # Salvando banda
    bw_path = Path(f"{args.prefix}_resource_BW.csv")
    with bw_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P'] + [f"P_{p}" for p in range(args.procs)])
        for i in range(args.procs):
            w.writerow([f"P_{i}"] + list(map(int, bw_mat[i])))
    write_power(args.prefix, power_mat)
    print(f"Wrote PEFT-like scenario to prefix {args.prefix}")
    print(f"Tasks={args.tasks}, Procs={args.procs}, Inherited={inherited_cols}, Edges={len(edges)}")

if __name__ == '__main__':
    main()
