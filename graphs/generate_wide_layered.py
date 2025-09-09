import argparse, random
from pathlib import Path
import numpy as np

def partition_layers(num_tasks:int, num_layers:int):
    base = num_tasks // num_layers
    rem = num_tasks % num_layers
    layers = []
    start = 0
    for i in range(num_layers):
        sz = base + (1 if i < rem else 0)
        layers.append(list(range(start, start+sz)))
        start += sz
    return layers

def generate_wide_layered(tasks:int, procs:int, layers:int, edge_prob:float, seed:int, prefix:Path):
    random.seed(seed)
    np.random.seed(seed)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    layer_sets = partition_layers(tasks, layers)
    adj = np.zeros((tasks, tasks), dtype=int)
    # Connect adjacent layers densely according to probability
    for li in range(len(layer_sets)-1):
        src_layer = layer_sets[li]
        dst_layer = layer_sets[li+1]
        for u in src_layer:
            connected = 0
            for v in dst_layer:
                if random.random() < edge_prob:
                    adj[u, v] = random.randint(8, 28)
                    connected += 1
            # ensure at least one edge forward
            if connected == 0:
                v = random.choice(dst_layer)
                adj[u, v] = random.randint(8, 28)

    # Execution times: layer-dependent slight scaling to avoid everything equal
    exec_matrix = np.zeros((tasks, procs), dtype=float)
    layer_speed = [random.uniform(0.85, 1.15) for _ in layer_sets]
    proc_speed = [random.uniform(0.75, 1.25) for _ in range(procs)]
    for li, layer in enumerate(layer_sets):
        for t in layer:
            base = random.randint(10, 24) * layer_speed[li]
            for p in range(procs):
                val = base / proc_speed[p] * random.uniform(0.9, 1.1)
                exec_matrix[t, p] = max(1.0, round(val, 2))

    # Power: correlate with mean exec per task
    power = np.zeros_like(exec_matrix)
    mean_exec = exec_matrix.mean(axis=1)
    for t in range(tasks):
        for p in range(procs):
            # Slight variation per processor
            power[t, p] = round(random.uniform(0.7, 1.3) * (1.2 + 0.15 * mean_exec[t] / mean_exec.max()), 2)

    # Bandwidth matrix: fully connected with moderate heterogeneity
    bw = np.zeros((procs, procs), dtype=float)
    for i in range(procs):
        for j in range(procs):
            if i == j:
                bw[i, j] = 0.0
            else:
                bw[i, j] = round(random.uniform(0.9, 1.4), 3)

    # Write files
    def write_connectivity():
        path = prefix.with_name(prefix.name + '_task_connectivity.csv')
        with open(path, 'w') as f:
            f.write('T,' + ','.join(f'T{i}' for i in range(tasks)) + '\n')
            for i in range(tasks):
                f.write(f'T{i},' + ','.join(str(adj[i, j]) for j in range(tasks)) + '\n')
        return path

    def write_exec():
        path = prefix.with_name(prefix.name + '_task_exe_time.csv')
        with open(path, 'w') as f:
            f.write('TP,' + ','.join(f'P_{p}' for p in range(procs)) + '\n')
            for t in range(tasks):
                f.write(f'T_{t},' + ','.join(str(int(round(exec_matrix[t, p]))) for p in range(procs)) + '\n')
        return path

    def write_power():
        path = prefix.with_name(prefix.name + '_task_power.csv')
        with open(path, 'w') as f:
            f.write('TP,' + ','.join(f'P_{p}' for p in range(procs)) + '\n')
            for t in range(tasks):
                f.write(f'T_{t},' + ','.join(f'{power[t, p]:.2f}' for p in range(procs)) + '\n')
        return path

    def write_bw():
        path = prefix.with_name(prefix.name + '_resource_BW.csv')
        with open(path, 'w') as f:
            f.write('P,' + ','.join(f'P_{p}' for p in range(procs)) + '\n')
            for i in range(procs):
                f.write(f'P_{i},' + ','.join(str(bw[i, j]) for j in range(procs)) + '\n')
        return path

    def write_map():
        path = prefix.with_name(prefix.name + '_task_map.txt')
        with open(path, 'w') as f:
            for t in range(tasks):
                f.write(f'{t} T_{t}\n')
        return path

    paths = {
        'connectivity': str(write_connectivity()),
        'exec': str(write_exec()),
        'power': str(write_power()),
        'bw': str(write_bw()),
        'map': str(write_map()),
        'layers': layers,
        'edge_prob': edge_prob
    }
    return paths

def main():
    ap = argparse.ArgumentParser(description='Generate wide layered DAG')
    ap.add_argument('--tasks', type=int, default=256)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--layers', type=int, default=16)
    ap.add_argument('--edge_prob', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', required=True)
    args = ap.parse_args()
    info = generate_wide_layered(args.tasks, args.procs, args.layers, args.edge_prob, args.seed, Path(args.prefix))
    print(info)

if __name__ == '__main__':
    main()
