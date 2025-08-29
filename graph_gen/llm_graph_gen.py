#!/usr/bin/env python3
"""LLM training workflow DAG generator.

Generates connectivity / execution time / bandwidth CSVs in the same format as the
existing random graph generator, but structuring tasks to approximate a single
training iteration with microbatch pipeline parallelism and gradient reduction.

Task taxonomy (indices assigned in this order):
 0  : DATA_LOAD
 1  : TOKENIZE
 2.. : Forward tasks F[s, mb]   for stage s in [0,S-1], microbatch mb in [0,M-1]
 Next: Backward tasks B[s, mb]  for stage s descending (S-1..0), microbatch mb in [0,M-1]
 Next: GRAD_REDUCE[s]           per pipeline stage (waits for all B[s,*])
 Last: OPT_STEP                 waits for all GRAD_REDUCE[*]

Dependencies encoded:
 DATA_LOAD -> TOKENIZE -> F[0,0]
 F[s,mb] -> F[s+1,mb]           (activation edge) for s < S-1
 Sequential feed of microbatches at stage 0: F[0,mb-1] -> F[0,mb]
 After F[S-1,mb] completes, start backward chain for that microbatch:
   F[S-1,mb] -> B[S-1,mb] -> B[S-2,mb] -> ... -> B[0,mb]
 Each backward stage produces gradients: B[s,mb] -> GRAD_REDUCE[s]
 All GRAD_REDUCE[s] -> OPT_STEP

Edge weights approximate bytes transferred:
  Activation edge F[s,mb] -> F[s+1,mb] : activation_size_bytes
  Forward->Backward start (F[S-1,mb] -> B[S-1,mb])                 : activation_size_bytes (reuse)
  Backward chain B[s,mb] -> B[s-1,mb]                               : activation_size_bytes/2 (smaller grad activations)
  Gradient edges B[s,mb] -> GRAD_REDUCE[s]                          : gradient_size_bytes
  Other control dependencies use weight 1 (negligible data)

Execution time matrix:
  Forward F[s,mb]: forward_flops_stage[s] / device_flops[proc]
  Backward B[s,mb]: backward_factor * forward_time
  Gradient reduction GRAD_REDUCE[s]: (gradient_size_bytes / red_bandwidth[proc]) * log(P) (rough heuristic)
  OPT_STEP: optimizer_flops / device_flops[proc]
  Data load / tokenize: simple constants scaled by IO_factor

Heterogeneity: device_flops[proc] = base_flops * (1 +/- HF * rand)
Bandwidth: symmetric random in [BW_LOW, BW_HIGH] (Gb/s units converted to bytes/sec if needed)
Note: We treat execution times already in seconds (user can scale via --time_scale).

Outputs (CSV):
  prefix_resource_BW.csv
  prefix_task_connectivity.csv
  prefix_task_exe_time.csv

This script focuses on structural fidelity; many aspects (collective scheduling,
1F1B overlap, parameter shard gathers) are simplified.
"""
import argparse, math, os, random
import numpy as np

# -------------------------- Helpers ---------------------------------

def _write_matrix_csv(path, header0, row_labels, matrix):
    with open(path, 'w') as f:
        f.write(','.join([header0] + row_labels) + '\n')
        for lbl, row in zip(row_labels, matrix):
            f.write(','.join([lbl] + [f"{x}" for x in row]) + '\n')

# -------------------------- Generation -------------------------------

def generate_llm_graph(args):
    S = args.stages
    M = args.microbatches
    P = args.resources
    HF = args.heterogeneity
    rng = random.Random(args.seed)

    # Base per-stage forward FLOPs (user may supply or default from hidden dim)
    if args.forward_flops_per_stage:
        f_flops = [float(x) for x in args.forward_flops_per_stage.split(',')]
        if len(f_flops) != S:
            raise SystemExit('--forward_flops_per_stage must provide exactly S comma-separated values')
    else:
        # Rough transformer block flops ~ 6 * hidden_dim^2 * seq_len_per_microbatch / S (simplified)
        per_stage = 6.0 * args.hidden_dim * args.hidden_dim * (args.tokens_per_microbatch / S)
        f_flops = [per_stage for _ in range(S)]

    # Activation & gradient sizes
    if args.activation_bytes_per_stage:
        act_sizes = [int(x) for x in args.activation_bytes_per_stage.split(',')]
        if len(act_sizes) != S: raise SystemExit('--activation_bytes_per_stage length mismatch')
    else:
        act_sizes = [args.tokens_per_microbatch * args.hidden_dim * 2 for _ in range(S)]  # fp16

    if args.gradient_bytes_per_stage:
        grad_sizes = [int(x) for x in args.gradient_bytes_per_stage.split(',')]
        if len(grad_sizes) != S: raise SystemExit('--gradient_bytes_per_stage length mismatch')
    else:
        grad_sizes = [args.hidden_dim * args.hidden_dim * 2 for _ in range(S)]  # param-sized approx

    optimizer_flops = args.optimizer_flops if args.optimizer_flops else sum(grad_sizes) * 1.0

    # Device flops (heterogeneous)
    base_flops = args.base_device_flops  # per second
    device_flops = []
    for p in range(P):
        scale = 1.0 + HF * (rng.uniform(-1.0, 1.0))
        device_flops.append(base_flops * scale)

    # Bandwidth matrix (symmetric), treat units as bytes/sec directly
    bw = np.zeros((P, P))
    for i in range(P):
        for j in range(i+1, P):
            val = rng.uniform(args.bw_low, args.bw_high)
            bw[i,j] = bw[j,i] = val

    # Task indexing
    def idx_data_load(): return 0
    def idx_tokenize(): return 1
    def idx_forward(s, mb): return 2 + (s + mb * S)
    forward_count = S * M
    def idx_backward(s, mb): return 2 + forward_count + ( (S - 1 - s) + mb * S )
    backward_count = S * M
    def idx_grad_reduce(s): return 2 + forward_count + backward_count + s
    def idx_opt_step(): return 2 + forward_count + backward_count + S

    total_tasks = 2 + forward_count + backward_count + S + 1

    # Connectivity matrix
    conn = np.zeros((total_tasks, total_tasks))
    # Data load chain
    conn[idx_data_load(), idx_tokenize()] = args.control_edge_weight
    # Forward microbatch feed at stage 0 & forward pipeline edges
    for mb in range(M):
        if mb == 0:
            conn[idx_tokenize(), idx_forward(0, mb)] = act_sizes[0]
        else:
            conn[idx_forward(0, mb-1), idx_forward(0, mb)] = act_sizes[0]
        for s in range(S-1):
            conn[idx_forward(s, mb), idx_forward(s+1, mb)] = act_sizes[s]
        # Forward -> start backward (at last stage)
        conn[idx_forward(S-1, mb), idx_backward(S-1, mb)] = act_sizes[S-1]
        # Backward chain
        for s in range(S-1, 0, -1):
            conn[idx_backward(s, mb), idx_backward(s-1, mb)] = max(1, act_sizes[s-1]//2)
        # Backward to gradient reduction per stage
        for s in range(S):
            conn[idx_backward(s, mb), idx_grad_reduce(s)] = grad_sizes[s]
    # Gradient reductions to optimizer
    for s in range(S):
        conn[idx_grad_reduce(s), idx_opt_step()] = args.control_edge_weight

    # Execution time matrix (tasks x P)
    exec_time = np.zeros((total_tasks, P))
    # Data tasks
    for p in range(P):
        exec_time[idx_data_load(), p] = args.data_load_time
        exec_time[idx_tokenize(), p] = args.tokenize_time
    # Forward
    for mb in range(M):
        for s in range(S):
            t = idx_forward(s, mb)
            for p in range(P):
                exec_time[t, p] = (f_flops[s] / device_flops[p]) * args.time_scale
    # Backward
    for mb in range(M):
        for s in range(S):
            t = idx_backward(s, mb)
            for p in range(P):
                exec_time[t, p] = (f_flops[s] * args.backward_factor / device_flops[p]) * args.time_scale
    # Gradient reduction (approx collective time ~ size / (min_bw) * log P)
    min_bw = np.min(bw[np.nonzero(bw)]) if np.any(bw>0) else 1.0
    logP = math.log(max(P,2), 2)
    for s in range(S):
        size = grad_sizes[s]
        base = (size / max(min_bw,1e-9)) * logP
        for p in range(P):
            exec_time[idx_grad_reduce(s), p] = base * args.time_scale
    # Optimizer step
    for p in range(P):
        exec_time[idx_opt_step(), p] = (optimizer_flops / device_flops[p]) * args.optimizer_factor * args.time_scale

    # Output CSVs
    os.makedirs(args.out, exist_ok=True)
    prefix = args.prefix
    bw_header = ['P'] + [f'P_{i}' for i in range(P)]
    bw_rows = [[f'P_{i}'] + [bw[i,j] for j in range(P)] for i in range(P)]
    # NOTE: execution time matrix rows must correspond to TASKS (T_i), not processors.
    # Previous version incorrectly wrote processor labels as rows, producing only P rows
    # and causing index errors when schedulers accessed task indices >= P.
    _write_matrix_csv(os.path.join(args.out, f'{prefix}_resource_BW.csv'), 'P', [f'P_{i}' for i in range(P)], bw)
    _write_matrix_csv(os.path.join(args.out, f'{prefix}_task_connectivity.csv'), 'T', [f'T_{i}' for i in range(total_tasks)], conn)
    # Execution time CSV expected format (by existing readers): first header token 'TP',
    # subsequent columns P_0..P_{P-1}; each row label T_i followed by execution times on each processor.
    exec_out = exec_time  # shape (tasks, P)
    with open(os.path.join(args.out, f'{prefix}_task_exe_time.csv'), 'w') as f:
        f.write('TP,' + ','.join([f'P_{j}' for j in range(P)]) + '\n')
        for ti in range(total_tasks):
            f.write(f'T_{ti},' + ','.join(str(exec_out[ti, pj]) for pj in range(P)) + '\n')

    # Human-readable task map
    with open(os.path.join(args.out, f'{prefix}_task_map.txt'), 'w') as f:
        f.write('Index,Label\n')
        f.write('0,DATA_LOAD\n')
        f.write('1,TOKENIZE\n')
        for mb in range(M):
            for s in range(S):
                f.write(f"{idx_forward(s,mb)},FWD[s={s},mb={mb}]\n")
        for mb in range(M):
            for s in range(S-1,-1,-1):
                f.write(f"{idx_backward(s,mb)},BWD[s={s},mb={mb}]\n")
        for s in range(S):
            f.write(f"{idx_grad_reduce(s)},GRAD_REDUCE[s={s}]\n")
        f.write(f"{idx_opt_step()},OPT_STEP\n")

    print(f"Generated LLM graph with {total_tasks} tasks; output prefix '{prefix}' in {args.out}")

# -------------------------- CLI -------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Structured LLM training DAG generator')
    p.add_argument('--stages', type=int, default=4, help='Pipeline stages S')
    p.add_argument('--microbatches', type=int, default=4, help='Microbatches M in one accumulation window')
    p.add_argument('--resources', type=int, default=4, help='Number of devices / processors P')
    p.add_argument('--heterogeneity', type=float, default=0.2, help='Heterogeneity factor HF (0..1) for device performance scaling')
    p.add_argument('--hidden_dim', type=int, default=4096, help='Hidden dimension (used for default flops & sizes)')
    p.add_argument('--tokens_per_microbatch', type=int, default=1024, help='Tokens per microbatch')
    p.add_argument('--forward_flops_per_stage', type=str, help='Comma list of FLOPs per stage (overrides heuristic)')
    p.add_argument('--activation_bytes_per_stage', type=str, help='Comma list activation sizes per stage (bytes)')
    p.add_argument('--gradient_bytes_per_stage', type=str, help='Comma list gradient sizes per stage (bytes)')
    p.add_argument('--backward_factor', type=float, default=2.0, help='Backward FLOPs multiplier relative to forward')
    p.add_argument('--optimizer_flops', type=float, help='Total FLOPs for optimizer step (overrides gradient-size heuristic)')
    p.add_argument('--optimizer_factor', type=float, default=0.2, help='Multiplier applied to optimizer_flops/device_flops to get time')
    p.add_argument('--base_device_flops', type=float, default=2.0e14, help='Baseline device TFLOPs expressed in FLOPs/s (e.g., 2e14 ~ 200 TFLOP/s)')
    p.add_argument('--bw_low', type=float, default=5e11, help='Lower bound bandwidth (bytes/sec)')
    p.add_argument('--bw_high', type=float, default=1e12, help='Upper bound bandwidth (bytes/sec)')
    p.add_argument('--data_load_time', type=float, default=0.02, help='Data load task time (s)')
    p.add_argument('--tokenize_time', type=float, default=0.01, help='Tokenization task time (s)')
    p.add_argument('--time_scale', type=float, default=1.0, help='Global multiplier for all compute times')
    p.add_argument('--control_edge_weight', type=int, default=1, help='Edge weight for pure control dependencies')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--out', type=str, default='graphs')
    p.add_argument('--prefix', type=str, default='llm')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    generate_llm_graph(args)
