"""Minimal comparison script for heterogeneous DAG scheduling heuristics.

Inputs:
  --dag   : task connectivity (adjacency / edge weight) CSV
  --exec  : task execution time matrix CSV (rows=tasks, cols=processors)
  --bw    : processor bandwidth matrix CSV (square)
  --power : optional power matrix CSV (rows=tasks, cols=processors)
  --algos : comma-separated algorithms to run (subset of HEFT,PEFT,DLS,HEFT-LA,IHEFT,IPEFT)

Output: One JSON line per algorithm with metrics: makespan, load_balance_ratio,
communication_cost, waiting_time, energy_cost (if power provided).
If a power matrix is supplied, an Energy subplot is automatically added to the metrics figure.
"""
from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Import algorithm modules (some optional)
from heft.heft.heft import (
    readDagMatrix as heft_read_dag,
    readCsvToNumpyMatrix as heft_read_mat,
    readCsvToDict as heft_read_power,
    schedule_dag as heft_schedule,
    _compute_makespan_and_idle as heft_mk_idle,
    _compute_load_balance as heft_load,
    _compute_communication_cost as heft_comm,
    _compute_waiting_time as heft_wait,
)
from peft.peft.peft import (
    readDagMatrix as peft_read_dag,
    readCsvToNumpyMatrix as peft_read_mat,
    readCsvToDict as peft_read_power,
    schedule_dag as peft_schedule,
    _compute_makespan_and_idle as peft_mk_idle,
    _compute_load_balance as peft_load,
    _compute_communication_cost as peft_comm,
    _compute_waiting_time as peft_wait,
)
try:
    from dls.dls.dls import (
        readDagMatrix as dls_read_dag,
        readCsvToNumpyMatrix as dls_read_mat,
        readCsvToDict as dls_read_power,
        schedule_dag as dls_schedule,
        _compute_makespan_and_idle as dls_mk_idle,
        _compute_load_balance as dls_load,
        _compute_communication_cost as dls_comm,
        _compute_waiting_time as dls_wait,
    )
except Exception:
    dls_read_dag = dls_read_mat = dls_read_power = dls_schedule = dls_mk_idle = dls_load = dls_comm = dls_wait = None
try:
    from heft_la.heft_la.heft_la import (
        readDagMatrix as hla_read_dag,
        readCsvToNumpyMatrix as hla_read_mat,
        readCsvToDict as hla_read_power,
        schedule_dag as hla_schedule,
        _compute_makespan_and_idle as hla_mk_idle,
        _compute_load_balance as hla_load,
        _compute_communication_cost as hla_comm,
        _compute_waiting_time as hla_wait,
    )
except Exception:
    hla_read_dag = hla_read_mat = hla_read_power = hla_schedule = hla_mk_idle = hla_load = hla_comm = hla_wait = None
try:
    from iheft.iheft.iheft import (
        readDagMatrix as iheft_read_dag,
        readCsvToNumpyMatrix as iheft_read_mat,
        readCsvToDict as iheft_read_power,
        schedule_dag as iheft_schedule,
        _compute_makespan_and_idle as iheft_mk_idle,
        _compute_load_balance as iheft_load,
        _compute_communication_cost as iheft_comm,
        _compute_waiting_time as iheft_wait,
    )
except Exception:
    iheft_read_dag = iheft_read_mat = iheft_read_power = iheft_schedule = iheft_mk_idle = iheft_load = iheft_comm = iheft_wait = None
try:
    from ipeft.ipeft.ipeft import (
        readDagMatrix as ipeft_read_dag,
        readCsvToNumpyMatrix as ipeft_read_mat,
        readCsvToDict as ipeft_read_power,
        schedule_dag as ipeft_schedule,
        _compute_makespan_and_idle as ipeft_mk_idle,
        _compute_load_balance as ipeft_load,
        _compute_communication_cost as ipeft_comm,
        _compute_waiting_time as ipeft_wait,
    )
except Exception:
    ipeft_read_dag = ipeft_read_mat = ipeft_read_power = ipeft_schedule = ipeft_mk_idle = ipeft_load = ipeft_comm = ipeft_wait = None

@dataclass
class AlgoSpec:
    name: str
    read_dag: object
    read_mat: object
    read_power: object
    schedule: object
    mk_idle: object
    load: object
    comm: object
    wait: object

def _build_registry():
    reg={}
    def add(spec:AlgoSpec):
        if None not in (spec.read_dag, spec.read_mat, spec.schedule, spec.mk_idle):
            reg[spec.name]=spec
    add(AlgoSpec('HEFT', heft_read_dag, heft_read_mat, heft_read_power, heft_schedule, heft_mk_idle, heft_load, heft_comm, heft_wait))
    add(AlgoSpec('PEFT', peft_read_dag, peft_read_mat, peft_read_power, peft_schedule, peft_mk_idle, peft_load, peft_comm, peft_wait))
    add(AlgoSpec('DLS', dls_read_dag, dls_read_mat, dls_read_power, dls_schedule, dls_mk_idle, dls_load, dls_comm, dls_wait))
    add(AlgoSpec('HEFT-LA', hla_read_dag, hla_read_mat, hla_read_power, hla_schedule, hla_mk_idle, hla_load, hla_comm, hla_wait))
    add(AlgoSpec('IHEFT', iheft_read_dag, iheft_read_mat, iheft_read_power, iheft_schedule, iheft_mk_idle, iheft_load, iheft_comm, iheft_wait))
    add(AlgoSpec('IPEFT', ipeft_read_dag, ipeft_read_mat, ipeft_read_power, ipeft_schedule, ipeft_mk_idle, ipeft_load, ipeft_comm, ipeft_wait))
    return reg

REGISTRY=_build_registry()

# Desired presentation order
ALGO_ORDER = ['DLS','HEFT','HEFT-LA','PEFT','IHEFT','IPEFT']

def run_algo(spec:AlgoSpec, dag_file:str, exec_file:str, bw_file:str, power_file:str|None):
    dag = spec.read_dag(dag_file, show_dag=False)
    comp = spec.read_mat(exec_file)
    bw = spec.read_mat(bw_file)
    power = spec.read_power(power_file) if (power_file and spec.read_power) else None
    proc_sched, _, _ = spec.schedule(dag, computation_matrix=comp, communication_matrix=bw)
    mk,_,_ = spec.mk_idle(proc_sched)
    busy,_,_,_ = spec.load(proc_sched)
    # Average busy time per processor. Redefine load balance per user request:
    #   lb_ratio = makespan / avg_busy
    # An ideal schedule keeps every processor busy for the full makespan,
    # giving lb_ratio == 1. Values >1 indicate imbalance (idle gaps).
    avg_busy = (sum(busy.values())/len(busy)) if busy else 0.0
    lb_ratio = (mk/avg_busy) if avg_busy>0 else 0.0
    comm_cost = spec.comm(dag, proc_sched, bw)
    wait_t = spec.wait(proc_sched)
    energy = 0.0
    if power:
        for p,jobs in proc_sched.items():
            for j in jobs:
                dur = float(j.end)-float(j.start)
                if dur>0:
                    try:
                        pw = float(power[j.task][j.proc])
                    except Exception:
                        pw = 0.0
                    energy += dur*pw
    return dict(algorithm=spec.name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None), proc_sched, dag

def _save_dag_image(dag:nx.DiGraph, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,7))
    try:
        pos=nx.nx_pydot.graphviz_layout(dag, prog='dot')
    except Exception:
        pos=nx.spring_layout(dag, seed=42)
    nx.draw(dag, pos=pos, with_labels=True, node_size=300, font_size=8)
    plt.title('DAG')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _save_gantt(proc_sched:dict, path:Path, title:str):
    """Save a Gantt chart with one clearly separated horizontal lane per processor.

    Ensures y-axis shows every processor index (0..P-1) even if some have no tasks.
    Bars centered in their lane; consistent lane height.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    procs = sorted(proc_sched.keys())
    if not procs:
        return
    P = max(procs)+1  # assume processors labeled 0..P-1
    lane_height = 0.8
    fig_height = 0.35 * P + 1.5  # scale height with processor count
    plt.figure(figsize=(14, fig_height))
    colors={}
    import random
    random.seed(42)
    for p in range(P):
        jobs = proc_sched.get(p, [])
        for ev in jobs:
            if ev.task not in colors:
                colors[ev.task] = (random.random(), random.random(), random.random())
            plt.barh(y=p, width=ev.end-ev.start, left=ev.start, height=lane_height, color=colors[ev.task], edgecolor='black', linewidth=0.2)
            # Only annotate if bar wide enough
            if (ev.end-ev.start) > 0.02 * max(ev.end for js in proc_sched.values() for ev in js):
                plt.text(ev.start + (ev.end-ev.start)/2, p, str(ev.task), va='center', ha='center', fontsize=6, color='white')
    plt.yticks(range(P), [str(i) for i in range(P)])
    plt.ylim(-0.5, P-0.5)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    plt.ylabel('Processor')
    plt.xlabel('Time')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def _plot_metrics(results:list[dict], path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Order results by ALGO_ORDER
    ordered=[r for name in ALGO_ORDER for r in results if r['algorithm']==name]
    if not ordered:
        return
    names=[r['algorithm'] for r in ordered]
    makespans=[r['makespan'] for r in ordered]
    load_bal=[r['load_balance_ratio'] for r in ordered]
    waits=[r['waiting_time'] for r in ordered]
    # Determine if energy is present
    energy_vals=[r['energy_cost'] for r in ordered if r.get('energy_cost') is not None]
    include_energy = any(ev is not None and ev>0 for ev in energy_vals)
    rows = 4 if include_energy else 3
    fig, ax = plt.subplots(rows, 1, figsize=(10, 3 * rows), constrained_layout=True)

    def _plot_single(a, vals, title, color, ylabel, fmt="{v:.2f}", prefer="min"):
        bars = a.bar(names, vals, color=color, edgecolor="black", linewidth=0.4, zorder=2)
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.tick_params(axis="x", rotation=25)

        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin
        if span == 0:
            span = abs(vmax) if vmax != 0 else 1.0

        # generous margins so labels sit comfortably inside the axes
        lower = min(0, vmin - span * 0.1)
        upper = vmax + span * 0.30
        a.set_ylim(lower, upper)

        grid_alpha = 0.25
        a.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=grid_alpha, zorder=0)

        text_offset = span * 0.05

        def place_text(bar, text):
            bh = bar.get_height()
            y = bh + text_offset
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.8),
            )

        if prefer == "min":
            best_val = vmin
            for bar, val in zip(bars, vals):
                pct = (val - best_val) / best_val * 100 if best_val != 0 else 0.0
                label = "(best)" if val == best_val else f"+{pct:.2f}%"
                place_text(bar, f"{fmt.format(v=val)}\n{label}")
        elif prefer == "max":
            best_val = vmax
            for bar, val in zip(bars, vals):
                pct = (best_val - val) / best_val * 100 if best_val != 0 else 0.0
                label = "(best)" if val == best_val else f"+{pct:.2f}% worse"
                place_text(bar, f"{fmt.format(v=val)}\n{label}")
        elif prefer == "close1":
            deviations = [abs(v - 1) for v in vals]
            best_dev = min(deviations)
            for bar, val, dev in zip(bars, vals, deviations):
                pct = dev * 100
                label = "(best)" if dev == best_dev else f"dev {pct:.2f}%"
                place_text(bar, f"{fmt.format(v=val)}\n{label}")
        else:
            for bar, val in zip(bars, vals):
                place_text(bar, f"{fmt.format(v=val)}")

    _plot_single(ax[0], makespans, "Makespan", "#4C72B0", "Makespan", fmt="{v:.3f}", prefer="min")
    # Load balance ratio: Makespan / Avg Busy (>=1, closer to 1 is better)
    _plot_single(
        ax[1],
        load_bal,
        "Load Balance (Makespan / Avg Busy) – closer to 1 is better",
        "#DD8452",
        "LB Ratio",
        fmt="{v:.3f}",
        prefer="close1",
    )
    _plot_single(ax[2], waits, "Waiting Time", "#55A868", "Waiting Time", fmt="{v:.3f}")
    if include_energy:
        # Replace None with 0 for plotting clarity
        energy_clean = [(e if e is not None else 0.0) for e in energy_vals]
        _plot_single(ax[3], energy_clean, "Energy Cost", "#937860", "Energy (J)", fmt="{v:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="DAG scheduling comparison & visualization")
    ap.add_argument('--dag', required=True)
    ap.add_argument('--exec', dest='exec_file', required=True)
    ap.add_argument('--bw', required=True)
    ap.add_argument('--power')
    ap.add_argument('--algos', required=True, help='Comma separated list (subset of DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT)')
    ap.add_argument('--out_dir', help='Directory to save artifacts (DAG image, Gantts, metrics plot)')
    ap.add_argument('--save_dag', action='store_true', help='Save DAG image')
    ap.add_argument('--plot_metrics', action='store_true', help='Save aggregate metrics plot')
    args = ap.parse_args()

    requested=[a.strip().upper() for a in args.algos.split(',') if a.strip()]
    unknown=[a for a in requested if a not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unsupported or unavailable algorithms: {unknown}. Available: {sorted(REGISTRY)}")

    out_dir=Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results=[]
    proc_schedules={}
    dag_cached=None
    for name in requested:
        spec=REGISTRY[name]
        metrics, sched, dag = run_algo(spec, args.dag, args.exec_file, args.bw, args.power)
        results.append(metrics)
        proc_schedules[name]=sched
        if dag_cached is None:
            dag_cached=dag
        print(json.dumps(metrics))

    # Reorder results for visualization consistency
    ordered_names=[n for n in ALGO_ORDER if n in proc_schedules]

    if out_dir and args.save_dag and dag_cached is not None:
        _save_dag_image(dag_cached, out_dir / 'dag.png')
    # Always save gantt charts for all requested algorithms if out_dir specified
    if out_dir:
        # Preserve canonical ordering where defined, then any extras
        gantt_order = [n for n in ALGO_ORDER if n in proc_schedules] + [n for n in proc_schedules if n not in ALGO_ORDER]
        for name in gantt_order:
            _save_gantt(proc_schedules[name], out_dir / f'gantt_{name}.png', f'{name} Schedule')
    if out_dir and args.plot_metrics:
        _plot_metrics(results, out_dir / 'metrics.png')

if __name__=='__main__':
    main()
