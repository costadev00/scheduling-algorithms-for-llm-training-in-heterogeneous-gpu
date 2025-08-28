"""Run HEFT and PEFT under identical input matrices and report comparable metrics.

Usage (PowerShell):
    python compare_same_dataset.py --dag graphs/canonicalgraph_task_connectivity.csv \
            --exec graphs/canonicalgraph_task_exe_time.csv --bw graphs/canonicalgraph_resource_BW.csv \
            --power graphs/canonicalgraph_task_power.csv --tag canonical --out_csv comparisons_canonical.csv

Outputs a row per algorithm with: makespan, load_balance_ratio, communication_cost, waiting_time, energy(if power), tag(optional).
If appending to an existing CSV without a 'tag' column, the script keeps the old 6-column format.
"""
from __future__ import annotations
import argparse, math, csv, json
from pathlib import Path

from heft.heft.heft import (
    readCsvToNumpyMatrix as heft_read_mat,
    readDagMatrix as heft_read_dag,
    readCsvToDict as heft_read_power,
    schedule_dag as heft_schedule,
    _compute_makespan_and_idle as heft_mk_idle,
    _compute_load_balance as heft_load,
    _compute_communication_cost as heft_comm,
    _compute_waiting_time as heft_wait,
)
from peft.peft.peft import (
    readCsvToNumpyMatrix as peft_read_mat,
    readDagMatrix as peft_read_dag,
    readCsvToDict as peft_read_power,
    schedule_dag as peft_schedule,
    _compute_makespan_and_idle as peft_mk_idle,
    _compute_load_balance as peft_load,
    _compute_communication_cost as peft_comm,
    _compute_waiting_time as peft_wait,
)
try:
    from heft_la.heft_la.heft_la import (
        readCsvToNumpyMatrix as hla_read_mat,
        readDagMatrix as hla_read_dag,
        readCsvToDict as hla_read_power,  # power ignored
        schedule_dag as hla_schedule,
        _compute_makespan_and_idle as hla_mk_idle,
        _compute_load_balance as hla_load,
        _compute_communication_cost as hla_comm,
        _compute_waiting_time as hla_wait,
    )
except Exception:
    hla_read_mat = hla_read_dag = hla_read_power = hla_schedule = hla_mk_idle = hla_load = hla_comm = hla_wait = None
try:
    from iheft.iheft.iheft import (
        readCsvToNumpyMatrix as iheft_read_mat,
        readDagMatrix as iheft_read_dag,
        readCsvToDict as iheft_read_power,
        schedule_dag as iheft_schedule,
        _compute_makespan_and_idle as iheft_mk_idle,
        _compute_load_balance as iheft_load,
        _compute_communication_cost as iheft_comm,
        _compute_waiting_time as iheft_wait,
    )
except Exception:
    iheft_read_mat = iheft_read_dag = iheft_read_power = iheft_schedule = iheft_mk_idle = iheft_load = iheft_comm = iheft_wait = None
try:
    from dls.dls.dls import (
        readCsvToNumpyMatrix as dls_read_mat,
        readDagMatrix as dls_read_dag,
        readCsvToDict as dls_read_power,
        schedule_dag as dls_schedule,
        _compute_makespan_and_idle as dls_mk_idle,
        _compute_load_balance as dls_load,
        _compute_communication_cost as dls_comm,
        _compute_waiting_time as dls_wait,
    )
except Exception:
    dls_read_mat = dls_read_dag = dls_read_power = dls_schedule = dls_mk_idle = dls_load = dls_comm = dls_wait = None
try:
    from heft.heft.gantt import showGanttChart as heft_show_gantt
    from peft.peft.gantt import showGanttChart as peft_show_gantt
except Exception:
    heft_show_gantt = peft_show_gantt = None

def run_algo(name:str, dag_file:str, exec_file:str, bw_file:str, power_file:str|None, return_schedule:bool=False):
    if name == 'HEFT':
        dag = heft_read_dag(dag_file, show_dag=False)
        comp = heft_read_mat(exec_file)
        bw = heft_read_mat(bw_file)
        power = heft_read_power(power_file) if power_file else None
        proc_sched, _, _ = heft_schedule(dag, computation_matrix=comp, communication_matrix=bw)
        mk,_,_ = heft_mk_idle(proc_sched)
        busy,_,_,_ = heft_load(proc_sched)
        avg_busy = (sum(busy.values())/len(busy)) if busy else math.inf
        lb_ratio = mk/avg_busy if avg_busy>0 else math.inf
        comm_cost = heft_comm(dag, proc_sched, bw, None)
        wait_t = heft_wait(proc_sched)
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
        metrics = dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
        return (metrics, proc_sched) if return_schedule else metrics
    elif name == 'PEFT':
        dag = peft_read_dag(dag_file, show_dag=False)
        comp = peft_read_mat(exec_file)
        bw = peft_read_mat(bw_file)
        power = peft_read_power(power_file) if power_file else None
        proc_sched, _, _ = peft_schedule(dag, computation_matrix=comp, communication_matrix=bw)
        mk,_,_ = peft_mk_idle(proc_sched)
        busy,_,_,_ = peft_load(proc_sched)
        avg_busy = (sum(busy.values())/len(busy)) if busy else math.inf
        lb_ratio = mk/avg_busy if avg_busy>0 else math.inf
        comm_cost = peft_comm(dag, proc_sched, bw)
        wait_t = peft_wait(proc_sched)
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
        metrics = dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
        return (metrics, proc_sched) if return_schedule else metrics
    elif name == 'DLS':
        if dls_schedule is None:
            raise RuntimeError("DLS module not available")
        dag = dls_read_dag(dag_file, show_dag=False)
        comp = dls_read_mat(exec_file)
        bw = dls_read_mat(bw_file)
        power = dls_read_power(power_file) if power_file else None
        proc_sched, _, _ = dls_schedule(dag, computation_matrix=comp, communication_matrix=bw)
        mk,_,_ = dls_mk_idle(proc_sched)
        busy,_,_,_ = dls_load(proc_sched)
        avg_busy = (sum(busy.values())/len(busy)) if busy else math.inf
        lb_ratio = mk/avg_busy if avg_busy>0 else math.inf
        comm_cost = dls_comm(dag, proc_sched, bw)
        wait_t = dls_wait(proc_sched)
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
        metrics = dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
        return (metrics, proc_sched) if return_schedule else metrics
    elif name == 'HEFT-LA':
        if hla_schedule is None:
            raise RuntimeError("HEFT-LA module not available")
        dag = hla_read_dag(dag_file, show_dag=False)
        comp = hla_read_mat(exec_file)
        bw = hla_read_mat(bw_file)
        power = hla_read_power(power_file) if power_file else None
        proc_sched, _, _ = hla_schedule(dag, computation_matrix=comp, communication_matrix=bw)
        mk,_,_ = hla_mk_idle(proc_sched)
        busy,_,_,_ = hla_load(proc_sched)
        avg_busy = (sum(busy.values())/len(busy)) if busy else math.inf
        lb_ratio = mk/avg_busy if avg_busy>0 else math.inf
        comm_cost = hla_comm(dag, proc_sched, bw, None)
        wait_t = hla_wait(proc_sched)
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
        metrics = dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
        return (metrics, proc_sched) if return_schedule else metrics
    elif name == 'IHEFT':
        if iheft_schedule is None:
            raise RuntimeError("IHEFT module not available")
        dag = iheft_read_dag(dag_file, show_dag=False)
        comp = iheft_read_mat(exec_file)
        bw = iheft_read_mat(bw_file)
        power = iheft_read_power(power_file) if power_file else None
        proc_sched, _, _ = iheft_schedule(dag, computation_matrix=comp, communication_matrix=bw)
        mk,_,_ = iheft_mk_idle(proc_sched)
        busy,_,_,_ = iheft_load(proc_sched)
        avg_busy = (sum(busy.values())/len(busy)) if busy else math.inf
        lb_ratio = mk/avg_busy if avg_busy>0 else math.inf
        comm_cost = iheft_comm(dag, proc_sched, bw)
        wait_t = iheft_wait(proc_sched)
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
        metrics = dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
        return (metrics, proc_sched) if return_schedule else metrics
    else:
        raise ValueError(f"Unknown algorithm name: {name}")

def main():
    ap = argparse.ArgumentParser(description="Compare HEFT vs PEFT on identical inputs")
    ap.add_argument('--dag', required=True)
    ap.add_argument('--exec', dest='exec_file', required=True)
    ap.add_argument('--bw', required=True)
    ap.add_argument('--power', help='optional power matrix CSV')
    ap.add_argument('--out_csv', help='optional CSV to append results')
    ap.add_argument('--tag', help='optional dataset/run tag written to CSV')
    ap.add_argument('--algos', default='HEFT,PEFT', help='comma-separated list of algorithms to run (supported: HEFT,PEFT,DLS,HEFT-LA,IHEFT)')
    ap.add_argument('--showGantt', action='store_true', help='display Gantt chart(s) for each algorithm')
    ap.add_argument('--save_gantt_dir', help='directory to save Gantt PNGs instead of (or in addition to) showing them')
    ap.add_argument('--report', action='store_true', help='print expanded JSON metrics report')
    ap.add_argument('--json_out', help='optional JSON file to write full metrics list')
    args = ap.parse_args()

    algo_list=[a.strip().upper() for a in args.algos.split(',') if a.strip()]
    unsupported=[a for a in algo_list if a not in {'HEFT','PEFT','DLS','HEFT-LA','IHEFT'}]
    if unsupported:
        raise SystemExit(f"Unsupported algorithms requested: {unsupported}")

    rows=[]; schedules={}
    for algo in algo_list:
        res = run_algo(algo, args.dag, args.exec_file, args.bw, args.power, return_schedule=args.showGantt or args.save_gantt_dir is not None)
        if isinstance(res, tuple):
            metrics, sched = res
            schedules[algo]=sched
        else:
            metrics=res
        if args.tag:
            metrics['tag']=args.tag
        rows.append(metrics)

    # Basic print
    for r in rows:
        print(r if not args.report else json.dumps(r, indent=2))

    # Optional JSON aggregate
    if args.json_out:
        with open(args.json_out,'w') as jf:
            json.dump(rows, jf, indent=2)

    # Gantt handling
    if (args.showGantt or args.save_gantt_dir) and schedules:
        import matplotlib.pyplot as plt
        save_dir = Path(args.save_gantt_dir) if args.save_gantt_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        for algo, sched in schedules.items():
            # choose appropriate gantt renderer
            renderer = heft_show_gantt if algo=='HEFT' else peft_show_gantt
            if renderer is None:
                continue
            if args.showGantt:
                renderer(sched)
            if save_dir:
                # Render to figure and save (renderer already shows, so create manual figure if only saving)
                if not args.showGantt:
                    renderer(sched)
                fname = f"{args.tag+'_' if args.tag else ''}{algo}_gantt.png"
                try:
                    plt.savefig(save_dir / fname, dpi=150, bbox_inches='tight')
                finally:
                    plt.close()

    if args.out_csv:
        out_path = Path(args.out_csv)
        write_header = not out_path.exists()
        existing_has_tag = False
        if not write_header:
            try:
                with out_path.open('r', newline='') as rf:
                    first = rf.readline().strip().split(',')
                    existing_has_tag = 'tag' in first
            except Exception:
                pass
        with out_path.open('a', newline='') as f:
            w = csv.writer(f)
            header_full = ['algorithm','makespan','load_balance_ratio','communication_cost','waiting_time','energy_cost','tag']
            header_legacy = header_full[:-1]  # without tag
            if write_header:
                w.writerow(header_full if any('tag' in r for r in rows) else header_legacy)
                existing_has_tag = any('tag' in r for r in rows)
            for r in rows:
                if existing_has_tag:
                    w.writerow([
                        r.get('algorithm'),
                        r.get('makespan'),
                        r.get('load_balance_ratio'),
                        r.get('communication_cost'),
                        r.get('waiting_time'),
                        r.get('energy_cost'),
                        r.get('tag') if 'tag' in r else None
                    ])
                else:
                    w.writerow([
                        r.get('algorithm'),
                        r.get('makespan'),
                        r.get('load_balance_ratio'),
                        r.get('communication_cost'),
                        r.get('waiting_time'),
                        r.get('energy_cost')
                    ])

if __name__ == '__main__':
    main()
