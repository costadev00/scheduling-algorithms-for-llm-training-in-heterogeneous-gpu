"""Run HEFT and PEFT under identical input matrices and report comparable metrics.

Usage (PowerShell):
  python compare_same_dataset.py --dag graphs/canonicalgraph_task_connectivity.csv \
      --exec graphs/canonicalgraph_task_exe_time.csv --bw graphs/canonicalgraph_resource_BW.csv \
      --power graphs/canonicalgraph_task_power.csv

Outputs a row per algorithm with: makespan, load_balance_ratio, communication_cost, waiting_time, energy(if power).
"""
from __future__ import annotations
import argparse, math, csv
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

def run_algo(name:str, dag_file:str, exec_file:str, bw_file:str, power_file:str|None):
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
                        try: pw = float(power[j.task][j.proc])
                        except Exception: pw = 0.0
                        energy += dur*pw
        return dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)
    else:
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
                        try: pw = float(power[j.task][j.proc])
                        except Exception: pw = 0.0
                        energy += dur*pw
        return dict(algorithm=name, makespan=mk, load_balance_ratio=lb_ratio, communication_cost=comm_cost, waiting_time=wait_t, energy_cost=energy if power else None)

def main():
    ap = argparse.ArgumentParser(description="Compare HEFT vs PEFT on identical inputs")
    ap.add_argument('--dag', required=True)
    ap.add_argument('--exec', dest='exec_file', required=True)
    ap.add_argument('--bw', required=True)
    ap.add_argument('--power', help='optional power matrix CSV')
    ap.add_argument('--out_csv', help='optional CSV to append results')
    args = ap.parse_args()

    rows=[]
    for algo in ['HEFT','PEFT']:
        metrics = run_algo(algo, args.dag, args.exec_file, args.bw, args.power)
        rows.append(metrics)

    for r in rows:
        print(r)

    if args.out_csv:
        out_path = Path(args.out_csv)
        write_header = not out_path.exists()
        with out_path.open('a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['algorithm','makespan','load_balance_ratio','communication_cost','waiting_time','energy_cost'])
            for r in rows:
                w.writerow([r['algorithm'], r['makespan'], r['load_balance_ratio'], r['communication_cost'], r['waiting_time'], r['energy_cost']])

if __name__ == '__main__':
    main()
