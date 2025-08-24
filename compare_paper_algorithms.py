"""Utility script to run paper-only HEFT and PEFT implementations on a given set of CSV matrices
and emit a simple CSV with metrics (makespan, load_balance_ratio, energy).

Usage (PowerShell example):
  python compare_paper_algorithms.py ^
     --algo heft --dag graphs/canonicalgraph_task_connectivity.csv ^
     --exec graphs/canonicalgraph_task_exe_time.csv ^
     --bw graphs/canonicalgraph_resource_BW.csv ^
     --out results_heft.csv

  python compare_paper_algorithms.py ^
     --algo peft --dag graphs/peftgraph_task_connectivity.csv ^
     --exec graphs/peftgraph_task_exe_time.csv ^
     --bw graphs/peftgraph_resource_BW.csv ^
     --out results_peft.csv

Optionally add --power for energy reporting.
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
import networkx as nx
from pathlib import Path

from heft.heft.heft_paper import schedule_heft_paper
from peft.peft.peft_paper import schedule_peft_paper

def read_csv_matrix(path: str):
    with open(path) as f:
        lines = [l.strip().split(',') for l in f.read().strip().splitlines() if l.strip()]
    arr = np.array(lines)[1:,1:]
    return arr.astype(float)

def build_dag(connectivity_matrix: np.ndarray):
    g = nx.DiGraph(connectivity_matrix)
    g.remove_edges_from([e for e in g.edges() if g.get_edge_data(*e)['weight'] == '0.0'])
    return g

def main():
    p = argparse.ArgumentParser(description="Compare paper-only HEFT/PEFT")
    p.add_argument('--algo', choices=['heft','peft'], required=True)
    p.add_argument('--dag', required=True, help='Connectivity matrix CSV (header row+col)')
    p.add_argument('--exec', dest='exec_file', required=True, help='Execution time matrix CSV')
    p.add_argument('--bw', required=True, help='Bandwidth matrix CSV')
    p.add_argument('--power', help='Optional power matrix CSV for energy calculation')
    p.add_argument('--out', required=True, help='Output CSV (appended if exists)')
    args = p.parse_args()

    comp = read_csv_matrix(args.exec_file)
    bw = read_csv_matrix(args.bw)
    dag_mat = read_csv_matrix(args.dag)
    dag = build_dag(dag_mat)
    power = read_csv_matrix(args.power) if args.power else None

    if args.algo == 'heft':
        _, _, metrics = schedule_heft_paper(dag, comp, bw, power)
    else:
        _, _, metrics = schedule_peft_paper(dag, comp, bw, power)

    out_path = Path(args.out)
    write_header = not out_path.exists()
    with out_path.open('a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['algorithm','makespan','load_balance_ratio','energy'])
        w.writerow([args.algo, metrics['makespan'], metrics['load_balance_ratio'], metrics['energy']])
    print(f"Wrote metrics for {args.algo} to {args.out}")

if __name__ == '__main__':
    main()
