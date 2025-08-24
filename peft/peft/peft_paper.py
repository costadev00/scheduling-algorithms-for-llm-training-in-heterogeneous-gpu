"""Paper-only PEFT implementation (Arabnejad & Barbosa, 2014).

Minimal PEFT faithful to original algorithm:
 1. Build Optimistic Cost Table (OCT) recursively.
 2. Task priority = average OCT row (rank).
 3. Schedule in descending rank order, respecting precedence (only schedule when predecessors done).
 4. Processor selection: minimize EFT + OCT(task, proc).
 5. No energy-aware heuristics; optional post-schedule energy reporting if power matrix supplied.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
import numpy as np
import networkx as nx

@dataclass
class TaskSchedule:
    task: int
    start: float
    end: float
    proc: int

def _normalize_edges(dag: nx.DiGraph, comm: np.ndarray):
    mask = np.ones_like(comm, dtype=bool)
    np.fill_diagonal(mask, False)
    bw_vals = comm[mask]
    bw_vals = bw_vals[bw_vals > 0]
    avg_bw = float(np.mean(bw_vals)) if len(bw_vals) else 1.0
    for u, v in dag.edges():
        w = float(dag[u][v]['weight'])
        dag[u][v]['avgcomm'] = w / avg_bw if avg_bw > 0 else 0.0

def _compute_oct(dag: nx.DiGraph, comp: np.ndarray, comm: np.ndarray):
    _normalize_edges(dag, comm)
    # Single sink assumption; add virtual sink if multiple.
    sinks = [n for n in dag.nodes() if dag.out_degree(n) == 0]
    if not sinks:
        raise ValueError("DAG has no sink")
    if len(sinks) > 1:
        virtual = max(dag.nodes()) + 1
        for s in sinks:
            dag.add_edge(s, virtual, weight=0.0, avgcomm=0.0)
        sink = virtual
    else:
        sink = sinks[0]
    P = comp.shape[1]
    oct_table: Dict[int, List[float]] = {sink: [0.0]*P}
    ranks: Dict[int, float] = {sink: 0.0}
    q = deque(dag.predecessors(sink))
    def can_process(n):
        return all(s in oct_table for s in dag.successors(n))
    while q:
        n = q.pop()
        if not can_process(n):
            q.appendleft(n)
            continue
        oct_table[n] = [0.0]*P
        for p in range(P):
            max_succ = -np.inf
            for s in dag.successors(n):
                min_cost = np.inf
                for sp in range(P):
                    succ_oct = oct_table[s][sp]
                    succ_comp = comp[s, sp] if s < comp.shape[0] else 0.0
                    comm_cost = dag[n][s]['avgcomm'] if p != sp else 0.0
                    cost = succ_oct + succ_comp + comm_cost
                    if cost < min_cost:
                        min_cost = cost
                if min_cost > max_succ:
                    max_succ = min_cost
            if max_succ == -np.inf:
                max_succ = 0.0
            oct_table[n][p] = max_succ
        ranks[n] = float(np.mean(oct_table[n]))
        for pred in dag.predecessors(n):
            if pred not in oct_table and pred not in q:
                q.append(pred)
    return oct_table, ranks

def _eft(task: int, proc: int, dag: nx.DiGraph, comp: np.ndarray, comm: np.ndarray, task_sched: Dict[int, TaskSchedule], proc_sched: Dict[int, List[TaskSchedule]]) -> TaskSchedule:
    ready = 0.0
    for pred in dag.predecessors(task):
        ps = task_sched[pred]
        if ps.proc == proc:
            arr = ps.end
        else:
            bw = comm[ps.proc, proc]
            comm_t = (float(dag[pred][task]['weight']) / bw) if bw > 0 else 0.0
            arr = ps.end + comm_t
        if arr > ready:
            ready = arr
    dur = float(comp[task, proc])
    jobs = proc_sched[proc]
    best = TaskSchedule(task, ready, ready + dur, proc)
    for i, job in enumerate(jobs):
        if i == 0 and job.start - dur >= ready:
            cand = TaskSchedule(task, ready, ready + dur, proc)
            if cand.end < best.end:
                best = cand
        if i < len(jobs) - 1:
            nxt = jobs[i+1]
            gap_start = max(ready, job.end)
            gap_end = nxt.start
            if gap_end - gap_start >= dur:
                cand = TaskSchedule(task, gap_start, gap_start + dur, proc)
                if cand.end < best.end:
                    best = cand
        if i == len(jobs) - 1:
            start = max(ready, job.end)
            cand = TaskSchedule(task, start, start + dur, proc)
            if cand.end < best.end:
                best = cand
    return best

def schedule_peft_paper(dag: nx.DiGraph, computation_matrix: np.ndarray, communication_matrix: np.ndarray, power_matrix: Optional[np.ndarray] = None):
    oct_table, ranks = _compute_oct(dag, computation_matrix, communication_matrix)
    tasks = [n for n in dag.nodes() if n < computation_matrix.shape[0]]
    # Sort by descending rank
    order = sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched: Dict[int, TaskSchedule] = {}
    proc_sched: Dict[int, List[TaskSchedule]] = {p: [] for p in range(communication_matrix.shape[0])}
    remaining = list(order)
    while remaining:
        progressed = False
        for t in list(remaining):
            if any(pred not in task_sched for pred in dag.predecessors(t)):
                continue
            best = None
            best_sum = None
            for p in proc_sched.keys():
                cand = _eft(t, p, dag, computation_matrix, communication_matrix, task_sched, proc_sched)
                score = cand.end + oct_table[t][p]
                if best is None or score < best_sum:  # type: ignore
                    best = cand
                    best_sum = score
            task_sched[t] = best  # type: ignore
            proc_sched[best.proc].append(best)  # type: ignore
            proc_sched[best.proc].sort(key=lambda j: j.start)
            remaining.remove(t)
            progressed = True
        if not progressed:
            raise RuntimeError("Cycle detected or unsatisfiable dependencies in DAG for PEFT paper implementation.")
    makespan = max((s.end for s in task_sched.values()), default=0.0)
    busy = {p: sum(j.end - j.start for j in jobs) for p, jobs in proc_sched.items()}
    avg_busy = (sum(busy.values()) / len(busy)) if busy else 0.0
    load_balance_ratio = makespan / avg_busy if avg_busy > 0 else float('inf')
    energy = None
    if power_matrix is not None:
        if power_matrix.shape != computation_matrix.shape:
            raise ValueError("power_matrix shape mismatch")
        e = 0.0
        for t, s in task_sched.items():
            e += (s.end - s.start) * float(power_matrix[t, s.proc])
        energy = e
    metrics = {"makespan": makespan, "load_balance_ratio": load_balance_ratio, "energy": energy}
    return proc_sched, task_sched, metrics

if __name__ == "__main__":
    import argparse
    def read_csv_matrix(path):
        with open(path) as f:
            lines = [l.strip().split(',') for l in f.read().strip().splitlines() if l.strip()]
        arr = np.array(lines)[1:,1:]
        return arr.astype(float)
    p = argparse.ArgumentParser(description="Paper-only PEFT")
    p.add_argument('--dag_file', required=True)
    p.add_argument('--exec_file', required=True)
    p.add_argument('--bw_file', required=True)
    p.add_argument('--power_file')
    a = p.parse_args()
    comp = read_csv_matrix(a.exec_file)
    bw = read_csv_matrix(a.bw_file)
    dag_mat = read_csv_matrix(a.dag_file)
    g = nx.DiGraph(dag_mat)
    g.remove_edges_from([e for e in g.edges() if g.get_edge_data(*e)['weight'] == '0.0'])
    power = read_csv_matrix(a.power_file) if a.power_file else None
    _, _, metrics = schedule_peft_paper(g, comp, bw, power)
    print(metrics)
