"""Paper-only HEFT implementation (Topcuoglu et al., 2002).

Minimal HEFT faithful to original algorithm:
 1. Upward rank: rank_u(t) = avg_exec_time(t) + max_{s in succ(t)} ( avg_comm_time(t,s) + rank_u(s) )
    with rank_u(exit) = avg_exec_time(exit).
 2. Processor selection: earliest finish time (EFT) with insertion policy.
 3. No alternative rank metrics, no startup latency vector, no energy-aware scheduling heuristics.
 4. Optional energy reporting (post-schedule) if a power matrix is supplied; does not affect scheduling.

Matrix conventions:
  - computation_matrix: shape (V,P) execution times.
  - communication_matrix: shape (P,P) bandwidth (0 on diagonal permitted; local comm cost=0).
  - dag: networkx.DiGraph with edge attribute 'weight' = data volume.

Returned metrics: makespan, load_balance_ratio, optional energy.
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

def _average_exec_time(computation_matrix: np.ndarray, task: int) -> float:
    return float(np.mean(computation_matrix[task]))

def _average_bandwidth(communication_matrix: np.ndarray) -> float:
    mask = np.ones_like(communication_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    vals = communication_matrix[mask]
    vals = vals[vals > 0]
    return float(np.mean(vals)) if len(vals) else 1.0

def _compute_upward_ranks(dag: nx.DiGraph, comp: np.ndarray, comm: np.ndarray) -> Dict[int, float]:
    avg_bw = _average_bandwidth(comm)
    for u, v in dag.edges():
        w = float(dag[u][v]['weight'])
        dag[u][v]['avgcomm'] = w / avg_bw if avg_bw > 0 else 0.0
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
    rank_u: Dict[int, float] = {}
    q = deque([sink])
    while q:
        n = q.popleft()
        succs = list(dag.successors(n))
        if any(s not in rank_u for s in succs):
            q.append(n)
            continue
        if n == sink:
            rank = _average_exec_time(comp, n) if n < comp.shape[0] else 0.0
        else:
            max_s = max((dag[n][s]['avgcomm'] + rank_u[s]) for s in succs) if succs else 0.0
            rank = _average_exec_time(comp, n) + max_s
        rank_u[n] = rank
        for p in dag.predecessors(n):
            if p not in rank_u and p not in q:
                q.append(p)
    return rank_u

def _earliest_finish(task: int, proc: int, dag: nx.DiGraph, comp: np.ndarray, comm: np.ndarray, task_sched: Dict[int, TaskSchedule], proc_sched: Dict[int, List[TaskSchedule]]) -> TaskSchedule:
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

def schedule_heft_paper(dag: nx.DiGraph, computation_matrix: np.ndarray, communication_matrix: np.ndarray, power_matrix: Optional[np.ndarray] = None):
    ranks = _compute_upward_ranks(dag, computation_matrix, communication_matrix)
    tasks = [n for n in dag.nodes() if n < computation_matrix.shape[0]]
    order = sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched: Dict[int, TaskSchedule] = {}
    proc_sched: Dict[int, List[TaskSchedule]] = {p: [] for p in range(communication_matrix.shape[0])}
    for t in order:
        for pred in dag.predecessors(t):
            if pred not in task_sched and pred < computation_matrix.shape[0]:
                raise RuntimeError(f"Predecessor {pred} of {t} not scheduled (non single-entry/sink DAG)")
        best = None
        for p in proc_sched.keys():
            cand = _earliest_finish(t, p, dag, computation_matrix, communication_matrix, task_sched, proc_sched)
            if best is None or cand.end < best.end:
                best = cand
        task_sched[t] = best  # type: ignore
        proc_sched[best.proc].append(best)  # type: ignore
        proc_sched[best.proc].sort(key=lambda j: j.start)
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
    p = argparse.ArgumentParser(description="Paper-only HEFT")
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
    _, _, metrics = schedule_heft_paper(g, comp, bw, power)
    print(metrics)
