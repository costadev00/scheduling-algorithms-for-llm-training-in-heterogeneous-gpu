"""Paper-only HEFT implementation (Topcuoglu et al., 2002).

Provided API kept compatible with existing scenario scripts:
 schedule_dag(dag, computation_matrix, communication_matrix, communication_startup=..., power_dict=...)
 helper functions: readCsvToNumpyMatrix, readCsvToDict, readDagMatrix,
 metrics helpers: _compute_makespan_and_idle, _compute_load_balance, _compute_communication_cost, _compute_waiting_time.

Only canonical HEFT is implemented: upward rank using average exec times and average communication (normalized by average bandwidth),
then processor selection by earliest finish time (EFT) with insertion based policy.
Energy is never used for scheduling; if power_dict provided, energy can be computed externally.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import logging, numpy as np, networkx as nx, matplotlib.pyplot as plt

logger = logging.getLogger("heft")

@dataclass
class ScheduleEvent:
    task: int
    start: float
    end: float
    proc: int

def readCsvToNumpyMatrix(csv_file: str) -> np.ndarray:
    with open(csv_file) as fd:
        rows = [r.strip().split(',') for r in fd.read().strip().splitlines() if r.strip()]
    arr = np.array(rows)[1:,1:]
    return arr.astype(float)

def readCsvToDict(csv_file: str):
    m = readCsvToNumpyMatrix(csv_file)
    return {i: row for i, row in enumerate(m)}

def readDagMatrix(dag_file: str, show_dag: bool=False):
    m = readCsvToNumpyMatrix(dag_file)
    dag = nx.DiGraph(m)
    dag.remove_edges_from([e for e in dag.edges() if dag.get_edge_data(*e)['weight'] == '0.0'])
    if show_dag:
        try:
            pos = nx.nx_pydot.graphviz_layout(dag, prog='dot')
        except Exception:
            pos = nx.spring_layout(dag, seed=42)
        nx.draw(dag, pos=pos, with_labels=True)
        plt.show()
    return dag

def _avg_exec(comp: np.ndarray, t: int) -> float:
    return float(np.mean(comp[t]))

def _avg_bandwidth(comm: np.ndarray) -> float:
    mask = np.ones_like(comm, dtype=bool)
    np.fill_diagonal(mask, False)
    vals = comm[mask]
    vals = vals[vals > 0]
    return float(np.mean(vals)) if len(vals) else 1.0

def _compute_upward_ranks(dag: nx.DiGraph, comp: np.ndarray, comm: np.ndarray) -> Dict[int,float]:
    avg_bw = _avg_bandwidth(comm)
    for u,v in dag.edges():
        w = float(dag[u][v]['weight'])
        dag[u][v]['norm_comm'] = w / avg_bw if avg_bw>0 else 0.0
    sinks = [n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks:
        raise ValueError("DAG has no sink")
    if len(sinks)>1:
        virtual = max(dag.nodes())+1
        for s in sinks:
            dag.add_edge(s, virtual, weight=0.0, norm_comm=0.0)
        sink = virtual
    else:
        sink = sinks[0]
    rank = {}
    pending = [sink]
    while pending:
        n = pending.pop()
        succs = list(dag.successors(n))
        if any(s not in rank for s in succs):
            pending.insert(0, n); continue
        if n == sink:
            r = _avg_exec(comp,n) if n < comp.shape[0] else 0.0
        else:
            r = _avg_exec(comp,n) + max((dag[n][s]['norm_comm'] + rank[s]) for s in succs)
        rank[n]=r
        for p in dag.predecessors(n):
            if p not in rank: pending.append(p)
    return rank

def _insertion_eft(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    ready = 0.0
    for pred in dag.predecessors(task):
        sj = task_sched[pred]
        if sj.proc == proc:
            arrival = sj.end
        else:
            bw = comm[sj.proc, proc]
            comm_t = (float(dag[pred][task]['weight']) / bw) if bw>0 else 0.0
            arrival = sj.end + comm_t
        if arrival > ready: ready = arrival
    dur = float(comp[task, proc])
    jobs = proc_sched[proc]
    best = ScheduleEvent(task, ready, ready+dur, proc)
    for i,j in enumerate(jobs):
        if i==0 and j.start - dur >= ready:
            cand = ScheduleEvent(task, ready, ready+dur, proc)
            if cand.end < best.end: best = cand
        if i < len(jobs)-1:
            nxt = jobs[i+1]
            gap_start = max(ready, j.end)
            gap_end = nxt.start
            if gap_end - gap_start >= dur:
                cand = ScheduleEvent(task, gap_start, gap_start+dur, proc)
                if cand.end < best.end: best = cand
        if i == len(jobs)-1:
            start = max(ready, j.end)
            cand = ScheduleEvent(task, start, start+dur, proc)
            if cand.end < best.end: best = cand
    return best

def schedule_dag(dag, computation_matrix, communication_matrix, communication_startup=None, proc_schedules=None, **kwargs):
    if proc_schedules is None: proc_schedules = {p:[] for p in range(communication_matrix.shape[0])}
    ranks = _compute_upward_ranks(dag, computation_matrix, communication_matrix)
    tasks = [n for n in dag.nodes() if n < computation_matrix.shape[0]]
    order = sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched: Dict[int,ScheduleEvent] = {}
    for p in range(communication_matrix.shape[0]):
        if p not in proc_schedules: proc_schedules[p] = []
    for t in order:
        for pred in dag.predecessors(t):
            if pred not in task_sched and pred < computation_matrix.shape[0]:
                raise RuntimeError("Predecessor not scheduled before child in HEFT (DAG invalid)")
        best=None
        for p in range(communication_matrix.shape[0]):
            cand = _insertion_eft(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules)
            if best is None or cand.end < best.end: best = cand
        task_sched[t]=best  # type: ignore
        proc_schedules[best.proc].append(best)  # type: ignore
        proc_schedules[best.proc].sort(key=lambda j:j.start)
    return proc_schedules, task_sched, {}

def _compute_makespan_and_idle(proc_schedules):
    makespan = max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        jobs_sorted=sorted(jobs, key=lambda j:j.start)
        idle=0.0
        if not jobs_sorted:
            idle = makespan
        else:
            idle += jobs_sorted[0].start
            for i in range(len(jobs_sorted)-1):
                idle += max(0.0, jobs_sorted[i+1].start - jobs_sorted[i].end)
            idle += max(0.0, makespan - jobs_sorted[-1].end)
        per_proc[p]=idle; total_idle+=idle
    return makespan, total_idle, per_proc

def _compute_load_balance(proc_schedules):
    busy={p:sum(ev.end-ev.start for ev in jobs) for p,jobs in proc_schedules.items()}
    vals=list(busy.values()); n=len(vals)
    if n==0: return busy,0.0,1.0,1.0
    mean=sum(vals)/n
    var=sum((v-mean)**2 for v in vals)/n
    std=var**0.5
    cv=std/mean if mean>0 else 0.0
    maxb=max(vals); minb=min(vals)
    imb = (maxb/minb) if minb>0 else float('inf') if maxb>0 else 1.0
    denom = n*sum(v*v for v in vals)
    fairness = (sum(vals)**2/denom) if denom>0 else 1.0
    return busy, cv, imb, fairness

def _compute_communication_cost(dag, proc_schedules, communication_matrix, communication_startup=None):
    task_map={}
    for p,jobs in proc_schedules.items():
        for ev in jobs: task_map[ev.task]=ev
    total=0.0
    for u,v in dag.edges():
        if u not in task_map or v not in task_map: continue
        pu=task_map[u].proc; pv=task_map[v].proc
        if pu==pv: continue
        bw = communication_matrix[pu,pv]
        if bw<=0: continue
        data = float(dag.get_edge_data(u,v)['weight'])
        startup = communication_startup[pu] if communication_startup is not None else 0.0
        total += data / bw + startup
    return total

def _compute_waiting_time(proc_schedules):
    total=0.0; count=0
    for jobs in proc_schedules.values():
        for ev in jobs:
            total += ev.start; count+=1
    return total/count if count else 0.0

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Paper-only HEFT (Topcuoglu 2002)")
    parser.add_argument('--dag_file', required=True)
    parser.add_argument('--exec_file', required=True)
    parser.add_argument('--bw_file', required=True)
    parser.add_argument('--power_file')
    args = parser.parse_args()
    comp = readCsvToNumpyMatrix(args.exec_file)
    bw = readCsvToNumpyMatrix(args.bw_file)
    dag = readDagMatrix(args.dag_file, show_dag=False)
    proc_sched, task_sched, _ = schedule_dag(dag, computation_matrix=comp, communication_matrix=bw)
    makespan,_,_= _compute_makespan_and_idle(proc_sched)
    busy,_,_,_= _compute_load_balance(proc_sched)
    avg_busy = sum(busy.values())/len(busy) if busy else 0.0
    lb_ratio = makespan/avg_busy if avg_busy>0 else float('inf')
    print({'makespan':makespan,'load_balance_ratio':lb_ratio})
