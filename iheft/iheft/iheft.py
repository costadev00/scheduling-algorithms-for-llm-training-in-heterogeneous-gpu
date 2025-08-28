"""Implementation of Improved HEFT (IHEFT) algorithm as described in provided paper details.

Key differences vs canonical HEFT:
- Task priority rank uses Weight_ni instead of average exec time. Weight_ni = | (max_exec - min_exec) / (max_exec / min_exec) | capturing heterogeneity & speedup dispersion.
- Processor selection: determine proc_with_min_eft and proc_with_min_exec for current task.
  * If they are the same -> choose it.
  * Else compute Weight_abstract = | (EFT_best - EFT_execbest) / (EFT_best / EFT_execbest) |.
    Cross_Threshold = Weight_ni / Weight_abstract.
    Draw r ~ Uniform[0.1,0.3]. If Cross_Threshold <= r -> choose processor with min execution time (local optimal), else choose min EFT (global optimal).
- Uses same insertion-based earliest finish computation as HEFT.

Complexity remains O(t^2 * p).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random, numpy as np, networkx as nx, matplotlib.pyplot as plt

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

def _compute_weights(comp: np.ndarray) -> Dict[int,float]:
    weights={}
    for t in range(comp.shape[0]):
        row = comp[t]
        max_exec = float(np.max(row))
        min_exec = float(np.min(row))
        if min_exec <= 0 or max_exec <= 0:
            weights[t]=0.0
            continue
        # Weight_ni = | (max - min) / (max / min) |
        weights[t] = abs((max_exec - min_exec)/(max_exec / min_exec))
    return weights

def _compute_modified_upward_ranks(dag: nx.DiGraph, comp: np.ndarray, weights: Dict[int,float], comm: np.ndarray) -> Dict[int,float]:
    # communication weight = raw edge weight / average positive bandwidth (same normalization as HEFT variant) to keep scale moderate
    mask = np.ones_like(comm, dtype=bool); np.fill_diagonal(mask, False)
    bw_vals = comm[mask]; bw_vals = bw_vals[bw_vals>0]
    avg_bw = float(np.mean(bw_vals)) if len(bw_vals) else 1.0
    for u,v in dag.edges():
        try:
            w = float(dag[u][v]['weight'])
        except Exception:
            w = 0.0
        dag[u][v]['norm_comm'] = w / avg_bw if avg_bw>0 else 0.0
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError("DAG has no sink")
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks:
            dag.add_edge(s, virtual, weight=0.0, norm_comm=0.0)
        sink=virtual
    else:
        sink=sinks[0]
    rank={}
    pending=[sink]
    while pending:
        n=pending.pop()
        succs=list(dag.successors(n))
        if any(s not in rank for s in succs):
            pending.insert(0,n); continue
        if n==sink:
            r = weights.get(n,0.0)
        else:
            if succs:
                r = weights.get(n,0.0) + max((dag[n][s]['norm_comm'] + rank[s]) for s in succs)
            else:
                r = weights.get(n,0.0)
        rank[n]=r
        for p in dag.predecessors(n):
            if p not in rank: pending.append(p)
    return rank

def _insertion_eft(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    ready=0.0
    for pred in dag.predecessors(task):
        sj=task_sched[pred]
        if sj.proc==proc:
            arrival=sj.end
        else:
            bw=comm[sj.proc, proc]
            comm_t=(float(dag[pred][task]['weight'])/bw) if bw>0 else 0.0
            arrival=sj.end+comm_t
        if arrival>ready: ready=arrival
    dur=float(comp[task,proc])
    jobs=proc_sched[proc]
    best=ScheduleEvent(task, ready, ready+dur, proc)
    for i,j in enumerate(jobs):
        if i==0 and j.start - dur >= ready:
            cand=ScheduleEvent(task, ready, ready+dur, proc)
            if cand.end<best.end: best=cand
        if i < len(jobs)-1:
            nxt=jobs[i+1]
            gap_start=max(ready, j.end); gap_end=nxt.start
            if gap_end-gap_start >= dur:
                cand=ScheduleEvent(task, gap_start, gap_start+dur, proc)
                if cand.end<best.end: best=cand
        if i==len(jobs)-1:
            start=max(ready, j.end)
            cand=ScheduleEvent(task, start, start+dur, proc)
            if cand.end<best.end: best=cand
    return best

def schedule_dag(dag, computation_matrix, communication_matrix, communication_startup=None, proc_schedules=None, rng=None, **kwargs):
    if proc_schedules is None: proc_schedules={p:[] for p in range(communication_matrix.shape[0])}
    if rng is None: rng=random.Random(42)
    weights=_compute_weights(computation_matrix)
    ranks=_compute_modified_upward_ranks(dag, computation_matrix, weights, communication_matrix)
    tasks=[n for n in dag.nodes() if n < computation_matrix.shape[0]]
    order=sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched: Dict[int,ScheduleEvent]={}
    for p in range(communication_matrix.shape[0]):
        if p not in proc_schedules: proc_schedules[p]=[]
    for t in order:
        for pred in dag.predecessors(t):
            if pred not in task_sched and pred < computation_matrix.shape[0]:
                raise RuntimeError("Predecessor not scheduled before child (invalid DAG ordering)")
        # evaluate EFT on all processors + collect raw exec times
        best_eft=None; best_eft_event=None
        best_exec_proc=None; best_exec_time=None; best_exec_event=None
        for p in range(communication_matrix.shape[0]):
            cand=_insertion_eft(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules)
            eft=cand.end
            if best_eft is None or eft < best_eft or (eft==best_eft and p < best_eft_event.proc):
                best_eft=eft; best_eft_event=cand
            raw_exec=float(computation_matrix[t,p])
            if best_exec_time is None or raw_exec < best_exec_time or (raw_exec==best_exec_time and p < (best_exec_proc or p)):
                best_exec_time=raw_exec; best_exec_proc=p; best_exec_event=cand
        # decision mechanism
        if best_eft_event.proc == best_exec_proc:
            chosen=best_eft_event
        else:
            eft_best=best_eft_event.end
            eft_exec=best_exec_event.end
            if eft_best<=0 or eft_exec<=0 or eft_best==eft_exec:
                chosen=best_eft_event
            else:
                weight_ni=weights.get(t,0.0)
                weight_abs=abs((eft_best - eft_exec)/(eft_best/eft_exec)) if eft_best and eft_exec else float('inf')
                if weight_abs==0:
                    chosen=best_eft_event
                else:
                    cross_threshold = (weight_ni / weight_abs) if weight_abs>0 else float('inf')
                    r = rng.uniform(0.1,0.3)
                    if cross_threshold <= r:
                        chosen=best_exec_event
                    else:
                        chosen=best_eft_event
        task_sched[t]=chosen
        proc_schedules[chosen.proc].append(chosen)
        proc_schedules[chosen.proc].sort(key=lambda j:j.start)
    return proc_schedules, task_sched, {"ranks":ranks, "weights":weights}

def _compute_makespan_and_idle(proc_schedules):
    makespan=max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        jobs_sorted=sorted(jobs, key=lambda j:j.start)
        idle=0.0
        if not jobs_sorted:
            idle=makespan
        else:
            idle+=jobs_sorted[0].start
            for i in range(len(jobs_sorted)-1):
                idle+=max(0.0, jobs_sorted[i+1].start - jobs_sorted[i].end)
            idle+=max(0.0, makespan - jobs_sorted[-1].end)
        per_proc[p]=idle; total_idle+=idle
    return makespan, total_idle, per_proc

def _compute_load_balance(proc_schedules):
    busy={p:sum(ev.end-ev.start for ev in jobs) for p,jobs in proc_schedules.items()}
    total_busy=sum(busy.values()); makespan=max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    return busy, total_busy, makespan, {p:busy[p]/makespan if makespan>0 else 0.0 for p in busy}

def _compute_communication_cost(dag, proc_schedules, comm_matrix, _power=None):
    # sum edge comm time where endpoints on different processors
    task_to_proc={ev.task:ev.proc for jobs in proc_schedules.values() for ev in jobs}
    total=0.0
    for u,v in dag.edges():
        pu=task_to_proc.get(u); pv=task_to_proc.get(v)
        if pu is None or pv is None: continue
        if pu==pv: continue
        bw=comm_matrix[pu,pv]; w=float(dag[u][v]['weight'])
        total += (w/bw) if bw>0 else 0.0
    return total

def _compute_waiting_time(proc_schedules):
    starts=[ev.start for jobs in proc_schedules.values() for ev in jobs]
    return float(np.mean(starts)) if starts else 0.0

__all__=[
    'schedule_dag','readCsvToNumpyMatrix','readDagMatrix','readCsvToDict',
    '_compute_makespan_and_idle','_compute_load_balance','_compute_communication_cost','_compute_waiting_time'
]
