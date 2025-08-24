from __future__ import annotations
"""Paper-only PEFT implementation (Arabnejad & Barbosa 2014).

Retains functions expected by scenario scripts: schedule_dag, readCsvToNumpyMatrix, readCsvToDict, readDagMatrix,
and metrics helpers: _compute_makespan_and_idle, _compute_load_balance, _compute_communication_cost, _compute_waiting_time.
Implements:
 1. Optimistic Cost Table (OCT) recursion.
 2. Task rank = average of OCT row.
 3. Scheduling order = descending rank with precedence safety.
 4. Processor selection = argmin(EFT + OCT[task,proc]).
Energy data (power_dict) unused in decisions; only for external reporting.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, networkx as nx, logging, matplotlib.pyplot as plt

logger = logging.getLogger("peft")

@dataclass
class ScheduleEvent:
    task:int; start:float; end:float; proc:int

def readCsvToNumpyMatrix(csv_file: str) -> np.ndarray:
    with open(csv_file) as fd:
        rows=[r.strip().split(',') for r in fd.read().strip().splitlines() if r.strip()]
    arr = np.array(rows)[1:,1:]
    return arr.astype(float)

def readCsvToDict(csv_file: str):
    m = readCsvToNumpyMatrix(csv_file)
    return {i: row for i,row in enumerate(m)}

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

def _normalize_edges(dag: nx.DiGraph, comm: np.ndarray):
    mask = np.ones_like(comm, dtype=bool); np.fill_diagonal(mask, False)
    vals = comm[mask]; vals = vals[vals>0]
    avg_bw = float(np.mean(vals)) if len(vals) else 1.0
    for u,v in dag.edges():
        w = float(dag[u][v]['weight'])
        dag[u][v]['norm_comm'] = w / avg_bw if avg_bw>0 else 0.0

def _build_oct(dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray):
    _normalize_edges(dag, comm)
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError("DAG has no sink")
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks: dag.add_edge(s, virtual, weight=0.0, norm_comm=0.0)
        sink=virtual
    else: sink=sinks[0]
    P=comp.shape[1]
    oct_table={sink:[0.0]*P}
    ranks={sink:0.0}
    from collections import deque
    q=deque(dag.predecessors(sink))
    def ready(n): return all(s in oct_table for s in dag.successors(n))
    while q:
        n=q.pop()
        if not ready(n): q.appendleft(n); continue
        oct_table[n]=[0.0]*P
        for p in range(P):
            max_succ=-1e30
            for s in dag.successors(n):
                min_cost=1e30
                for sp in range(P):
                    succ_oct=oct_table[s][sp]
                    succ_comp = comp[s,sp] if s < comp.shape[0] else 0.0
                    comm_cost = dag[n][s]['norm_comm'] if p!=sp else 0.0
                    cost = succ_oct + succ_comp + comm_cost
                    if cost < min_cost: min_cost=cost
                if min_cost > max_succ: max_succ=min_cost
            if max_succ < 0: max_succ = 0.0
            oct_table[n][p]=max_succ
        ranks[n]=float(np.mean(oct_table[n]))
        for pred in dag.predecessors(n):
            if pred not in oct_table and pred not in q: q.append(pred)
    return oct_table, ranks

def _eft(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    ready=0.0
    for pred in dag.predecessors(task):
        sj=task_sched[pred]
        if sj.proc==proc: arr=sj.end
        else:
            bw = comm[sj.proc, proc]
            comm_t = (float(dag[pred][task]['weight']) / bw) if bw>0 else 0.0
            arr = sj.end + comm_t
        if arr>ready: ready=arr
    dur=float(comp[task,proc])
    jobs=proc_sched[proc]
    best=ScheduleEvent(task, ready, ready+dur, proc)
    for i,j in enumerate(jobs):
        if i==0 and j.start - dur >= ready:
            cand=ScheduleEvent(task, ready, ready+dur, proc)
            if cand.end < best.end: best=cand
        if i < len(jobs)-1:
            nxt=jobs[i+1]
            gap_start=max(ready,j.end); gap_end=nxt.start
            if gap_end-gap_start >= dur:
                cand=ScheduleEvent(task, gap_start, gap_start+dur, proc)
                if cand.end < best.end: best=cand
        if i==len(jobs)-1:
            start=max(ready,j.end); cand=ScheduleEvent(task,start,start+dur,proc)
            if cand.end < best.end: best=cand
    return best

def schedule_dag(dag, computation_matrix, communication_matrix, proc_schedules=None, **kwargs):
    if proc_schedules is None: proc_schedules={p:[] for p in range(communication_matrix.shape[0])}
    oct_table, ranks = _build_oct(dag, computation_matrix, communication_matrix)
    tasks=[n for n in dag.nodes() if n < computation_matrix.shape[0]]
    order=sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched:Dict[int,ScheduleEvent]={}
    remaining=list(order)
    while remaining:
        progressed=False
        for t in list(remaining):
            if any(pred not in task_sched for pred in dag.predecessors(t)): continue
            best=None; best_score=None
            for p in range(communication_matrix.shape[0]):
                cand=_eft(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules)
                score = cand.end + oct_table[t][p]
                if best is None or score < best_score:
                    best=cand; best_score=score
            task_sched[t]=best  # type: ignore
            proc_schedules[best.proc].append(best)  # type: ignore
            proc_schedules[best.proc].sort(key=lambda j:j.start)
            remaining.remove(t); progressed=True
        if not progressed:
            raise RuntimeError("Deadlock in PEFT scheduling (cyclic DAG?)")
    return proc_schedules, task_sched, {}

def _compute_makespan_and_idle(proc_schedules):
    makespan = max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        js=sorted(jobs,key=lambda j:j.start); idle=0.0
        if not js: idle=makespan
        else:
            idle+=js[0].start
            for i in range(len(js)-1): idle += max(0.0, js[i+1].start - js[i].end)
            idle+=max(0.0, makespan - js[-1].end)
        per_proc[p]=idle; total_idle+=idle
    return makespan,total_idle,per_proc

def _compute_load_balance(proc_schedules):
    busy={p:sum(ev.end-ev.start for ev in jobs) for p,jobs in proc_schedules.items()}; vals=list(busy.values()); n=len(vals)
    if n==0: return busy,0.0,1.0,1.0
    mean=sum(vals)/n; var=sum((v-mean)**2 for v in vals)/n; std=var**0.5; cv=std/mean if mean>0 else 0.0
    maxb=max(vals); minb=min(vals); imb=(maxb/minb) if minb>0 else float('inf') if maxb>0 else 1.0
    denom=n*sum(v*v for v in vals); fairness=(sum(vals)**2/denom) if denom>0 else 1.0
    return busy,cv,imb,fairness

def _compute_communication_cost(dag, proc_schedules, communication_matrix):
    task_map={}
    for p,jobs in proc_schedules.items():
        for ev in jobs: task_map[ev.task]=ev
    total=0.0
    for u,v in dag.edges():
        if u not in task_map or v not in task_map: continue
        pu=task_map[u].proc; pv=task_map[v].proc
        if pu==pv: continue
        bw=communication_matrix[pu,pv]
        if bw<=0: continue
        data=float(dag.get_edge_data(u,v)['weight'])
        total += data / bw
    return total

def _compute_waiting_time(proc_schedules):
    total=0.0; count=0
    for jobs in proc_schedules.values():
        for ev in jobs: total+=ev.start; count+=1
    return total/count if count else 0.0

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description="Paper-only PEFT")
    p.add_argument('--dag_file', required=True)
    p.add_argument('--exec_file', required=True)
    p.add_argument('--bw_file', required=True)
    a=p.parse_args()
    comp=readCsvToNumpyMatrix(a.exec_file); bw=readCsvToNumpyMatrix(a.bw_file); dag=readDagMatrix(a.dag_file)
    proc_sched, task_sched, _ = schedule_dag(dag, computation_matrix=comp, communication_matrix=bw)
    mk,_,_= _compute_makespan_and_idle(proc_sched)
    busy,_,_,_= _compute_load_balance(proc_sched)
    avg=sum(busy.values())/len(busy) if busy else 0.0
    print({'makespan':mk,'load_balance_ratio': (mk/avg if avg>0 else float('inf'))})
