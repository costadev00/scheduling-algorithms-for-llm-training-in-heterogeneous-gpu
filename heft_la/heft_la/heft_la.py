from __future__ import annotations
"""HEFT-LA (Lookahead HEFT) paper-faithful style implementation.

Concept (based on lookahead heuristic variant of HEFT described in attachment):
 - Compute standard HEFT upward ranks (avg exec + max successor(comm + rank)).
 - Instead of selecting only earliest finish time (EFT) for current task's mapping, enrich processor choice by a one-step lookahead:
    For each candidate processor p for current task t, we tentatively place t on p (using insertion-based earliest finish). Then for each unscheduled child c of t we estimate its best earliest finish among all processors given this tentative placement (using same EFT logic, ignoring interactions among multiple children). The lookahead score = EFT(t on p) + lambda * min_child_EFT_sum or alternative combination. Paper specifies summing or averaging predicted child completion times; we implement sum of earliest child EFTs (excluding already scheduled descendants). Tie-break: choose mapping with minimal lookahead score; fallback to pure EFT.
 - Lambda weighting set to 1.0 (can be parameterized) to keep paper baseline.

API kept consistent with other algorithms: schedule_dag(dag, computation_matrix, communication_matrix, ...)
Energy/power not used in decisions.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, networkx as nx

@dataclass
class ScheduleEvent:
    task:int; start:float; end:float; proc:int

def readCsvToNumpyMatrix(csv_file: str) -> np.ndarray:
    with open(csv_file) as fd:
        rows=[r.strip().split(',') for r in fd.read().strip().splitlines() if r.strip()]
    arr=np.array(rows)[1:,1:]
    return arr.astype(float)

def readCsvToDict(csv_file: str):
    m=readCsvToNumpyMatrix(csv_file)
    return {i: row for i,row in enumerate(m)}

def readDagMatrix(dag_file: str, show_dag: bool=False):
    import matplotlib.pyplot as plt
    m=readCsvToNumpyMatrix(dag_file)
    dag=nx.DiGraph(m)
    dag.remove_edges_from([e for e in dag.edges() if dag.get_edge_data(*e)['weight']=='0.0'])
    if show_dag:
        try:
            pos=nx.nx_pydot.graphviz_layout(dag, prog='dot')
        except Exception:
            pos=nx.spring_layout(dag, seed=42)
        nx.draw(dag, pos=pos, with_labels=True)
        plt.show()
    return dag

def _avg_exec(comp:np.ndarray, t:int)->float: return float(np.mean(comp[t]))

def _avg_bandwidth(comm:np.ndarray)->float:
    mask=np.ones_like(comm, dtype=bool); np.fill_diagonal(mask, False)
    vals=comm[mask]; vals=vals[vals>0]
    return float(np.mean(vals)) if len(vals) else 1.0

def _compute_upward_ranks(dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray)->Dict[int,float]:
    avg_bw=_avg_bandwidth(comm)
    for u,v in dag.edges():
        w=float(dag[u][v]['weight']); dag[u][v]['norm_comm']= w/avg_bw if avg_bw>0 else 0.0
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError('DAG has no sink')
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks: dag.add_edge(s, virtual, weight=0.0, norm_comm=0.0)
        sink=virtual
    else: sink=sinks[0]
    rank={}; stack=[sink]
    while stack:
        n=stack.pop()
        succs=list(dag.successors(n))
        if any(s not in rank for s in succs):
            stack.insert(0,n); continue
        if n==sink and n>=comp.shape[0]: val=0.0
        else:
            base=_avg_exec(comp,n) if n<comp.shape[0] else 0.0
            if succs:
                val=base+max(dag[n][s]['norm_comm']+rank[s] for s in succs)
            else: val=base
        rank[n]=val
        for p in dag.predecessors(n):
            if p not in rank: stack.append(p)
    return rank

def _insertion_eft(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    ready=0.0
    for pred in dag.predecessors(task):
        sj=task_sched[pred]
        if sj.proc==proc: arr=sj.end
        else:
            bw=comm[sj.proc, proc]; data=float(dag[pred][task]['weight'])
            comm_t=data / bw if bw>0 else 0.0
            arr=sj.end+comm_t
        if arr>ready: ready=arr
    dur=float(comp[task,proc])
    jobs=proc_sched[proc]
    best=ScheduleEvent(task, ready, ready+dur, proc)
    for i,j in enumerate(jobs):
        if i==0 and j.start - ready >= dur:
            cand=ScheduleEvent(task, ready, ready+dur, proc)
            if cand.end < best.end: best=cand
        if i < len(jobs)-1:
            nxt=jobs[i+1]
            gap_start=max(ready, j.end); gap_end=nxt.start
            if gap_end - gap_start >= dur:
                cand=ScheduleEvent(task, gap_start, gap_start+dur, proc)
                if cand.end < best.end: best=cand
        if i==len(jobs)-1:
            start=max(ready, j.end)
            cand=ScheduleEvent(task, start, start+dur, proc)
            if cand.end < best.end: best=cand
    return best

def _predict_child_efts(task:int, placement:ScheduleEvent, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    # Temporarily consider task scheduled as placement already
    predictions=0.0; count=0
    for child in dag.successors(task):
        if child in task_sched: continue
        # compute earliest data ready time at each proc given placement
        best_child_end=None
        for p in range(comm.shape[0]):
            # compute EST for child if scheduled on p after placement
            ready=0.0
            # predecessors of child may include our task plus others
            for pred in dag.predecessors(child):
                if pred==task:
                    if placement.proc==p: arr=placement.end
                    else:
                        bw=comm[placement.proc, p]; data=float(dag[pred][child]['weight'])
                        comm_t=data / bw if bw>0 else 0.0
                        arr=placement.end+comm_t
                else:
                    if pred not in task_sched:
                        # cannot predict reliably yet -> skip this child
                        arr=None; break
                    sj=task_sched[pred]
                    if sj.proc==p: arr=sj.end
                    else:
                        bw=comm[sj.proc, p]; data=float(dag[pred][child]['weight'])
                        comm_t=data / bw if bw>0 else 0.0
                        arr=sj.end+comm_t
                if arr is None: break
                if arr>ready: ready=arr
            else:
                dur=float(comp[child,p]) if child < comp.shape[0] else 0.0
                # simple non-insertion earliest finish (approx) on p
                # consider gaps quickly -> just end at max(ready, last_end)
                last_end = proc_sched[p][-1].end if proc_sched[p] else 0.0
                start=max(ready, last_end)
                end=start+dur
                if best_child_end is None or end < best_child_end:
                    best_child_end=end
        if best_child_end is not None:
            predictions += best_child_end; count+=1
    return predictions, count

def schedule_dag(dag, computation_matrix, communication_matrix, communication_startup=None, proc_schedules=None, lookahead_weight:float=1.0, **kwargs):
    if proc_schedules is None: proc_schedules={p:[] for p in range(communication_matrix.shape[0])}
    ranks=_compute_upward_ranks(dag, computation_matrix, communication_matrix)
    tasks=[n for n in dag.nodes() if n < computation_matrix.shape[0]]
    order=sorted(tasks, key=lambda n: ranks[n], reverse=True)
    task_sched:Dict[int,ScheduleEvent]={}
    for p in proc_schedules:
        proc_schedules[p].sort(key=lambda j:j.start)
    for t in order:
        # ensure predecessors scheduled
        for pred in dag.predecessors(t):
            if pred not in task_sched and pred < computation_matrix.shape[0]:
                raise RuntimeError('Predecessor not scheduled before child in HEFT-LA (DAG invalid)')
        best=None; best_score=None; best_eft=None
        for p in range(communication_matrix.shape[0]):
            placement=_insertion_eft(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules)
            child_sum, child_cnt=_predict_child_efts(t, placement, dag, computation_matrix, communication_matrix, task_sched, proc_schedules)
            lookahead = child_sum if child_cnt>0 else 0.0
            score = placement.end + lookahead_weight*lookahead
            if best is None or score < best_score or (score==best_score and placement.end < best_eft):
                best=placement; best_score=score; best_eft=placement.end
        task_sched[t]=best  # type: ignore
        proc_schedules[best.proc].append(best)  # type: ignore
        proc_schedules[best.proc].sort(key=lambda j:j.start)
    return proc_schedules, task_sched, {'ranks':ranks}

def _compute_makespan_and_idle(proc_schedules):
    makespan = max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        js=sorted(jobs,key=lambda j:j.start); idle=0.0
        if not js: idle=makespan
        else:
            idle+=js[0].start
            for i in range(len(js)-1): idle+= max(0.0, js[i+1].start - js[i].end)
            idle+= max(0.0, makespan - js[-1].end)
        per_proc[p]=idle; total_idle+=idle
    return makespan,total_idle,per_proc

def _compute_load_balance(proc_schedules):
    busy={p:sum(ev.end-ev.start for ev in jobs) for p,jobs in proc_schedules.items()}; vals=list(busy.values()); n=len(vals)
    if n==0: return busy,0.0,1.0,1.0
    mean=sum(vals)/n; var=sum((v-mean)**2 for v in vals)/n; std=var**0.5; cv=std/mean if mean>0 else 0.0
    maxb=max(vals); minb=min(vals); imb=(maxb/minb) if minb>0 else float('inf') if maxb>0 else 1.0
    denom=n*sum(v*v for v in vals); fairness=(sum(vals)**2/denom) if denom>0 else 1.0
    return busy,cv,imb,fairness

def _compute_communication_cost(dag, proc_schedules, communication_matrix, communication_startup=None):
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
        startup=communication_startup[pu] if communication_startup is not None else 0.0
        total += data / bw + startup
    return total

def _compute_waiting_time(proc_schedules):
    total=0.0; count=0
    for jobs in proc_schedules.values():
        for ev in jobs:
            total+=ev.start; count+=1
    return total/count if count else 0.0

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='HEFT Lookahead (paper-style)')
    p.add_argument('--dag_file', required=True)
    p.add_argument('--exec_file', required=True)
    p.add_argument('--bw_file', required=True)
    p.add_argument('--lookahead_weight', type=float, default=1.0)
    a=p.parse_args()
    comp=readCsvToNumpyMatrix(a.exec_file); bw=readCsvToNumpyMatrix(a.bw_file); dag=readDagMatrix(a.dag_file)
    proc_sched, task_sched, meta = schedule_dag(dag, computation_matrix=comp, communication_matrix=bw, lookahead_weight=a.lookahead_weight)
    mk,_,_= _compute_makespan_and_idle(proc_sched)
    busy,_,_,_= _compute_load_balance(proc_sched)
    avg=sum(busy.values())/len(busy) if busy else 0.0
    print({'makespan':mk,'load_balance_ratio': (mk/avg if avg>0 else float('inf'))})
