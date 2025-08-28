from __future__ import annotations
"""Dynamic Level Scheduling (DLS) – paper-faithful DL1 variant (Sih & Lee, IEEE TPDS 1993).

Key characteristics implemented here (differences vs earlier HEFT-like version):
    * CPU placement has NO gap insertion: task starts at max(DA(t,p), TF(p)), where TF(p) is
        the finish time of the last task already assigned to processor p.
    * Communication contention is modelled via per-(src,dst) LINK RESERVATIONS. Each data
        transfer reserves the earliest non-overlapping window of length data/bandwidth
        starting no earlier than the finish time of the source task. DA(t,p) is the max over
        predecessors of their (finish time + reserved communication window). Same-processor
        predecessors contribute their finish time (no comm).
    * Static levels SL* use the ADJUSTED MEDIAN execution time per task (if the statistical
        median would be inf, use the largest finite execution time) matching the paper's
        handling of infeasible processors.
    * DL1(t,p) = SL*(t) - EST(t,p) + (E*(t) - E(t,p)) with EST = max(DA, TF).

Not implemented (extensions left as future work): descendant correction term (DL2), and
the generalized dynamic level (GDL) scarcity adjustment.

Tie-breaking: higher DL1, then smaller EST, then lower task id, then lower proc id.
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
    dag.remove_edges_from([e for e in dag.edges() if float(dag.get_edge_data(*e)['weight'])==0.0])
    if show_dag:
        try:
            pos=nx.nx_pydot.graphviz_layout(dag, prog='dot')
        except Exception:
            pos=nx.spring_layout(dag, seed=42)
        nx.draw(dag, pos=pos, with_labels=True)
        plt.show()
    return dag

def _adjusted_median_exec(comp:np.ndarray, t:int)->float:
    """Adjusted median E*(t) per paper: median over full row (including inf). If median is inf,
    substitute largest finite exec time (if any); if none finite, return inf."""
    row=comp[t]
    med=np.median(row)
    if np.isfinite(med):
        return float(med)
    finite=row[np.isfinite(row)]
    return float(np.max(finite)) if finite.size else float('inf')

def _compute_static_levels(comp:np.ndarray, dag:nx.DiGraph)->Dict[int,float]:
    # using adjusted median exec time for heterogeneity (SL*)
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError("DAG has no sink")
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks: dag.add_edge(s, virtual, weight=0.0)
        sink=virtual
    else: sink=sinks[0]
    SL={}
    pending=[sink]
    while pending:
        n=pending.pop()
        succs=list(dag.successors(n))
        if any(s not in SL for s in succs):
            pending.insert(0,n); continue
        if n==sink and n>=comp.shape[0]:
            val=0.0
        else:
            base=_adjusted_median_exec(comp,n) if n < comp.shape[0] else 0.0
            if succs:
                val=base+max(SL[s] for s in succs)
            else:
                val=base
        SL[n]=val
        for p in dag.predecessors(n):
            if p not in SL: pending.append(p)
    return SL

def _find_earliest_comm_start(intervals:List[tuple], ready:float, dur:float)->float:
    """Return earliest start >= ready for an interval of length dur given existing sorted intervals."""
    start=ready
    for s,e in intervals:
        if start + dur <= s:
            break  # fits before this interval
        if start < e:
            start = e  # push after overlap
    return start

def _simulate_data_arrival(task:int, dest_proc:int, dag:nx.DiGraph, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], link_schedules:Dict, route_fn):
    """Compute DA(t,dest_proc) and tentative communication reservation plan without mutating state.
    route_fn(src,dst) -> list of link identifiers to reserve as a path. For a fully switched
    crossbar you can return [(src,dst)] to emulate per-pair dedicated links.
    Returns (data_avail, plan) with plan list entries (pred, links_list, start, end).
    If any predecessor requires an infeasible transfer (no finite bw) => returns (inf, [])."""
    data_avail=0.0
    plan=[]
    for pred in dag.predecessors(task):
        ev=task_sched[pred]
        if ev.proc == dest_proc:
            arrival=ev.end
        else:
            src=ev.proc; dst=dest_proc
            data=float(dag[pred][task]['weight'])
            if data==0:
                arrival=ev.end
            else:
                bw=comm[src,dst]
                if bw <= 0:
                    return float('inf'), []  # infeasible path (no bandwidth entry)
                dur=data / bw
                links = route_fn(src,dst)
                start_candidate=ev.end
                start=start_candidate
                while True:
                    adjusted=start
                    for link in links:
                        intervals=link_schedules[link]
                        adjusted=_find_earliest_comm_start(intervals, adjusted, dur)
                    if adjusted==start:  # stable => all links free concurrently
                        start=adjusted; break
                    start=adjusted
                end=start+dur
                plan.append((pred, links, start, end))
                arrival=end
        if arrival>data_avail: data_avail=arrival
    return data_avail, plan

def _commit_comm_plan(plan:List[tuple], link_schedules:Dict):
    for _, links, s, e in plan:
        for link in links:
            lst=link_schedules[link]
            idx=0
            while idx < len(lst) and lst[idx][0] <= s:
                idx += 1
            lst.insert(idx,(s,e))

def _tail_place(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]], link_schedules, route_fn):
    data_avail, plan = _simulate_data_arrival(task, proc, dag, comm, task_sched, link_schedules, route_fn)
    tf = max((ev.end for ev in proc_sched[proc]), default=0.0)
    start = max(data_avail, tf)
    dur = float(comp[task, proc])
    return ScheduleEvent(task, start, start+dur, proc), data_avail, plan

def schedule_dag(dag, computation_matrix, communication_matrix, proc_schedules=None, use_dl1:bool=True, route_fn=None, **kwargs):
    if proc_schedules is None: proc_schedules={p:[] for p in range(communication_matrix.shape[0])}
    # link_schedules[(src,dst)] = sorted list of (start,end) reservations
    link_schedules={}
    P=communication_matrix.shape[0]
    if route_fn is None:
        def route_fn(i,j): return [(i,j)] if i!=j else []
    for i in range(P):
        for j in range(P):
            if i==j: continue
            for link in route_fn(i,j):
                link_schedules.setdefault(link, [])
    SL=_compute_static_levels(computation_matrix, dag)
    tasks=[n for n in dag.nodes() if n < computation_matrix.shape[0]]
    ready=set(t for t in tasks if dag.in_degree(t)==0)
    task_sched:Dict[int,ScheduleEvent]={}
    while len(task_sched)<len(tasks):
        # build candidate list of ready tasks (all predecessors scheduled)
        avail=[t for t in tasks if t not in task_sched and all(pred in task_sched for pred in dag.predecessors(t))]
        if not avail:
            raise RuntimeError("Deadlock in DLS (cyclic DAG?)")
        best_pair=None; best_dl=None; best_placement=None; best_est=None; best_plan=[]
        median_exec_cache={t:_adjusted_median_exec(computation_matrix,t) for t in avail}
        for t in avail:
            for p in range(communication_matrix.shape[0]):
                placement, data_avail, plan = _tail_place(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules, link_schedules, route_fn)
                est=placement.start  # EST = max(DA, TF) already folded into placement
                # DL components
                SL_t=SL[t]
                if use_dl1:
                    E_star=median_exec_cache[t]
                    A = E_star - float(computation_matrix[t,p])
                    dl = SL_t - est + A
                else:
                    dl = SL_t - est
                if best_pair is None or dl > best_dl or (dl==best_dl and (est < best_est or (est==best_est and (t < best_pair[0] or (t==best_pair[0] and p < best_pair[1]))))):
                    best_pair=(t,p); best_dl=dl; best_placement=placement; best_est=est; best_plan=plan
        # commit selection
        t_sel,p_sel=best_pair
        ev=best_placement
        proc_schedules[p_sel].append(ev)
        proc_schedules[p_sel].sort(key=lambda j:j.start)
        task_sched[t_sel]=ev
        # reserve the communication windows now (only those associated with chosen processor)
        _commit_comm_plan(best_plan, link_schedules)
    return proc_schedules, task_sched, {"static_levels":SL}

def _compute_makespan_and_idle(proc_schedules):
    makespan = max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        js=sorted(jobs, key=lambda j:j.start); idle=0.0
        if not js: idle=makespan
        else:
            idle += js[0].start
            for i in range(len(js)-1): idle += max(0.0, js[i+1].start - js[i].end)
            idle += max(0.0, makespan - js[-1].end)
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
        for ev in jobs:
            total += ev.start; count+=1
    return total/count if count else 0.0

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description='Paper-based DLS (DL1 variant)')
    p.add_argument('--dag_file', required=True)
    p.add_argument('--exec_file', required=True)
    p.add_argument('--bw_file', required=True)
    p.add_argument('--use_dl1', action='store_true', help='enable DL1 heterogeneity term (default ON)', default=True)
    a=p.parse_args()
    comp=readCsvToNumpyMatrix(a.exec_file); bw=readCsvToNumpyMatrix(a.bw_file); dag=readDagMatrix(a.dag_file)
    proc_sched, task_sched, meta = schedule_dag(dag, computation_matrix=comp, communication_matrix=bw, use_dl1=a.use_dl1)
    mk,_,_= _compute_makespan_and_idle(proc_sched)
    busy,_,_,_= _compute_load_balance(proc_sched)
    avg=sum(busy.values())/len(busy) if busy else 0.0
    print({'makespan':mk,'load_balance_ratio': (mk/avg if avg>0 else float('inf'))})
