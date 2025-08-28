"""Paper-based IPEFT implementation.

Key features vs PEFT:
    * Dual tables: PCT (pessimistic) + CNCT (critical-node constrained) instead of single OCT.
    * AEST / ALST based identification of Critical Nodes (CN) and Critical Node Parents (CNP).
    * Rank(t) = mean(PCT[t,*]) + avg_exec_time(t).
    * Processor selection minimizes EFT (+ CNCT penalty unless task is CNP).

Recent correctness fixes:
    * Proper ALST initialization across multiple exits: exits no longer all forced critical; ALST exit values set using global makespan T (see _compute_AEST_ALST).
    * Deterministic ready tie-breaking: (rank, out_degree, -task_id).
    * Defensive handling of zero-bandwidth edges: scheduling on a processor with unreachable comm path yields infinite EFT (pruned by selection).
    * Input shape assertions to catch mismatches early.
    * Removed unused avg-comm helpers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np, networkx as nx, matplotlib.pyplot as plt

@dataclass
class ScheduleEvent:
    task:int; start:float; end:float; proc:int

def readCsvToNumpyMatrix(csv_file: str) -> np.ndarray:
    with open(csv_file) as fd:
        rows=[r.strip().split(',') for r in fd.read().strip().splitlines() if r.strip()]
    return np.array(rows)[1:,1:].astype(float)

def readCsvToDict(csv_file: str):
    m=readCsvToNumpyMatrix(csv_file)
    return {i:row for i,row in enumerate(m)}

def readDagMatrix(dag_file: str, show_dag: bool=False):
    """Read adjacency matrix CSV and build DAG excluding zero-weight (absent) edges.

    Prevents creation of spurious zero-weight edges that could introduce cycles on
    dense zero blocks when using networkx.DiGraph(matrix) directly.
    """
    m=readCsvToNumpyMatrix(dag_file)
    G=nx.DiGraph()
    n=m.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            w=float(m[i,j])
            if w>0.0:
                G.add_edge(i,j,weight=w)
    if show_dag:
        try:
            pos=nx.nx_pydot.graphviz_layout(G, prog='dot')
        except Exception:
            pos=nx.spring_layout(G, seed=42)
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()
    return G

def _avg_exec(comp:np.ndarray, t:int)->float:
    return float(np.mean(comp[t])) if t < comp.shape[0] else 0.0

def _avg_comm_time_per_unit(comm:np.ndarray)->float:
    """Mean of reciprocals of bandwidths (expected time per unit data across random processor pair)."""
    mask=np.ones_like(comm, dtype=bool); np.fill_diagonal(mask, False)
    bw=comm[mask]; bw=bw[bw>0]
    return float(np.mean(1.0/bw)) if len(bw) else 1.0

def _compute_AEST_ALST(dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray):
    """Compute AEST / ALST using average execution and communication times.

    Critical fix: For multiple exits, ALST for exits must be derived from global makespan T so
    that only true critical nodes satisfy AEST==ALST, preventing all exits being misclassified as CN.
    """
    # Average time per unit data: mean of reciprocals
    mask=np.ones_like(comm, dtype=bool); np.fill_diagonal(mask, False)
    bw=comm[mask]; bw=bw[bw>0]
    avg_time=float(np.mean(1.0/bw)) if len(bw) else 1.0
    avg_w={t: float(np.mean(comp[t])) if t < comp.shape[0] else 0.0 for t in dag.nodes()}
    avg_c={(u,v): float(dag[u][v]['weight']) * avg_time for u,v in dag.edges()}
    # Forward AEST (longest with avg costs)
    from collections import deque
    indeg={n:dag.in_degree(n) for n in dag.nodes()}
    q=deque([n for n,d in indeg.items() if d==0])
    AEST={n:0.0 for n in q}; topo=[]
    while q:
        n=q.popleft(); topo.append(n)
        for s in dag.successors(n):
            est=AEST[n] + avg_w.get(n,0.0) + avg_c[(n,s)]
            if s not in AEST or est > AEST[s]:
                AEST[s]=est
            indeg[s]-=1
            if indeg[s]==0:
                q.append(s)
    # Backward ALST using global makespan for exits
    rev=topo[::-1]
    exit_nodes=[n for n in dag.nodes() if dag.out_degree(n)==0]
    T=max((AEST.get(e,0.0) + avg_w.get(e,0.0) for e in exit_nodes), default=0.0)
    ALST={e: T - avg_w.get(e,0.0) for e in exit_nodes}
    for n in rev:
        if n in exit_nodes: continue
        succs=list(dag.successors(n))
        cand=[ALST[s] - avg_c[(n,s)] for s in succs if s in ALST]
        if cand:
            ALST[n] = min(cand) - avg_w.get(n,0.0)
    return AEST, ALST, avg_w, avg_c

def _identify_cn_cnp(dag:nx.DiGraph, AEST:Dict[int,float], ALST:Dict[int,float], tol:float=1e-7):
    """Identify Critical Nodes (CN) and Critical Node Parents (CNP) per paper definitions.

    CN  iff AEST(t) == ALST(t) (within tolerance).
    CNP iff t not in CN and t has at least one immediate successor in CN.
    crit_succ[u] = set of CN successors of u (used by CNCT restriction).
    """
    CN={t for t in dag.nodes() if abs(ALST.get(t,0.0) - AEST.get(t,0.0)) <= tol}
    crit_succ={u:{v for v in dag.successors(u) if v in CN} for u in dag.nodes()}
    CNP={t:(t not in CN and len(crit_succ[t])>0) for t in dag.nodes()}
    return CN, crit_succ, CNP

def _compute_PCT(dag:nx.DiGraph, comp:np.ndarray, avg_c:Dict[Tuple[int,int],float]):
    """Pessimistic Cost Table using average communication time c_{ix}.
    PCT[i,p] = max_{succ j} ( max_{q} ( c_{i,j} (0 if q==p) + w_{j,q} + PCT[j,q] ) )
    (Current row excludes w_{i,p}; rank adds avg w_i later as per implementation spec.)
    """
    P=comp.shape[1]
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError('No sink')
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks: dag.add_edge(s,virtual,weight=0.0)
        sinks=[virtual]
    sink=sinks[0]
    PCT={sink:[0.0]*P}
    pending=[n for n in dag.predecessors(sink)]
    while pending:
        n=pending.pop(0)
        succs=list(dag.successors(n))
        if any(s not in PCT for s in succs):
            pending.append(n); continue
        row=[0.0]*P
        for pj in range(P):
            max_succ=-1e30
            for sx in succs:
                inner_max=-1e30
                for pk in range(P):
                    comm_cost = 0.0 if pk==pj else avg_c.get((n,sx),0.0)
                    val = PCT[sx][pk] + (comp[sx,pk] if sx < comp.shape[0] else 0.0) + comm_cost
                    if val>inner_max: inner_max=val
                if inner_max>max_succ: max_succ=inner_max
            row[pj]=max_succ
        PCT[n]=row
        for p in dag.predecessors(n):
            if p not in PCT and p not in pending: pending.append(p)
    return PCT

def _compute_CNCT(dag:nx.DiGraph, comp:np.ndarray, avg_c:Dict[Tuple[int,int],float], crit_succ:Dict[int,Set[int]]):
    """Critical-Node Cost Table using only CN successors when present.
    CNCT[i,p] = max_{crit succ j in CN(i) or all succ if none CN} ( min_{q} ( c_{i,j} (0 if q==p) + w_{j,q} + CNCT[j,q] ) ).
    """
    P=comp.shape[1]
    sinks=[n for n in dag.nodes() if dag.out_degree(n)==0]
    if not sinks: raise ValueError('No sink')
    if len(sinks)>1:
        virtual=max(dag.nodes())+1
        for s in sinks: dag.add_edge(s,virtual,weight=0.0)
        sinks=[virtual]
    sink=sinks[0]
    CNCT={sink:[0.0]*P}
    pending=[n for n in dag.predecessors(sink)]
    while pending:
        n=pending.pop(0)
        succs=list(dag.successors(n))
        cn_succ=[s for s in succs if s in crit_succ.get(n,set())]
        use_succ=cn_succ if cn_succ else succs
        if any(s not in CNCT for s in use_succ):
            pending.append(n); continue
        row=[0.0]*P
        for pj in range(P):
            max_succ=-1e30
            for sx in use_succ:
                inner_min=1e30
                for pk in range(P):
                    comm_cost = 0.0 if pk==pj else avg_c.get((n,sx),0.0)
                    val = CNCT[sx][pk] + (comp[sx,pk] if sx < comp.shape[0] else 0.0) + comm_cost
                    if val<inner_min: inner_min=val
                if inner_min>max_succ: max_succ=inner_min
            row[pj]=max_succ
        CNCT[n]=row
        for p in dag.predecessors(n):
            if p not in CNCT and p not in pending: pending.append(p)
    return CNCT

def _rank_pct(PCT:Dict[int,List[float]], comp:np.ndarray):
    ranks={}
    for t,row in PCT.items():
        avg_w = float(np.mean(comp[t])) if t < comp.shape[0] else 0.0
        ranks[t] = float(np.mean(row)) + avg_w
    return ranks

def _insertion_eft(task:int, proc:int, dag:nx.DiGraph, comp:np.ndarray, comm:np.ndarray, task_sched:Dict[int,ScheduleEvent], proc_sched:Dict[int,List[ScheduleEvent]]):
    """HEFT-style gap insertion with defensive zero-bandwidth handling.
    If any required inter-processor edge has bw<=0, returns an infinite-time event (ignored by selection)."""
    # Detect infeasible comm early
    for pred in dag.predecessors(task):
        sj=task_sched[pred]
        if sj.proc != proc:
            data=float(dag[pred][task]['weight'])
            if data>0 and comm[sj.proc, proc] <= 0:
                return ScheduleEvent(task, float('inf'), float('inf'), proc)
    ready=0.0
    for pred in dag.predecessors(task):
        sj=task_sched[pred]
        if sj.proc==proc:
            arr=sj.end
        else:
            data=float(dag[pred][task]['weight'])
            if data==0:
                arr=sj.end
            else:
                bw=comm[sj.proc, proc]
                if bw<=0:
                    return ScheduleEvent(task, float('inf'), float('inf'), proc)
                arr=sj.end + data / bw
        if arr>ready: ready=arr
    dur=float(comp[task,proc])
    jobs=proc_sched[proc]
    best=ScheduleEvent(task, ready, ready+dur, proc)
    for i,j in enumerate(jobs):
        if i==0 and j.start - dur >= ready:
            cand=ScheduleEvent(task, ready, ready+dur, proc)
            if cand.end<best.end: best=cand
        if i < len(jobs)-1:
            nxt=jobs[i+1]; gap_start=max(ready,j.end); gap_end=nxt.start
            if gap_end-gap_start >= dur:
                cand=ScheduleEvent(task, gap_start, gap_start+dur, proc)
                if cand.end<best.end: best=cand
        if i==len(jobs)-1:
            start=max(ready,j.end)
            cand=ScheduleEvent(task,start,start+dur,proc)
            if cand.end<best.end: best=cand
    return best

def schedule_dag(dag, computation_matrix, communication_matrix, proc_schedules=None, **kwargs):
    # Shape & consistency checks
    P=communication_matrix.shape[0]
    assert communication_matrix.shape==(P,P), "communication_matrix must be square PxP"
    assert computation_matrix.shape[1]==P, "computation_matrix column count must match #processors"
    assert computation_matrix.shape[0]==dag.number_of_nodes(), "computation_matrix rows must equal number of DAG nodes"
    if proc_schedules is None: proc_schedules={p:[] for p in range(P)}
    # Precompute AEST/ALST and comm times
    AEST, ALST, avg_w_map, avg_c = _compute_AEST_ALST(dag, computation_matrix, communication_matrix)
    # Identify CN / CNP sets
    CN, crit_succ, CNP = _identify_cn_cnp(dag, AEST, ALST)
    # Build cost tables using average communication time (zero when same processor)
    PCT=_compute_PCT(dag.copy(), computation_matrix, avg_c)
    CNCT=_compute_CNCT(dag.copy(), computation_matrix, avg_c, crit_succ)
    ranks=_rank_pct(PCT, computation_matrix)
    tasks=[n for n in dag.nodes() if n < computation_matrix.shape[0]]
    unscheduled=set(tasks)
    in_deg={t:dag.in_degree(t) for t in tasks}
    ready={t for t in tasks if in_deg[t]==0}
    task_sched:Dict[int,ScheduleEvent]={}
    for p in range(communication_matrix.shape[0]):
        proc_schedules.setdefault(p,[])
    while ready:
        # Highest rank with deterministic tie-break (rank, out_degree, -task_id)
        t=max(ready, key=lambda n: (ranks.get(n,0.0), dag.out_degree(n), -n))
        is_cnp=CNP.get(t, False)
        best=None; best_score=None
        for p in range(communication_matrix.shape[0]):
            eft_event=_insertion_eft(t,p,dag,computation_matrix,communication_matrix,task_sched,proc_schedules)
            eft=eft_event.end
            score = eft if is_cnp else eft + CNCT.get(t,[0.0]*communication_matrix.shape[1])[p]
            if best is None or score < best_score or (score==best_score and eft < best.end) or (score==best_score and eft==best.end and p < best.proc):
                best=eft_event; best_score=score
        task_sched[t]=best  # type: ignore
        proc_schedules[best.proc].append(best)  # type: ignore
        proc_schedules[best.proc].sort(key=lambda j:j.start)
        ready.remove(t); unscheduled.remove(t)
        for s in dag.successors(t):
            if s in unscheduled:
                in_deg[s]-=1
                if in_deg[s]==0: ready.add(s)
    if unscheduled:
        raise RuntimeError('Cycle or unreachable tasks in DAG (unscheduled remaining).')
    return proc_schedules, task_sched, {"PCT":PCT, "CNCT":CNCT, "AEST":AEST, "ALST":ALST, "crit_succ":crit_succ, "CN":CN, "CNP":CNP}

def _compute_makespan_and_idle(proc_schedules):
    makespan=max((ev.end for jobs in proc_schedules.values() for ev in jobs), default=0.0)
    total_idle=0.0; per_proc={}
    for p,jobs in proc_schedules.items():
        js=sorted(jobs,key=lambda j:j.start); idle=0.0
        if not js: idle=makespan
        else:
            idle+=js[0].start
            for i in range(len(js)-1): idle+=max(0.0, js[i+1].start - js[i].end)
            idle+=max(0.0, makespan - js[-1].end)
        per_proc[p]=idle; total_idle+=idle
    return makespan,total_idle,per_proc

def _compute_load_balance(proc_schedules):
    busy={p:sum(ev.end-ev.start for ev in jobs) for p,jobs in proc_schedules.items()}
    vals=list(busy.values()); n=len(vals)
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
    starts=[ev.start for jobs in proc_schedules.values() for ev in jobs]
    return float(np.mean(starts)) if starts else 0.0

__all__=[
    'schedule_dag','readCsvToNumpyMatrix','readDagMatrix','readCsvToDict',
    '_compute_makespan_and_idle','_compute_load_balance','_compute_communication_cost','_compute_waiting_time'
]
