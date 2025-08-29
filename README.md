# Scheduling Algorithms for Heterogeneous GPU Task Graphs

This project provides clean, paper-faithful Python implementations of classic heterogeneous DAG scheduling heuristics (plus a simple single‑step lookahead variant) used to map a task graph onto multiple GPUs / processing elements (PEs):

- **HEFT** (Heterogeneous Earliest Finish Time) – Topcuoglu et al., 2002
- **PEFT** (Predict Earliest Finish Time) – Arabnejad & Barbosa, 2014
- **DLS** (Dynamic Level Scheduling, DL1 variant) – Sih & Lee, 1993 (characterized by Hagras & Janeček, 2003)
- **HEFT‑LA** (Lookahead HEFT, 1‑level child EFT sum) – Lightweight extension that scores a candidate processor for task t by EFT(t,p) + Σ predicted earliest child EFTs assuming t placed on p. Demonstrates benefit on high fan‑out / heterogeneous successor patterns.
- **IHEFT** (Improved HEFT from provided paper) – Modified upward rank using heterogeneity Weight_ni and a stochastic cross-over rule between global EFT-minimizing processor and local fastest-exec processor based on dynamic threshold.
- **IPEFT** (Improved PEFT; dual pessimistic / critical-node cost tables) – Extends PEFT by blending pessimistic and critical-successor–focused continuation costs (PCT & CNCT) plus average earliest/latest start analysis to emphasize likely critical path nodes when ranking and selecting processors.

Both implementations are trimmed to their canonical algorithmic logic (no custom energy-aware heuristics). They expose identical helper utilities for loading CSV inputs and computing schedule quality metrics, enabling fair side‑by‑side evaluation.

## Project Goal
Provide a reproducible baseline for comparing HEFT, PEFT, DLS, and an illustrative HEFT lookahead variant (HEFT‑LA) on heterogeneous GPU task graphs using only the decision rules described in the original papers (except the clearly marked optional lookahead). This baseline supports research on scheduling for large model (LLM) training pipelines, synthetic DAG benchmarks, and educational exploration of classic list scheduling heuristics.

## Input Data Model
Each workload (DAG scheduling instance) is defined by CSV matrices (with header row + column):
- Task connectivity matrix: edge weight = communication load (data volume) from parent to child.
- Task execution matrix: rows=tasks, cols=processors; entry = execution time of task on that processor.
- Processor bandwidth matrix: bandwidth (rate) between processors (diagonal typically 0 or ignored). Communication time = edge load / bandwidth if tasks mapped to different processors; 0 otherwise.
- (Optional) Task power matrix: power(task, processor) values used only for post‑hoc energy accounting (not for scheduling decisions).

## Setup
```powershell
python -m pip install -r requirements.txt
```

## Run (Canonical Example Datasets)
Example CSVs reside under `heft/test/` and `peft/test/`.

HEFT (report + optional Gantt chart):
```powershell
python -m heft.heft --report --showGantt `
	-d heft/test/canonicalgraph_task_connectivity.csv `
	-p heft/test/canonicalgraph_resource_BW.csv `
	-t heft/test/canonicalgraph_task_exe_time.csv
```

PEFT (report + optional Gantt chart):
DLS (DL1 variant; report + optional Gantt chart):
```powershell
python -m dls.dls --report --showGantt `
	--dag_file graphs/canonicalgraph_task_connectivity.csv `
	--exec_file graphs/canonicalgraph_task_exe_time.csv `
	--bw_file graphs/canonicalgraph_resource_BW.csv
```

Unified comparison including DLS, HEFT‑LA and IHEFT:
```powershell
python compare_same_dataset.py --dag graphs/canonicalgraph_task_connectivity.csv `
	--exec graphs/canonicalgraph_task_exe_time.csv --bw graphs/canonicalgraph_resource_BW.csv `
	--algos HEFT,PEFT,DLS,HEFT-LA,IHEFT,IPEFT --report
```
```powershell
python -m peft.peft --report --showGantt `
	-d peft/test/canonicalgraph_task_connectivity.csv `
	-p peft/test/canonicalgraph_resource_BW.csv `
	-t peft/test/canonicalgraph_task_exe_time.csv
```

Minimal run (no Gantt, metrics only) – remove `--showGantt`.

### Output Metrics (`--report`)
Printed for both algorithms:
- Makespan
- Total & per‑processor idle time
- Load balance stats (busy time per processor, coefficient of variation, imbalance ratio, Jain’s fairness)
- Average waiting time (mean task start)
- (If power CSV provided) Energy = Σ(duration × power)

## Testing
Lightweight tests validate metric helpers:
```powershell
pytest
```

## Code Structure
```
heft/
	heft/heft.py        # Canonical HEFT implementation
	heft/gantt.py       # Gantt chart rendering helper
	heft/dag_merge.py   # (Optional) DAG merge utilities for multi-workflow experiments
peft/
	peft/peft.py        # Canonical PEFT implementation
graphs/               # Input CSVs + generated plots/reports
```

## Algorithm Summary
HEFT:
1. Compute upward rank: avg exec time + max(successor rank + normalized comm).
2. Schedule tasks in descending rank; per task choose processor with earliest finish (insertion heuristic).

PEFT:
DLS (DL1 variant implemented):
1. Compute static levels SL* using median execution time per task (heterogeneity-aware).
2. At each step evaluate all (ready task, processor) pairs with DL1(t,p)=SL*(t) - EST(t,p) + (median_exec(t) - exec(t,p)).
3. Pick pair with maximum DL1 (tie-break: lower EST, task id, processor id) and schedule using insertion-based earliest start.
4. Recompute dynamic levels and repeat until all tasks scheduled.
1. Build Optimistic Cost Table (OCT).
2. Rank = mean OCT row; order by descending rank.
3. For each task choose processor minimizing (EFT + OCT[task, proc]).

HEFT‑LA (1‑level lookahead variant, not a published algorithm but a didactic extension):
1. Use the same HEFT upward rank ordering.
2. For each ready task t and processor p, compute EFT(t,p) via insertion policy.
3. For each child c of t, temporarily assume t ends at that EFT on p and estimate earliest child finish on its best processor (ignoring contention) => predicted_EFT_child.
4. Score S(t,p)=EFT(t,p)+Σ predicted_EFT_child; pick p with minimal score (ties: earliest EFT, lower proc id).
5. Falls back to HEFT behavior when fan‑out is 0 or successors homogeneous.

Lookahead demonstration dataset (graphs/lookahead_graph_*):
IHEFT (Improved HEFT):
1. Compute Weight_ni = | (max_exec_i - min_exec_i) / (max_exec_i / min_exec_i) | per task capturing heterogeneity dispersion.
2. Modified upward rank: rank_i = Weight_ni + max_{succ j} ( normalized_comm_{i,j} + rank_j ).
3. Order tasks by decreasing rank.
4. For each task evaluate insertion-based EFT on all processors; record processor with minimum execution time separately.
5. If both processors identical choose it; else compute Weight_abstract = | (EFT_best - EFT_execBest) / (EFT_best / EFT_execBest) | and Cross_Threshold = Weight_ni / Weight_abstract.
6. Draw r ~ Uniform[0.1,0.3]; if Cross_Threshold <= r select fastest-exec processor (local); else select min-EFT (global).
Stochastic element can yield schedules equal to or different from canonical HEFT; seed fixed (42) for reproducibility in current implementation.
```powershell
python compare_same_dataset.py --dag graphs/lookahead_graph_task_connectivity.csv `
	--exec graphs/lookahead_graph_task_exe_time.csv --bw graphs/lookahead_graph_resource_BW.csv `
	--algos HEFT,HEFT-LA,PEFT,DLS --report --tag lookahead_demo
```
Example result (may vary slightly with environment): HEFT makespan 94.5 vs HEFT‑LA 93.5 with markedly lower communication cost due to child‑aware placement of the high fan‑out root.

IPEFT (Improved PEFT):
1. Precompute average execution time per task (used in several averages) and average communication cost (edge load / mean bandwidth) for normalized continuity.
2. For each task compute AEST (Average Earliest Start Time) and ALST (Average Latest Start Time) by propagating mean start bounds forward/backward over processors; this yields slack windows identifying potentially critical successors (low slack = high criticality).
3. Identify critical successors of each task: a child is critical if its (ALST - AEST) slack is below a small epsilon (implementation uses numerical tolerance) or minimal among its sibling set.
4. Build Pessimistic Cost Table (PCT): PCT[i,p] = exec(i,p) + max_{succ j} ( max_{q} ( comm(i,j,p,q) + PCT[j,q] ) ). This assumes worst-case (slowest) continuation across children & processors, biasing ranks toward tasks whose worst descendant chains are long.
5. Build Critical-Node Cost Table (CNCT): Similar recursion but considers only critical successors and takes for each critical successor the best continuation processor: CNCT[i,p] = exec(i,p) + max_{crit succ j} ( min_{q} ( comm(i,j,p,q) + CNCT[j,q] ) ). If no critical successors, CNCT[i,p] = exec(i,p).
6. Rank (rankPCT) of task i = mean_p PCT[i,p] + mean execution time of i (adds intrinsic weight; mirrors provided specification).
7. Order tasks by decreasing rankPCT (list scheduling order).
8. For each ready task, evaluate insertion-based Earliest Finish Time (EFT) on every processor. Processor score = EFT + CNCT[i,p]. Ties broken by earlier EFT then lower processor id.
9. Insert task at earliest legal start (respecting precedence & existing reservations) as in HEFT/PEFT.
10. Proceed until all tasks scheduled; metrics reported identically.

Differences vs PEFT: PEFT relies on a single optimistic (best-continuation) OCT table and ranks via its row mean; IPEFT introduces dual pessimistic (PCT) and critical-focused (CNCT) tables plus critical-successor filtering via AEST/ALST slack, aiming to improve discrimination on true critical path tasks while still guarding against worst-case branching. Official default: critical nodes ARE penalized (i.e. not exempt) so their processor choice still considers CNCT continuation impact. A flag (`--ipeft_exempt_cn`) can be used to revert to the earlier experimental behavior that exempted CN from penalty.

Config toggles: `--ipeft_rank_include_avg_exec` (adds avg exec time into rank) and `--ipeft_exempt_cn` (do not penalize CN). Communication averaging mode is fixed (reciprocal of mean bandwidth) and no longer configurable.

Empirical note (included example runs in this repo): On a dense 120-task synthetic graph IPEFT produced a slightly higher makespan than PEFT in one seed (reflecting heuristic variability) but reduced load imbalance. Behavior will vary with communication heterogeneity; further tuning of slack threshold or CNCT blending may improve results.

## Energy Handling
Energy is not part of placement logic; if a power matrix is supplied it’s aggregated post‑schedule for reporting only.

## Extending
Potential extensions (not included to keep paper fidelity):
- Energy‑aware rank weighting or multi‑objective scoring
- Dynamic voltage/frequency scaling (DVFS) models
- Online arrival of workflows using `dag_merge` strategies

## Citation
If you use this baseline, cite the original papers:
- H. Topcuoglu, S. Hariri, M.-Y. Wu, "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing", IEEE TPDS, 2002.
- H. Arabnejad, J. Barbosa, "List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table", IEEE TPDS, 2014.
- G. C. Sih, E. A. Lee, "A Compile-Time Scheduling Heuristic for Interconnection-Constrained Heterogeneous Processor Architectures", IEEE TPDS (DLS), 1993.
- T. Hagras, J. Janeček, "Static vs. dynamic list-scheduling performance comparison", Acta Polytechnica, 2003.
- (Add citation for the IPEFT improvement paper once bibliographic details are finalized.)

## License
(Add license information here if applicable.)

---
Paper‑only baseline; contributions welcome for optional experimental branches.
