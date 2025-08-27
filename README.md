# Scheduling Algorithms for Heterogeneous GPU Task Graphs

This project provides clean, paper-faithful Python implementations of classic heterogeneous DAG scheduling heuristics (plus a simple single‑step lookahead variant) used to map a task graph onto multiple GPUs / processing elements (PEs):

- **HEFT** (Heterogeneous Earliest Finish Time) – Topcuoglu et al., 2002
- **PEFT** (Predict Earliest Finish Time) – Arabnejad & Barbosa, 2014
- **DLS** (Dynamic Level Scheduling, DL1 variant) – Sih & Lee, 1993 (characterized by Hagras & Janeček, 2003)
- **HEFT‑LA** (Lookahead HEFT, 1‑level child EFT sum) – Lightweight extension that scores a candidate processor for task t by EFT(t,p) + Σ predicted earliest child EFTs assuming t placed on p. Demonstrates benefit on high fan‑out / heterogeneous successor patterns.

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

Unified comparison including DLS and HEFT‑LA:
```powershell
python compare_same_dataset.py --dag graphs/canonicalgraph_task_connectivity.csv `
	--exec graphs/canonicalgraph_task_exe_time.csv --bw graphs/canonicalgraph_resource_BW.csv `
	--algos HEFT,PEFT,DLS,HEFT-LA --report
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

### Scenario Evaluations
Scenario 3 (example large DAG comparison) regeneration:
```powershell
python summarize_scenario3.py
python "graphs/scenario 3/scenario3_plots.py"
```
This produces `graphs/scenario 3/scenario3_summary.csv` and plot PNGs plus an updated `SCENARIO3_REPORT.md`.

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
graphs/               # Scenario input CSVs + generated plots/reports
summarize_scenario*.py# Scenario scripts (aggregate metrics across resource counts)
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
```powershell
python compare_same_dataset.py --dag graphs/lookahead_graph_task_connectivity.csv `
	--exec graphs/lookahead_graph_task_exe_time.csv --bw graphs/lookahead_graph_resource_BW.csv `
	--algos HEFT,HEFT-LA,PEFT,DLS --report --tag lookahead_demo
```
Example result (may vary slightly with environment): HEFT makespan 94.5 vs HEFT‑LA 93.5 with markedly lower communication cost due to child‑aware placement of the high fan‑out root.

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

## License
(Add license information here if applicable.)

---
Paper‑only baseline; contributions welcome for optional experimental branches.
