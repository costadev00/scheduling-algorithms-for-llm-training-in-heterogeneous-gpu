# Scheduling Algorithms for Heterogeneous GPU Task Graphs

This project provides clean, paper-faithful Python implementations of two classic heterogeneous DAG scheduling heuristics used to map a task graph onto multiple GPUs / processing elements (PEs):

- **HEFT** (Heterogeneous Earliest Finish Time) – Topcuoglu et al., 2002
- **PEFT** (Predict Earliest Finish Time) – Arabnejad & Barbosa, 2014

Both implementations are trimmed to their canonical algorithmic logic (no custom energy-aware heuristics). They expose identical helper utilities for loading CSV inputs and computing schedule quality metrics, enabling fair side‑by‑side evaluation.

## Project Goal
Provide a reproducible baseline for comparing HEFT vs. PEFT on heterogeneous GPU task graphs using only the decision rules described in the original papers. This baseline supports research on scheduling for large model (LLM) training pipelines, synthetic DAG benchmarks, and educational exploration of classic list scheduling heuristics.

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
1. Build Optimistic Cost Table (OCT).
2. Rank = mean OCT row; order by descending rank.
3. For each task choose processor minimizing (EFT + OCT[task, proc]).

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

## License
(Add license information here if applicable.)

---
Paper‑only baseline; contributions welcome for optional experimental branches.
