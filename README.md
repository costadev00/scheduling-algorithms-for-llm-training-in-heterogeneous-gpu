<div align="center">

# Scheduling Algorithms for Heterogeneous GPU Task Graphs

Baseline, paper-faithful Python implementations of six classic / illustrative heterogeneous DAG scheduling heuristics for mapping large	 training or pipeline workloads onto multi‑GPU (heterogeneous PE) systems.

</div>

## 1. Purpose & Scope
This repository provides a reproducible reference to compare list‑scheduling heuristics on synthetic or real task graphs that model large model (LLM) training or data‑parallel / pipeline stages. All algorithms share a common I/O format and metric computation so differences reflect their decision logic—not instrumentation artifacts.

Included algorithms:
| Acronym | Description | Notes |
|---------|-------------|-------|
| HEFT | Heterogeneous Earliest Finish Time | Canonical 2002 algorithm |
| PEFT | Predict Earliest Finish Time | OCT continuation costs |
| DLS | Dynamic Level Scheduling (DL1) | Median‑based level scoring |
| HEFT‑LA | 1‑level Lookahead HEFT | Educational, child EFT sum |
| IHEFT | Improved HEFT variant | Heterogeneity weight + stochastic crossover |
| IPEFT | Improved PEFT variant | Dual pessimistic / critical continuation tables |

Energy accounting is post‑hoc only; scheduling is performance‑centric.

## 2. Benchmarking Entrypoint: `compare_same_dataset.py`
`compare_same_dataset.py` is the primary script for this project. It loads one DAG instance (a quadruple of CSV files), runs one or more algorithms, prints metrics as JSON (1 line per algorithm), and optionally writes visual artifacts.

### 2.1 Script Arguments
| Flag | Required | Description |
|------|----------|-------------|
| `--dag` | Yes | Task connectivity CSV (edge weights = data volume) |
| `--exec` | Yes | Task execution time matrix CSV |
| `--bw` | Yes | Processor bandwidth matrix CSV |
| `--power` | No  | Task power matrix (enables energy metric & energy bar) |
| `--algos` | Yes | Comma list subset of `DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT` |
| `--out_dir` | No | Directory to store `gantt_*.png`, `metrics.png`, `dag.png` (if requested) |
| `--save_dag` | No | Generate `dag.png` (Graphviz if available, otherwise fallback layout) |
| `--plot_metrics` | No | Save multi-bar metrics figure (`metrics.png`) |

Minimum viable command (metrics only, console output):
```powershell
python compare_same_dataset.py `
  --dag graphs/canonicalgraph_task_connectivity.csv `
  --exec graphs/canonicalgraph_task_exe_time.csv `
  --bw  graphs/canonicalgraph_resource_BW.csv `
  --algos HEFT,PEFT
```

Full artifact command (adds Gantts + metrics plot):
```powershell
python compare_same_dataset.py `
  --dag graphs/Scenario_1/peft256_8proc_task_connectivity.csv `
  --exec graphs/Scenario_1/peft256_8proc_task_exe_time.csv `
  --bw  graphs/Scenario_1/peft256_8proc_resource_BW.csv `
  --power graphs/Scenario_1/peft256_8proc_task_power.csv `
  --algos DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT `
  --out_dir outputs/scenario1_256x8_labels `
  --save_dag --plot_metrics
```

### 2.2 Batch Running a Scenario Family
Run all processor counts (8/16/32) for Scenario 3 (example PowerShell loop):
```powershell
foreach ($p in 8,16,32) {
  python compare_same_dataset.py `
    --dag   graphs/Scenario_3/peft1024_${p}proc_task_connectivity.csv `
    --exec  graphs/Scenario_3/peft1024_${p}proc_task_exe_time.csv `
    --bw    graphs/Scenario_3/peft1024_${p}proc_resource_BW.csv `
    --power graphs/Scenario_3/peft1024_${p}proc_task_power.csv `
    --algos HEFT,PEFT,DLS,HEFT-LA,IHEFT,IPEFT `
    --out_dir outputs/scenario3_1024x${p}_labels `
    --plot_metrics   # omit --save_dag to speed up large runs
}
```

### 2.3 Capturing Metrics to a File
Append JSON lines to a log while still seeing console output:
```powershell
python compare_same_dataset.py `
  --dag graphs/canonicalgraph_task_connectivity.csv `
  --exec graphs/canonicalgraph_task_exe_time.csv `
  --bw graphs/canonicalgraph_resource_BW.csv `
  --algos HEFT,PEFT,DLS | Tee-Object -FilePath outputs/run_log.jsonl -Append
```

### 2.4 Typical Benchmark Workflow
1. Pick scenario CSV set (or create your own).  
2. Run `compare_same_dataset.py` for each (tasks, processors) variant.  
3. (Optional) Re-run with fewer algorithms to isolate differences.  
4. (Optional) Aggregate with `aggregate_all_results.py` then visualize cross‑scenario trends via `plot_aggregate_results.py`.  
5. Inspect per-run Gantt & metrics to understand differences (look for load balance, comm dominance).  

### 2.5 Dependency Setup
Install requirements (or minimal manual list if the file is empty):
```powershell
python -m pip install -r requirements.txt
# OR minimal manual:
python -m pip install numpy networkx matplotlib pydot
```
Install Graphviz (optional, for nicer DAG layout). Without it the fallback layout path is used automatically.

### 2.6 Performance Tips
| Situation | Tip |
|-----------|-----|
| Large task counts (≥2048) | Drop `--save_dag` to skip heavy layout |
| Many repeated runs | Omit `--plot_metrics` until final pass |
| Focus on scheduling core | Use a reduced algo list (e.g. `--algos HEFT,PEFT`) |
| Fast iteration | Run single small scenario (256×8) before scaling up |

### 2.7 Exit Codes
* Non‑zero exit only if: unsupported algorithm requested OR internal exception while reading CSVs / scheduling.
* Missing algorithms (import failures) are excluded silently—confirm by checking printed JSON lines contain all requested names.

---

## 3. Input Data Model
Each scheduling instance is specified by 3–4 CSV matrices (header row + column):
1. Connectivity (tasks × tasks): edge weight = data volume parent→child (0 if none).
2. Execution times (tasks × processors): time(task, proc).
3. Bandwidth (processors × processors): rate(proc_i, proc_j). Comm time = load / bw if mapped to different procs.
4. (Optional) Power (tasks × processors): power draw used only for energy = Σ (duration × power).

File naming convention for provided scenarios:
`peft{TASKS}_{PROCS}proc_task_connectivity.csv`
`peft{TASKS}_{PROCS}proc_task_exe_time.csv`
`peft{TASKS}_{PROCS}proc_resource_BW.csv`
`peft{TASKS}_{PROCS}proc_task_power.csv`

## 4. Running Provided Scenarios (Alternative View)
Scenarios 1–5 scale task counts (256, 512, 1024, 2048, 4096) and processors (8,16,32). Run one variant (example: Scenario 1, 256 tasks, 8 procs):
```powershell
python compare_same_dataset.py `
  --dag   graphs/Scenario_1/peft256_8proc_task_connectivity.csv `
  --exec  graphs/Scenario_1/peft256_8proc_task_exe_time.csv `
  --bw    graphs/Scenario_1/peft256_8proc_resource_BW.csv `
  --power graphs/Scenario_1/peft256_8proc_task_power.csv `
  --algos DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT `
  --out_dir outputs/scenario1_256x8_labels `
  --save_dag --plot_metrics
```
Repeat by swapping the filenames for `_16proc_` or `_32proc_`, and for other `Scenario_N` directories.

### Fast Re-run Without DAG Image
Skip DAG plotting (faster at large scale) by omitting `--save_dag`.

### Algorithms Subset
Use a subset, e.g. only HEFT & PEFT:
```powershell
--algos HEFT,PEFT
```

## 5. Aggregated Metrics & Global Plots (Optional Layer)
After regenerating scenarios you can rebuild cross‑scenario summaries (if you have the aggregator script populated):
```powershell
python aggregate_all_results.py
python plot_aggregate_results.py   # writes per-scenario & comparative plots to outputs/plots/
```
Artifacts:
* `outputs/aggregate_metrics.jsonl` – one JSON object per (scenario, procs, algorithm)
* `outputs/aggregate_metrics.csv`   – tabular form
* `outputs/plots/*.png`             – metric comparisons (makespan, energy, waiting time, load balance, etc.)

## 6. Output Artifacts (Per Run)
Inside `--out_dir` you may see:
| File | Meaning |
|------|---------|
| `dag.png` | (Optional) DAG visualization (Graphviz or fallback layout) |
| `gantt_<ALG>.png` | Per‑algorithm schedule (processor lanes vs time) |
| `metrics.png` | Bar charts: makespan, load balance ratio, waiting time (+ energy if power provided) |
| Console JSON lines | Machine‑readable metrics for downstream aggregation |

Metric definitions:
* Makespan: max task finish time.
* Load balance ratio: makespan / (average busy time). Closer to 1 is better.
* Waiting time: average task ready‑to‑start latency.
* Communication cost: sum of edge data / bw for cross‑processor edges encountered.
* Energy (optional): Σ (task duration × power(task, proc)).

## 7. Algorithm Decision Logic (Concise)
* **HEFT**: Upward rank (avg exec + max succ (comm + rank)); choose proc with earliest finish via insertion.
* **PEFT**: Precompute optimistic continuation table (OCT); rank by mean OCT row; choose proc minimizing EFT + OCT.
* **DLS (DL1)**: Dynamic level = static level − earliest start + (median exec − exec(task, proc)); pick max each step.
* **HEFT‑LA**: HEFT order; processor score = EFT(task, p) + Σ predicted earliest child EFT if task on p.
* **IHEFT**: Modified upward rank with heterogeneity weight; stochastic crossover between best EFT and fastest local exec.
* **IPEFT**: Dual tables (PCT worst‑case, CNCT critical successors) + slack based criticality; score = EFT + CNCT.

## 8. Adding / Modifying a Scenario
1. Generate new CSV matrices (follow header format of existing ones).  
2. Place them in a new `graphs/Scenario_X/` folder using the same naming pattern.  
3. Run `compare_same_dataset.py` with appropriate `--dag/--exec/--bw/--power` arguments.  
4. (Optional) Re‑aggregate & re‑plot.  

## 9. Limitations / Design Choices
| Aspect | Current Behavior |
|--------|------------------|
| Task granularity | Tasks are atomic (no splitting across processors) |
| Preemption | Not supported (non‑preemptive once started) |
| Energy-aware placement | Not implemented; energy purely observational |
| Network model | Static bandwidth matrix, constant rate |
| Randomness | IHEFT uses fixed seed (42) for reproducibility |

To simulate splitting a large task, manually decompose it into multiple subtasks and adjust edges accordingly (see discussion in issues / docs if added later).

## 10. Troubleshooting
| Symptom | Cause / Fix |
|---------|-------------|
| `"dot" not found` | Install Graphviz or omit `--save_dag` (fallback layout used otherwise). |
| `ModuleNotFoundError: scipy` | Not required; fallback layout path avoids SciPy—ignore or install `scipy`. |
| Blank / empty plots | Ensure algorithms you requested are available (import errors silently drop ones not installed). |
| Very large makespan for DLS | Expected on highly heterogeneous / communication heavy graphs; compare relative metrics. |

## 11. Extending the Codebase
Potential research directions (not in baseline to preserve paper fidelity):
* Multi‑objective (energy / thermal) scoring.
* Online DAG arrivals & merging.
* Divisible / malleable tasks (fractional scheduling) – requires redesign of schedule & metrics.
* Adaptive communication compression models.

## 12. Citation
If you use this baseline, cite the original papers:
* H. Topcuoglu, S. Hariri, M.-Y. Wu, 2002 (HEFT)
* H. Arabnejad, J. Barbosa, 2014 (PEFT)
* G. C. Sih, E. A. Lee, 1993 (DLS)
* T. Hagras, J. Janeček, 2003 (DLS characterization)
* (Add the specific IHEFT / IPEFT source once finalized)

## 13. License
Released under the **MIT License**. See the [`LICENSE`](./LICENSE) file for full text.
You may freely use, modify, distribute, and sublicense the code provided the
copyright and license notices are preserved.

Badge suggestion (optional):
```
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
```

---
Contributions (bug fixes, focused extensions) are welcome—please keep core algorithm logic faithful to source papers.
