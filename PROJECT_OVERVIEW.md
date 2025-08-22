# Scheduling Algorithms for Heterogeneous GPU Training — Project Overview

## 1. Purpose
Provide reference implementations of two classic list‑scheduling algorithms (HEFT and PEFT) adapted for heterogeneous GPU (or generic accelerator) clusters, plus utilities to generate synthetic DAG workloads, visualize task graphs / schedules, and evaluate scheduling quality via multiple metrics.

## 2. Core Algorithms
| Algorithm | Key Idea | Ranking Phase | Processor Selection |
|-----------|----------|---------------|---------------------|
| HEFT (Heterogeneous Earliest Finish Time) | Prioritize tasks by upward rank then pick processor giving earliest finish (energy-aware if power provided) | Rank-U (or EDP variant auto-used if power file) | EFT or composite (finish + normalized energy) |
| PEFT (Predict Earliest Finish Time) | Improves ordering with Optimistic Cost Table (OCT) | OCT (optimistic future completion) | EFT (or energy-minimizing if power file) |

Common mechanics:
- DAG represented as adjacency (weight = communication data size) matrix.
- Execution time matrix (tasks × processors) encodes heterogeneity.
- Communication matrix (processors × processors) + optional startup row.
- List scheduling loop: pick next ready task (by priority metric) → evaluate EFT on each processor → choose minimal finish time.

## 3. Repository Structure (selected)
```
heft/                HEFT package (module: heft.heft)
peft/                PEFT package (module: peft.peft)
graphs/              Centralized dataset CSVs (all DAG + runtime + bandwidth sets)
graph_gen/           Global DAG + matrices generator (graph_gen.py, graph.config)
briefs-project/      Focused explanatory markdown (e.g., breakdowns)
PROJECT_OVERVIEW.md  (this file)
```

## 4. Dataset Triplets (+ Optional Extras)
For a dataset prefix X (e.g. `canonicalgraph`):
- X`_task_connectivity.csv`  : v×v adjacency matrix (header row+col present; numeric block after trimming)
- X`_task_exe_time.csv`      : v×q execution times (task i on processor j)
- X`_resource_BW.csv`        : q×q bandwidth (or (q+1)×q where extra last row = startup L)

Optional supplementary files (auto‑detected when present):
- X`_resource_BW_startup.csv` (alternate explicit startup row variant)
- X`_task_power.csv` (per task power draw for energy / EDP extensions; currently placeholder logic)

CSV conventions:
- First row & first column are labels → stripped on load.
- Zero on DAG edge means “no edge”.
- Diagonal of bandwidth matrix should be 0.

## 5. Global Graph Generator
File: `graph_gen/graph_gen.py`
Config: `graph_gen/graph.config`

Parameters (in config):
- RC (resource count), GH (graph height-ish control), TC (task count), AOD (avg out-degree), CCR (comm/comp ratio factor), HF (heterogeneity factor), CDR (comm density randomness), LBW (baseline bandwidth), SEED.

Usage (PowerShell, from repo root):
```powershell
python graph_gen\graph_gen.py --config graph_gen\graph.config --out graphs --prefix mygraph
```
Outputs: `mygraph_task_connectivity.csv`, `mygraph_task_exe_time.csv`, `mygraph_resource_BW.csv` into `graphs/`.
Repeat with different `--prefix` / modify config / override params via CLI flags (see `--help`).

## 6. Running the Schedulers
Run from repo root (absolute) or `cd` into package directory (relative). Examples (relative usage shown):

HEFT:
```powershell
cd heft
python -m heft.heft -d ..\graphs\canonicalgraph_task_connectivity.csv -t ..\graphs\canonicalgraph_task_exe_time.csv -p ..\graphs\canonicalgraph_resource_BW.csv --report --showDAG --showGantt
```
PEFT:
```powershell
cd ..\peft
python -m peft.peft -d ..\graphs\peftgraph_task_connectivity.csv -t ..\graphs\peftgraph_task_exe_time.csv -p ..\graphs\peftgraph_resource_BW.csv --report --showDAG --showGantt
```
(Use absolute paths if preferred; omit `--showDAG` if Graphviz not installed—spring layout fallback will still plot.)

## 7. Metrics (Lean Set & Interaction with Scheduling)
Reported with `--report`:
* Makespan – Max finish time across all processors.
* Load Balance Ratio – makespan / average busy time (1.0 ideal; >1 implies imbalance).
* Communication Cost – Realized sum of (data_size / bandwidth [+ startup if present]) over inter‑processor edges.
* Waiting Time – Average task start time.
* Energy Cost – Sum of (task duration × task power) when a power file is supplied.

How metrics affect scheduling decisions now (no extra flags):
* If a power file is present, HEFT automatically switches its ranking to an energy‑aware (EDP-like) upward rank and uses a composite objective (finish time + normalized energy) for processor choice.
* If a power file is present, PEFT automatically chooses the processor minimizing (duration × power) with EFT/OCT tie‑breakers; otherwise pure EFT.
* Without a power file both algorithms revert to pure time-based goals (classic HEFT / PEFT behavior).

Internal concepts (not printed): Rank‑U / EDP-Rank (HEFT), OCT (PEFT), EFT (gap insertion). Energy integration is transparent once `_task_power.csv` is available.

Supplying power data simply requires adding the matching `*_task_power.csv` file beside the other three matrices; no special energy flags are needed.

## 8. Visualization
Flags:
- `--showDAG`: renders DAG (Graphviz preferred; spring fallback).
- `--showGantt`: renders schedule timeline with task bars per processor.

## 9. Extensibility Points
Where to add improvements:
- Additional Ranking Heuristics: implement new rank function; plug into selection phase.
- Energy Model: integrate `_task_power.csv` into cost function (e.g., multi‑objective or weighted sum with time).
- Multi‑Run Benchmark Script: iterate over all `graphs/*_task_connectivity.csv` and compile comparison table.
- Test Suite Restoration: adapt `pytest.ini` or relocate tests so they discover centralized `graphs/` data.

## 10. Quick Comparison (Qualitative)
- HEFT: Fast, widely used baseline, effective for many heterogeneous contexts.
- PEFT: Often reduces makespan vs HEFT on communication-heavy graphs by improved ordering (OCT) at slight extra preprocessing cost.

## 11. Roadmap Ideas
1. Reinstate & expand unit tests (parse, rank correctness, schedule validity invariants, metric calculations).
2. Batch evaluation CLI (e.g., `python tools\benchmark.py --algos heft peft --glob graphs/*_task_connectivity.csv`).
3. Energy / Power integration with optional objectives (time, energy, EDP Pareto report).
4. JSON export of schedules (interoperability with visualization dashboards).
5. CI workflow (lint + tests) via GitHub Actions.
6. Performance profiling & optimization of hot loops (candidate reuse, vectorization of EFT computations).

## 12. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| DAG plot blank / warning about Graphviz | `dot` not on PATH | Install Graphviz or ignore (spring layout used) |
| ValueError parsing CSV | Header/shape mismatch | Check first row/col present; consistent v, q across files |
| Long scheduling time for very large graphs | O(v·q·log q + edges) scaling + Python overhead | Profile; consider pruning candidate processors or Cythonizing |

## 13. Dependencies
Core: `numpy`, `networkx`, `matplotlib` (see `requirements.txt`).
Optional: `pydot` + Graphviz binary (`dot`) for improved DAG layout.

## 14. License / Attribution
(Insert license info here if adding a LICENSE file later.)

---
Maintainers: Feel free to edit this overview as architecture evolves.
