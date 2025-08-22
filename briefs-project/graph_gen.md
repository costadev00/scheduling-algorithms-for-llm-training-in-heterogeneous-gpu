# Graph Generator Reference (`graph_gen.py`)

## Purpose
Generate synthetic DAG scheduling scenarios (HEFT / PEFT) with controllable structure and the five core application parameters:
1. Number of processors (resources)
2. Number of tasks (nodes)
3. Cost of computation (task execution times)
4. Cost of communication (data transfer times across processors)
5. Energy cost (execution energy: time × power)

The script derives computation times from communication characteristics so you can steer relative dominance of compute vs communication. Energy adds an orthogonal dimension governed by the power matrix (optional) combined with the generated execution times.

---
## Core Parameters (Conceptual)
| Concept | Meaning | Primary Switch / Source | How It Influences Outputs |
|---------|---------|------------------------|----------------------------|
| Processors | Count of processing elements (columns) | `--RC` (RC) | Sets matrix sizes (`_resource_BW`, `_task_exe_time`) |
| Tasks | Number of DAG nodes (rows) | `--TC` (TC) | Sets size of connectivity & execution matrices |
| Computation Cost | Per-(task,proc) execution times | Derived via `CCR`, shaped by `HF` | Higher `CCR` → smaller compute times; `HF` spreads times across processors |
| Communication Cost | Time to move data between processors | Edge weights (`CDR`), bandwidths (`LBW`), optional startup | Edge weight / bandwidth (+ startup in HEFT) accumulates to reported comm cost |
| Energy Cost | Sum over tasks: (duration × power) | `_task_power.csv` (optional) + schedule | Enables energy-aware scheduling (ranking + assignment) and reporting |

---
## Generator Inputs & Overrides
Values come first from a config file (default `graph.config`), then any CLI overrides you supply:

| Flag | Config Key | Description |
|------|------------|-------------|
| `--RC` | RC | Number of processors |
| `--TC` | TC | Total tasks (nodes) |
| `--GH` | GH | Graph height (levels) |
| `--AOD` | AOD | Average additional out-degree (extra cross-level edges) |
| `--CCR` | CCR | Communication-to-computation ratio (Comm / Comp). Lower = costlier compute.|
| `--HF` | HF | Heterogeneity factor [0,1]; 0 uniform; 1 wide spread of exec times |
| `--CDR_LOW` / `--CDR_HIGH` | CDR | Edge (data) weight range (affects data volume) |
| `--LBW_LOW` / `--LBW_HIGH` | LBW | Bandwidth range used to fill symmetric resource matrix |
| `--SEED` | SEED | Base random seed |
| `--prefix` | — | Output file prefix (supports `{i}` placeholder with `--repeat`) |
| `--out` | — | Output directory (created if missing) |
| `--repeat` | — | Generate N scenarios (incrementing seed) |

### Derived Internals
- Execution time baseline per task ≈ (estimated average comm time / `CCR`).
- Task heterogeneity: values sampled uniformly in `[mean*(1-HF), mean*(1+HF)]` per processor.
- Communication matrix: symmetric random bandwidths per pair from `LBW` range.
- Edge data sizes: integers from `CDR` range (drives per-edge volume).

---
## Output Files per Scenario
| File | Shape | Contents | Required |
|------|-------|----------|----------|
| `<prefix>_resource_BW.csv` | `RC × RC` | Bandwidth between processors (diagonal 0) | Yes |
| `<prefix>_task_connectivity.csv` | `TC × TC` | DAG adjacency (0 = no edge, >0 = data size) | Yes |
| `<prefix>_task_exe_time.csv` | `TC × RC` | Execution time per (task, processor) | Yes |
| `<prefix>_task_power.csv` | `TC × RC` | Power map for energy / EDP metrics | No |

> HEFT only: A variant bandwidth file may include an extra last row (startup costs). The generator does **not** emit this row; you can append it manually if needed.

---
## Communication vs Computation Control
| Goal | Increase | Decrease |
|------|----------|----------|
| Make communication dominate | `CDR_HIGH`, `CDR_LOW`; lower `LBW_*`; raise `CCR` slightly (keeps compute smaller) | `LBW_*`, `CCR` (too high shrinks compute too much) |
| Make computation dominate | Lower `CDR_*`; raise `LBW_*`; lower `CCR` (inflates compute) | `CCR` (raising reduces compute), `LBW_*` (if reduced) |
| More variability in execution | `HF`→ closer to 1 | `HF`→ 0 |

### Rule of Thumb
```
communication_time(edge) = edge_weight / bandwidth (+ startup for HEFT)
execution_time(task,proc) ~ average_comm_time / CCR  (with HF spread)
task_energy(task) = (end_time - start_time) * power(task, proc)
```
Lower `CCR` ⇒ larger execution times relative to communication. Energy scales linearly with both execution duration and supplied power values.

### Providing Power (Energy) Data (Auto-Activates Energy-Aware Scheduling)
Create `<prefix>_task_power.csv` mirroring the shape of the execution matrix:
```
P,P_0,P_1,...,P_{RC-1}
T_0, 35, 40, ...
T_1, 20, 25, ...
...
```
Guidelines:
- Units: arbitrary (assume Watts); reported energy will be Watt·time_unit.
- Simple model: draw powers uniformly in a range (e.g. 20–60) or proportional to execution time (higher time → higher power or vice versa depending on modeling goal).
- Consistency: keep ranges comparable across scenarios when you want fair energy comparisons.

Once this file exists alongside the other three matrices, both algorithms adapt automatically:
* HEFT: energy-aware rank + composite (finish + normalized energy) processor selection.
* PEFT: energy-minimizing processor choice with OCT/EFT tie-breaks.

Potential future enhancement: `--gen-power MIN MAX` to auto-generate this file (not implemented yet).

---
## Examples
### 1. High Communication, Few Processors
```powershell
python graph_gen\graph_gen.py --RC 2 --TC 40 --CCR 4.0 \
  --CDR_LOW 500 --CDR_HIGH 1200 --LBW_LOW 40 --LBW_HIGH 60 \
  --prefix commHeavy
```
Explanation: Large data edges + modest bandwidth + higher CCR keeps compute comparatively small.

### 2. High Computation, Many Processors
```powershell
python graph_gen\graph_gen.py --RC 8 --TC 40 --CCR 0.5 \
  --CDR_LOW 20 --CDR_HIGH 60 --LBW_LOW 250 --LBW_HIGH 400 \
  --HF 0.6 --prefix compHeavy
```
Explanation: Small edge weights, high bandwidth reduce comm time; low CCR inflates compute.

### 3. Batch 10 Variants (Seed Sweep)
```powershell
python graph_gen\graph_gen.py --RC 4 --TC 60 --CCR 1.5 \
  --CDR_LOW 80 --CDR_HIGH 160 --LBW_LOW 120 --LBW_HIGH 220 \
  --repeat 10 --prefix series_{i}
```
Outputs: `series_0_*`, `series_1_*`, ..., `series_9_*`.

---
## Minimal Invocation
If config already defines everything:
```powershell
python graph_gen\graph_gen.py --config graph.config --out ..\graphs --prefix mygraph
```
Supply overrides as needed, e.g.:
```powershell
python graph_gen\graph_gen.py --config graph.config --RC 6 --TC 120 --CCR 0.8 --prefix tuned
```

---
## Suggested Extensions (Not Yet Implemented)
- Auto power file generation (`_task_power.csv`) based on a model (e.g., proportional to execution time or randomized within a watt range).
- Direct startup latency synthesis into `_resource_BW` with an `--startup LOW HIGH` range.
- Output a JSON summary (`--summary-json`) capturing the four core parameter values for experiment tracking.

Let me know if you want any of these implemented next.
