# HEFT PowerShell Command — Markdown Breakdown

Here’s a clear breakdown of what that PowerShell command does and how the script processes it.

## What the command does

It runs two commands in sequence; PowerShell uses `;` to chain commands:

1. Change directory to the HEFT package folder, so relative paths resolve:

```powershell
cd C:\Users\mathe\Documents\GitHub\scheduling-algorithms-for-llm-training-in-heterogeneous-gpu\heft
```

2. Run the HEFT module as a script with options:

```powershell
python -m heft.heft --report --showDAG -d test\canonicalgraph_task_connectivity.csv -p test\canonicalgraph_resource_BW.csv -t test\canonicalgraph_task_exe_time.csv
```

### `python -m heft.heft`

* Uses Python’s “run module” mode to execute the `heft.py` entry point.
* The current directory contains the `heft` package, so it’s discoverable on `sys.path`.

## The flags and inputs

* `--report`: print schedule metrics (makespan, idle per processor, load-balance stats).
* `--showDAG`: show a plot of the input DAG.

  * Prefers Graphviz `dot` layout if available; otherwise falls back to a spring layout with a warning.
* `-d`: path to the DAG connectivity CSV (adjacency matrix).

  * Here: `test\canonicalgraph_task_connectivity.csv`
* `-p`: path to the processor-to-processor bandwidth/latency CSV.

  * Here: `test\canonicalgraph_resource_BW.csv`
  * If the matrix is non-square with one extra final row, that last row is treated as communication startup costs (L), and the top `q×q` part is the bandwidth matrix (C).
  * If square, startup costs default to zeros.
* `-t`: path to the task execution time CSV (`v×q`), where `v = number of tasks`, `q = number of processors`.

  * Here: `test\canonicalgraph_task_exe_time.csv`

**CSV format note:** All three CSVs are expected to have a header row and header column (labels). The loader strips the first row and first column and converts the rest to floats.

## What the script does internally

### Parse args and logging

* Parse args and configure logging based on `-l/--loglevel` (default `INFO`).

### Read inputs

* `readCsvToNumpyMatrix()` loads and trims each CSV to a numeric NumPy array.

#### `readDagMatrix()`

* Builds a NetworkX `DiGraph` from the connectivity matrix.
* Removes edges with weight `0` (no edge).
* If `--showDAG` is set, attempts to plot using Graphviz; falls back to spring layout if Graphviz’s `dot` isn’t found.

### Preprocess communication matrix

* If it has one extra row, the last row becomes the startup vector **L** (one per processor) and the square part is used as **C**.
* Otherwise **L** is a zero vector.

### Schedule with HEFT

* Computes **Upward Rank (Rank-U)** per task (default metric: **MEAN** of per-processor times + average comm weight to successors).
* Sorts tasks by decreasing rank, with the root scheduled first.
* For each task, computes the **earliest finish time (EFT)** on each processor:

  * Accounts for data-ready time based on predecessors, inter-processor comm bandwidth, and startup costs.
  * Finds the earliest gap on that processor’s timeline to place the task without overlaps.
* Selects the processor giving the minimum EFT and appends a `ScheduleEvent(task, start, end, proc)`.

### Output and report

* Logs each processor’s job list.
* If `--report`, computes and prints:

  * **Makespan** (max end time across all processors).
  * **Total idle time** (sum of idle within `[0, makespan]` across processors).
  * **Per-processor idle**.
  * **Average waiting time** (mean task start time).
  * **Load-balance metrics**:

    * Coefficient of variation of busy time.
    * Imbalance ratio (max/min busy).
    * Jain’s fairness index.

### Optional Gantt

* If `--showGantt` was passed (not in your command), it displays a Gantt chart of the schedule.

## File shape expectations

* **DAG (`-d`)**: `v×v` adjacency matrix (after removing header row/col).
* **Execution times (`-t`)**: `v×q` matrix **W** (task vs. processor runtime).
* **Processor connectivity (`-p`)**:

  * Square `q×q` for bandwidth matrix **C**; or `(q+1)×q` where the last row is startup costs **L**.
  * Entries on the diagonal of **C** should be `0` (no inter-processor comm cost to itself).

## Path and working directory tips

* Because you `cd` into the `heft` folder first, relative paths like `test\...` resolve to `heft\test\...`.
* If you run from the repo root instead, either:

  * Use absolute paths, **or**
  * Adjust the relative paths, e.g. `-d heft\test\canonicalgraph_task_connectivity.csv`.

## Dependencies

* **Required:** `numpy`, `networkx`, `matplotlib`.
* **For Graphviz layout (prettier DAG):** `pydot` and the Graphviz `dot` binary on `PATH`.

  * Without it, the script falls back to a spring layout and still runs.

---

**Full flow:** parse args → load matrices → build DAG → compute Rank-U → schedule tasks via EFT → print/log metrics → (optionally) plot DAG and Gantt.
