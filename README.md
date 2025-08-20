# Scheduling algorithms for LLM training in heterogeneous GPU

This repository unifies two classic DAG scheduling heuristics implemented in Python:
- HEFT (Heterogeneous Earliest Finish Time) in `heft/`
- PEFT (Predict Earliest Finish Time) in `peft/`

Both tools schedule a task graph (DAG) onto a set of heterogeneous processing elements (PEs), and can render a Gantt chart. A unified Python environment is provided at the repo root.

## Setup

1) Install dependencies once from the repo root:
```
python -m pip install -r requirements.txt
```

2) Optional: run tests for both projects from the root:
```
pytest
```

## Quick start (canonical datasets)

The canonical task, bandwidth, and execution-time CSVs are included under each project’s `test/` folder.

- HEFT: run and show a Gantt chart
```
python -m heft.heft --showGantt \
	-d heft/test/canonicalgraph_task_connectivity.csv \
	-p heft/test/canonicalgraph_resource_BW.csv \
	-t heft/test/canonicalgraph_task_exe_time.csv
```

- PEFT: run and show a Gantt chart
```
python -m peft.peft --showGantt \
	-d peft/test/canonicalgraph_task_connectivity.csv \
	-p peft/test/canonicalgraph_resource_BW.csv \
	-t peft/test/canonicalgraph_task_exe_time.csv
```

## Metrics report

Both CLIs support `--report` to print:
- Makespan
- Total and per-processor idle time
- Load balancing metrics (busy time per processor, coefficient of variation, imbalance ratio, Jain’s fairness)

Examples:

- HEFT with metrics:
```
python -m heft.heft --report \
	-d heft/test/canonicalgraph_task_connectivity.csv \
	-p heft/test/canonicalgraph_resource_BW.csv \
	-t heft/test/canonicalgraph_task_exe_time.csv
```

- PEFT with metrics:
```
python -m peft.peft --report \
	-d peft/test/canonicalgraph_task_connectivity.csv \
	-p peft/test/canonicalgraph_resource_BW.csv \
	-t peft/test/canonicalgraph_task_exe_time.csv
```

Run metrics and chart together:

- HEFT (report + Gantt):
```
python -m heft.heft --report --showGantt \
	-d heft/test/canonicalgraph_task_connectivity.csv \
	-p heft/test/canonicalgraph_resource_BW.csv \
	-t heft/test/canonicalgraph_task_exe_time.csv
```

- PEFT (report + Gantt):
```
python -m peft.peft --report --showGantt \
	-d peft/test/canonicalgraph_task_connectivity.csv \
	-p peft/test/canonicalgraph_resource_BW.csv \
	-t peft/test/canonicalgraph_task_exe_time.csv
```

## HEFT EDP modes (optional)

HEFT also supports energy–delay product objectives with `--op_mode "EDP RELATIVE"` or `"EDP ABSOLUTE"`. These require a power CSV:
```
python -m heft.heft --op_mode "EDP RELATIVE" --power_file heft/test/canonicalgraph_task_power.csv \
	-d heft/test/canonicalgraph_task_connectivity.csv \
	-p heft/test/canonicalgraph_resource_BW.csv \
	-t heft/test/canonicalgraph_task_exe_time.csv --showGantt
```

Notes:
- PEFT ignores `--power_file` (only HEFT supports EDP modes).
- CSVs use a header row/column; the loaders strip the headers.
