# Random50 Benchmark Report

Tag: `random50`
Tasks: 50
Processors: 4
Edge probability base: 0.18 (with enforced connectivity for each non-entry node)

| Algorithm | Makespan | Load Balance Ratio | Communication Cost | Avg Waiting Time |
|-----------|----------|--------------------|--------------------|------------------|
| HEFT      | 547.214  | 1.6633             | 307.627            | 232.9942         |
| PEFT      | 545.514  | 1.6518             | 305.885            | 232.6071         |
| DLS       | 547.214  | 1.6633             | 307.627            | 232.9942         |
| HEFT-LA   | 547.214  | 1.6633             | 307.627            | 232.9942         |
| IHEFT     | 547.214  | 1.6633             | 307.627            | 232.9942         |

(Values rounded; energy not computed.)

## Observations
- PEFT achieved a slightly shorter makespan (~1.7 units improvement, ~0.31%) and marginally lower communication cost.
- All other heuristics converged to the same schedule on this random instance (identical metrics), indicating similar critical path and processor choices under their decision rules.
- The small improvement from PEFT suggests OCT look-ahead captured a minor downstream benefit not differentiated by the weight/rank variants.

## Reproduction
```powershell
python graphs/generate_random50.py
python compare_same_dataset.py --dag graphs/random50_task_connectivity.csv `
  --exec graphs/random50_task_exe_time.csv --bw graphs/random50_resource_BW.csv `
  --algos HEFT,PEFT,DLS,HEFT-LA,IHEFT --tag random50 --report \
  --json_out graphs/random50_results.json
```

---
Generated automatically.
