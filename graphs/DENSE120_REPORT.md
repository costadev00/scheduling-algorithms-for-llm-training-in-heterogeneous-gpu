# Dense120 Benchmark Report

Tag: `dense120`
Tasks: 120  |  Processors: 4  |  Edge probability: 0.55 (high connectivity)
Execution time range: 5–160  |  Communication weight range: 5–80

| Algorithm | Makespan | Load Balance Ratio | Communication Cost | Avg Waiting Time |
|-----------|----------|--------------------|--------------------|------------------|
| HEFT      | 3175.244 | 3.03851            | 8816.93            | 1540.188         |
| PEFT      | 3169.176 | 3.03198            | 8635.90            | 1538.719         |
| DLS       | 3175.244 | 3.03851            | 8816.93            | 1540.188         |
| HEFT-LA   | 3169.176 | 3.03125            | 8716.96            | 1538.708         |
| IHEFT     | 3175.244 | 3.03851            | 8816.93            | 1540.188         |

(Values rounded; energy not computed.)

## Observations
- PEFT and HEFT-LA achieve a modest makespan reduction (~6.07 units, ~0.19%) relative to HEFT/DLS/IHEFT, showing advantage of look-ahead style reasoning in dense graphs with heavy comm.
- PEFT also delivers the lowest communication cost (≈ -2.1% vs HEFT), while HEFT-LA narrows the gap but retains slightly higher comm vs PEFT.
- DLS and IHEFT coincide with HEFT on this dense instance, suggesting their ranking / selection heuristics didn't alter the critical path mapping under high contention.
- Load balance ratios mirror makespan differences; look-ahead variants slightly improve utilization.

## Reproduction
```powershell
python graphs/generate_dense120.py
python compare_same_dataset.py --dag graphs/dense120_task_connectivity.csv `
  --exec graphs/dense120_task_exe_time.csv --bw graphs/dense120_resource_BW.csv `
  --algos HEFT,PEFT,DLS,HEFT-LA,IHEFT --tag dense120 --report \
  --json_out graphs/dense120_results.json --out_csv graphs/dense120_results.csv
```

---
Generated automatically.
