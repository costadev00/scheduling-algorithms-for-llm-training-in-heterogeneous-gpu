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
| IPEFT     | 3171.449 | 3.03270            | 8767.41            | 1538.858         |

(Values rounded; energy not computed.)

## Observations
- PEFT and HEFT-LA obtain the best makespan (≈6.07 units, ~0.19% faster than HEFT baseline). IPEFT lands in between (improves makespan vs HEFT by ~3.80 units) on this instance.
- PEFT preserves its advantage in lowest communication cost (≈ -2.1% vs HEFT). HEFT-LA trades a slight comm increase over PEFT (+0.94%) for equal makespan. IPEFT sits between PEFT and HEFT-LA on comm cost.
- IPEFT’s blended PCT/CNCT ranking alters a small portion of placements (reflected in its intermediate makespan & comm profile) without matching PEFT’s best-case continuation guidance here.
- DLS, HEFT, and IHEFT coincide (same schedule) indicating their ranking differences collapse under this dense high-connectivity structure.
- Load balance ratios track makespan: lower makespan variants modestly reduce imbalance; differences remain minor (<0.25% absolute).

## Reproduction
```powershell
python graphs/generate_dense120.py
python compare_same_dataset.py --dag graphs/dense120_task_connectivity.csv `
  --exec graphs/dense120_task_exe_time.csv --bw graphs/dense120_resource_BW.csv `
  --algos HEFT,PEFT,DLS,HEFT-LA,IHEFT,IPEFT --tag dense120 --report \
  --json_out graphs/dense120_results.json --out_csv graphs/dense120_results.csv
```

---
Generated automatically.
