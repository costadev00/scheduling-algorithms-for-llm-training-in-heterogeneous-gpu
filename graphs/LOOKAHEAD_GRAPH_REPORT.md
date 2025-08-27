# Lookahead Graph Scheduling Report

Tag: `lookahead_demo`

## Dataset
- Connectivity: `graphs/lookahead_graph_task_connectivity.csv`
- Execution Times: `graphs/lookahead_graph_task_exe_time.csv`
- Bandwidth: `graphs/lookahead_graph_resource_BW.csv`
- Tasks: 10 (T0 is high fan-out root to T1..T5, which then feed into a converging chain T6..T9)
- Processors: 4 (P0..P3)

Design intent: Root T0 has moderately heterogeneous runtimes across processors, but its successors (T1..T5) each have markedly faster execution on processor P2 or P1 compared to others, encouraging a placement of T0 that minimizes downstream cross-processor communication when successors choose their optimal processors. HEFT may pick a processor for T0 based only on its own EFT, while HEFT-LA/PEFT consider child effects.

## Results
| Algorithm | Makespan | Load Balance Ratio | Communication Cost | Avg Waiting Time |
|-----------|----------|--------------------|--------------------|------------------|
| HEFT      | 94.5     | 2.2909             | 49.50              | 20.3833          |
| HEFT-LA   | 93.5     | 2.2530             | 23.83              | 20.0500          |
| PEFT      | 93.5     | 2.2530             | 23.83              | 20.0500          |
| DLS (DL1) | 94.5     | 2.2909             | 49.50              | 20.3833          |

(All energy fields null: no power matrix provided.)

## Observations
- HEFT-LA and PEFT reduce communication cost by ~52% compared to HEFT/DLS (49.5 -> 23.83) by co-locating or better aligning the high fan-out root with its successors' preferred processors.
- Makespan improvement is modest (1 time unit) because the DAG's critical path lies mostly in the later chain (T6..T9); however, lowered communication shrinks idle gaps.
- DLS behaves similarly to HEFT here because its dynamic level scoring does not explicitly incorporate future successor placement beyond current EST and relative speed terms.
- PEFT matches HEFT-LA on this instance since the OCT-based lookahead implicitly captures similar downstream readiness advantages.

## Takeaways
A single-step child EFT lookahead (HEFT-LA) can close the gap to more involved OCT reasoning (PEFT) on structured fan-out cases with heterogeneous successor preferences, at minimal extra computational cost.

## Reproduction
```powershell
python compare_same_dataset.py \` 
  --dag graphs/lookahead_graph_task_connectivity.csv \` 
  --exec graphs/lookahead_graph_task_exe_time.csv \` 
  --bw graphs/lookahead_graph_resource_BW.csv \` 
  --algos HEFT,HEFT-LA,PEFT,DLS --report --tag lookahead_demo
```

Optional JSON export:
```powershell
python compare_same_dataset.py --dag graphs/lookahead_graph_task_connectivity.csv \
  --exec graphs/lookahead_graph_task_exe_time.csv --bw graphs/lookahead_graph_resource_BW.csv \
  --algos HEFT,HEFT-LA,PEFT,DLS --report --tag lookahead_demo \
  --json_out graphs/lookahead_graph_results.json
```

---
Generated automatically.
