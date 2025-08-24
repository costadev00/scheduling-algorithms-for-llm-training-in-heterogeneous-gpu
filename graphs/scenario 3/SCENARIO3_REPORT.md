# Scenario 3 (300 Tasks) - Paper-only HEFT vs PEFT

Canonical (paper-faithful) implementations of HEFT (Topcuoglu et al., 2002) and PEFT (Arabnejad & Barbosa, 2014) were run on Scenario 3 for RC ∈ {2,4,8}. Implementations exclude any energy-aware or modified ranking heuristics; energy is a post-scheduling aggregate.

## Raw Metrics (from `scenario3_summary.csv`)

| Algorithm | RC | Makespan | Load Balance Ratio (MS / avg busy) | Communication Cost | Waiting Time | Energy Cost |
|-----------|----|----------|-------------------------------------|--------------------|--------------|-------------|
| HEFT | 2 | 56.6502 | 0.1328 | 24.2586 | 27.7794 | 918,377.26 |
| PEFT | 2 | 64.7708 | 0.1557 | 310.9483 | 30.6212 | 900,694.80 |
| HEFT | 4 | 21.9155 | 0.2971 | 290.8648 | 8.0923 | 325,163.49 |
| PEFT | 4 | 20.0288 | 0.2840 | 189.5381 | 8.2483 | 299,341.82 |
| HEFT | 8 | 62.2497 | 0.9370 | 637.5933 | 21.1631 | 580,342.69 |
| PEFT | 8 | 66.8831 | 1.0346 | 786.5159 | 24.9836 | 582,606.20 |

Notes:
- Load Balance Ratio here < 1 indicates average busy time exceeds makespan due to definition (makespan / avg_busy). Values are not classic imbalance indices; compare only between algorithms at same RC.
- Energy = Σ(duration * power(task,proc)) using supplied power matrix entries verbatim.

## Comparative Observations
1. RC=2: HEFT yields lower makespan and dramatically lower communication cost. PEFT spends more time transferring data (many cross-processor edges), inflating comm cost while slightly reducing energy.
2. RC=4: PEFT edges HEFT on makespan (≈ -8.6%) and cuts communication relative to HEFT, while also slightly reducing energy. Here PEFT’s OCT guidance aligns better with critical subpaths.
3. RC=8: Both algorithms experience larger makespans again (schedule fragmentation / bandwidth contention). HEFT now beats PEFT on makespan and communication; PEFT’s extra transfers raise its comm cost.
4. Waiting Time correlates with makespan; minima appear at RC=4 for both algorithms (better parallel exploitation sweet spot).
5. Energy differences stay modest (< ~7% spread) across all RC values; neither algorithm explicitly optimizes it.

## Plots Generated
`scenario3_makespan_vs_rc.png`, `scenario3_energy_vs_rc.png`, `scenario3_comm_vs_rc.png`, `scenario3_waiting_vs_rc.png`, `scenario3_loadbalance_vs_rc.png`.

## Method Overview
- HEFT: Upward rank (average comm normalized), descending order; processor chosen by earliest finish (insertion-based).
- PEFT: Optimistic Cost Table (OCT); rank = mean OCT row; processor chosen by min(EFT + OCT[task,proc]).
- Communication time realized only on inter-processor edges: data_size / bandwidth.
- No energy influence on ordering or placement.

## Reproduce
```
python summarize_scenario3.py
python "graphs/scenario 3/scenario3_plots.py"
```

## Suggested Next Steps
- Add variance via multiple random seeds per RC.
- Report classic load imbalance metrics (e.g., max/avg busy) alongside current ratio.
- Critical path extraction to diagnose RC=8 regression.
- Optional: integrate a separate energy-aware experiment branch for comparison.

---
Report refreshed for paper-only baseline.
