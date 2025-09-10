from __future__ import annotations
import json
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
COMPARE = ROOT / 'compare_same_dataset.py'
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    # (scenario_folder, tasks)
    ('Scenario_1', 256),
    ('Scenario_2', 512),
    ('Scenario_3', 1024),
    ('Scenario_4', 2048),
    ('Scenario_5', 4096),
]
PROCS = [8, 16, 32]
ALGOS = 'DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT'

rows = []
jsonl_path = OUT_DIR / 'aggregate_metrics.jsonl'
csv_path = OUT_DIR / 'aggregate_metrics.csv'

with jsonl_path.open('w', encoding='utf-8') as jf:
    for scen, tasks in SCENARIOS:
        for p in PROCS:
            prefix = ROOT / 'graphs' / scen / f'peft{tasks}_{p}proc'
            dag = f'{prefix}_task_connectivity.csv'
            exe = f'{prefix}_task_exe_time.csv'
            bw  = f'{prefix}_resource_BW.csv'
            powf= f'{prefix}_task_power.csv'
            # If files are missing, skip gracefully
            if not (Path(dag).exists() and Path(exe).exists() and Path(bw).exists() and Path(powf).exists()):
                print(f'[skip] Missing inputs for {scen} {tasks}x{p}')
                continue
            cmd = [
                'python', str(COMPARE),
                '--dag', dag,
                '--exec', exe,
                '--bw', bw,
                '--power', powf,
                '--algos', ALGOS,
            ]
            print(f'Running: {scen} {tasks}x{p} ...')
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print('Error running compare_same_dataset.py:', proc.stderr)
                continue
            # Parse JSON lines from stdout
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec_out = {
                    'scenario': scen,
                    'tasks': tasks,
                    'procs': p,
                    'algorithm': rec.get('algorithm'),
                    'makespan': rec.get('makespan'),
                    'load_balance_ratio': rec.get('load_balance_ratio'),
                    'communication_cost': rec.get('communication_cost'),
                    'waiting_time': rec.get('waiting_time'),
                    'energy_cost': rec.get('energy_cost'),
                }
                jf.write(json.dumps(rec_out) + '\n')
                rows.append(rec_out)

# Write CSV
fieldnames = ['scenario','tasks','procs','algorithm','makespan','load_balance_ratio','communication_cost','waiting_time','energy_cost']
with csv_path.open('w', newline='', encoding='utf-8') as cf:
    w = csv.DictWriter(cf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f'Wrote {len(rows)} rows to {jsonl_path} and {csv_path}')
