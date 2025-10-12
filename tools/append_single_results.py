"""Append compare_same_dataset.py JSON output for one scenario/proc into outputs aggregate files."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMPARE = ROOT / 'compare_same_dataset.py'
OUT_JSONL = ROOT / 'outputs' / 'aggregate_metrics.jsonl'
OUT_CSV = ROOT / 'outputs' / 'aggregate_metrics.csv'

def run_and_append(dag, exe, bw, power, scen, tasks, procs):
    cmd = ['python', str(COMPARE), '--dag', dag, '--exec', exe, '--bw', bw, '--power', power, '--algos', 'DLS,HEFT,HEFT-LA,PEFT,IHEFT,IPEFT']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print('Error:', proc.stderr)
        return 0
    rows = []
    for line in proc.stdout.splitlines():
        line=line.strip()
        if not line or not line.startswith('{'): continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out = {
            'scenario': scen,
            'tasks': tasks,
            'procs': procs,
            'algorithm': rec.get('algorithm'),
            'makespan': rec.get('makespan'),
            'load_balance_ratio': rec.get('load_balance_ratio'),
            'communication_cost': rec.get('communication_cost'),
            'waiting_time': rec.get('waiting_time'),
            'energy_cost': rec.get('energy_cost'),
        }
        rows.append(out)
        with OUT_JSONL.open('a', encoding='utf-8') as jf:
            jf.write(json.dumps(out) + '\n')
    # Append to CSV by simple write (assume header exists)
    if rows:
        import csv
        with OUT_CSV.open('a', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            for r in rows:
                w.writerow([r['scenario'], r['tasks'], r['procs'], r['algorithm'], r['makespan'], r['load_balance_ratio'], r['communication_cost'], r['waiting_time'], r['energy_cost']])
    return len(rows)

def main():
    # args: scen tasks procs prefix
    if len(sys.argv)!=5:
        print('usage: append_single_results.py <scenario> <tasks> <procs> <prefix>')
        sys.exit(1)
    scen, tasks_s, procs_s, prefix = sys.argv[1:5]
    tasks=int(tasks_s); procs=int(procs_s)
    dag = f'{prefix}_task_connectivity.csv'
    exe = f'{prefix}_task_exe_time.csv'
    bw  = f'{prefix}_resource_BW.csv'
    power = f'{prefix}_task_power.csv'
    cnt = run_and_append(dag, exe, bw, power, scen, tasks, procs)
    print(f'Appended {cnt} rows for {scen} {tasks}x{procs}')

if __name__=='__main__':
    main()
