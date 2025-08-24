import csv, os
import matplotlib.pyplot as plt

BASE = os.path.join('graphs','scenario 3')
SUMMARY = os.path.join(BASE, 'scenario3_summary.csv')

if not os.path.exists(SUMMARY):
    raise SystemExit('Summary CSV not found: ' + SUMMARY)

rc_vals = {'HEFT': [], 'PEFT': []}
metrics = ['makespan','energy_cost','communication_cost','waiting_time','load_balance_ratio']
values = {m: {'HEFT': [], 'PEFT': []} for m in metrics}

with open(SUMMARY) as f:
    r = csv.DictReader(f)
    for row in r:
        alg = row['algorithm']
        rc = int(float(row['RC']))
        rc_vals[alg].append(rc)
        for m in metrics:
            values[m][alg].append(float(row[m]))

# Order by RC
for alg in rc_vals:
    order = sorted(range(len(rc_vals[alg])), key=lambda i: rc_vals[alg][i])
    rc_vals[alg] = [rc_vals[alg][i] for i in order]
    for m in metrics:
        values[m][alg] = [values[m][alg][i] for i in order]

plot_specs = [
    ('makespan','Makespan','Scenario 3 (300 tasks): Makespan vs RC','scenario3_makespan_vs_rc.png'),
    ('energy_cost','Energy Cost','Scenario 3 (300 tasks): Energy vs RC','scenario3_energy_vs_rc.png'),
    ('communication_cost','Communication Cost','Scenario 3 (300 tasks): Communication Cost vs RC','scenario3_comm_vs_rc.png'),
    ('waiting_time','Average Waiting Time','Scenario 3 (300 tasks): Waiting Time vs RC','scenario3_waiting_vs_rc.png'),
    ('load_balance_ratio','Load Balance Ratio','Scenario 3 (300 tasks): Load Balance Ratio vs RC','scenario3_loadbalance_vs_rc.png'),
]

for key, ylabel, title, fname in plot_specs:
    plt.figure(figsize=(5,3.2))
    plt.plot(rc_vals['HEFT'], values[key]['HEFT'], marker='o', label='HEFT')
    plt.plot(rc_vals['PEFT'], values[key]['PEFT'], marker='s', label='PEFT')
    plt.xlabel('Processor Count (RC)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE, fname)
    plt.savefig(out_path, dpi=150)

print('Plots written to scenario3_*_vs_rc.png')
