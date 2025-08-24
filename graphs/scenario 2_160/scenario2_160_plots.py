import csv, os
import matplotlib.pyplot as plt

BASE = os.path.join('graphs','scenario 2_160')
SUMMARY = os.path.join(BASE, 'scenario2_160_summary.csv')

rc_vals = {'HEFT': [], 'PEFT': []}
makespan = {'HEFT': [], 'PEFT': []}
energy = {'HEFT': [], 'PEFT': []}
comm_cost = {'HEFT': [], 'PEFT': []}
waiting = {'HEFT': [], 'PEFT': []}
load_balance = {'HEFT': [], 'PEFT': []}

with open(SUMMARY) as f:
    r = csv.DictReader(f)
    for row in r:
        alg = row['algorithm']
        rc = int(float(row['RC']))
        rc_vals[alg].append(rc)
        makespan[alg].append(float(row['makespan']))
        energy[alg].append(float(row['energy_cost']))
        comm_cost[alg].append(float(row['communication_cost']))
        waiting[alg].append(float(row['waiting_time']))
        load_balance[alg].append(float(row['load_balance_ratio']))

for alg in rc_vals:
    order = sorted(range(len(rc_vals[alg])), key=lambda i: rc_vals[alg][i])
    rc_vals[alg] = [rc_vals[alg][i] for i in order]
    makespan[alg] = [makespan[alg][i] for i in order]
    energy[alg] = [energy[alg][i] for i in order]
    comm_cost[alg] = [comm_cost[alg][i] for i in order]
    waiting[alg] = [waiting[alg][i] for i in order]
    load_balance[alg] = [load_balance[alg][i] for i in order]

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], makespan['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], makespan['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Makespan')
plt.title('Scenario 2 (160 tasks): Makespan vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_160_makespan_vs_rc.png'), dpi=150)

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], energy['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], energy['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Energy Cost')
plt.title('Scenario 2 (160 tasks): Energy vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_160_energy_vs_rc.png'), dpi=150)

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], comm_cost['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], comm_cost['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Communication Cost')
plt.title('Scenario 2 (160 tasks): Communication Cost vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_160_comm_vs_rc.png'), dpi=150)

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], waiting['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], waiting['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Average Waiting Time')
plt.title('Scenario 2 (160 tasks): Waiting Time vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_160_waiting_vs_rc.png'), dpi=150)

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], load_balance['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], load_balance['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Load Balance Ratio')
plt.title('Scenario 2 (160 tasks): Load Balance Ratio vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_160_loadbalance_vs_rc.png'), dpi=150)

print('Plots written to scenario2_160_*_vs_rc.png')
