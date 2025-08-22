import csv, os
import matplotlib.pyplot as plt

BASE = os.path.join('graphs','scenario 1')
SUMMARY = os.path.join(BASE, 'scenario1_summary.csv')

rc_vals = {'HEFT': [], 'PEFT': []}
makespan = {'HEFT': [], 'PEFT': []}
energy = {'HEFT': [], 'PEFT': []}

with open(SUMMARY) as f:
    r = csv.DictReader(f)
    for row in r:
        alg = row['algorithm']
        rc = int(float(row['RC']))
        ms = float(row['makespan'])
        en = float(row['energy_cost'])
        rc_vals[alg].append(rc)
        makespan[alg].append(ms)
        energy[alg].append(en)

# Sort by RC
for alg in rc_vals:
    order = sorted(range(len(rc_vals[alg])), key=lambda i: rc_vals[alg][i])
    rc_vals[alg] = [rc_vals[alg][i] for i in order]
    makespan[alg] = [makespan[alg][i] for i in order]
    energy[alg] = [energy[alg][i] for i in order]

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], makespan['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], makespan['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Makespan')
plt.title('Scenario 1: Makespan vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario1_makespan_vs_rc.png'), dpi=150)

plt.figure(figsize=(5,3.2))
plt.plot(rc_vals['HEFT'], energy['HEFT'], marker='o', label='HEFT')
plt.plot(rc_vals['PEFT'], energy['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Energy Cost')
plt.title('Scenario 1: Energy vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario1_energy_vs_rc.png'), dpi=150)

print('Plots written to scenario1_makespan_vs_rc.png and scenario1_energy_vs_rc.png')
