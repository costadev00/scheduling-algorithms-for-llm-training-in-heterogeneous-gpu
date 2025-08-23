import csv, os
import matplotlib.pyplot as plt
BASE = os.path.join('graphs','scenario 2')
SUMMARY = os.path.join(BASE,'scenario2_summary.csv')
rc={'HEFT':[], 'PEFT':[]}
ms={'HEFT':[], 'PEFT':[]}
energy={'HEFT':[], 'PEFT':[]}
with open(SUMMARY) as f:
    r=csv.DictReader(f)
    for row in r:
        alg=row['algorithm']
        rc_val=int(float(row['RC']))
        rc[alg].append(rc_val)
        ms[alg].append(float(row['makespan']))
        energy[alg].append(float(row['energy_cost']))
for alg in rc:
    order=sorted(range(len(rc[alg])), key=lambda i: rc[alg][i])
    rc[alg]=[rc[alg][i] for i in order]
    ms[alg]=[ms[alg][i] for i in order]
    energy[alg]=[energy[alg][i] for i in order]
plt.figure(figsize=(5,3.2))
plt.plot(rc['HEFT'], ms['HEFT'], marker='o', label='HEFT')
plt.plot(rc['PEFT'], ms['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Makespan')
plt.title('Scenario 2: Makespan vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_makespan_vs_rc.png'), dpi=150)
plt.figure(figsize=(5,3.2))
plt.plot(rc['HEFT'], energy['HEFT'], marker='o', label='HEFT')
plt.plot(rc['PEFT'], energy['PEFT'], marker='s', label='PEFT')
plt.xlabel('Processor Count (RC)')
plt.ylabel('Energy Cost')
plt.title('Scenario 2: Energy vs RC')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE,'scenario2_energy_vs_rc.png'), dpi=150)
print('Scenario 2 plots generated')
