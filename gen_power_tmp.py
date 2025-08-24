import csv,random,os
base = r"graphs\scenario 2_160"
random.seed(43210)
for rc in [2,4,8]:
    prefix=f"scen2_160_rc{rc}"
    exe=os.path.join(base,f"{prefix}_task_exe_time.csv")
    if not os.path.exists(exe): continue
    with open(exe) as f: rows=list(csv.reader(f))
    tasks=[r[0] for r in rows[1:]]
    out=[['TP']+[f'P_{i}' for i in range(rc)]]
    for t in tasks:
        vals=[round(random.uniform(1,60)*(1 if random.random()<0.85 else random.uniform(1.5,2.2)),2) for _ in range(rc)]
        out.append([t]+vals)
    with open(os.path.join(base,f"{prefix}_task_power.csv"),'w',newline='') as f:
        csv.writer(f).writerows(out)
print("Power files written.")
