import os, csv, math
from heft.heft.heft import (
    readCsvToNumpyMatrix as heft_read_matrix,
    readDagMatrix as heft_read_dag,
    readCsvToDict as heft_read_power,
    schedule_dag as heft_schedule,
    _compute_makespan_and_idle as heft_makespan_idle,
    _compute_load_balance as heft_load_balance,
    _compute_communication_cost as heft_comm_cost,
    _compute_waiting_time as heft_wait_time,
)
from peft.peft.peft import (
    readCsvToNumpyMatrix as peft_read_matrix,
    readDagMatrix as peft_read_dag,
    readCsvToDict as peft_read_power,
    schedule_dag as peft_schedule,
    _compute_makespan_and_idle as peft_makespan_idle,
    _compute_load_balance as peft_load_balance,
    _compute_communication_cost as peft_comm_cost,
    _compute_waiting_time as peft_wait_time,
)

BASE = os.path.join('graphs', 'scenario 2')
PREFIXES = [
    ('scen2_rc2', 2),
    ('scen2_rc4', 4),
    ('scen2_rc8', 8),
]

out_rows = [[
    'algorithm','RC','makespan','load_balance_ratio','communication_cost','waiting_time','energy_cost'
]]

for prefix, rc in PREFIXES:
    dag_file = os.path.join(BASE, f'{prefix}_task_connectivity.csv')
    bw_file = os.path.join(BASE, f'{prefix}_resource_BW.csv')
    exe_file = os.path.join(BASE, f'{prefix}_task_exe_time.csv')
    power_file = os.path.join(BASE, f'{prefix}_task_power.csv')
    if not all(os.path.exists(p) for p in [dag_file,bw_file,exe_file,power_file]):
        continue

    # HEFT
    comm_matrix_heft = heft_read_matrix(bw_file)
    if comm_matrix_heft.shape[0] != comm_matrix_heft.shape[1]:
        startup = comm_matrix_heft[-1,:]
        comm_matrix_core = comm_matrix_heft[0:-1,:]
    else:
        startup = [0]*comm_matrix_heft.shape[0]
        comm_matrix_core = comm_matrix_heft
    comp_matrix_heft = heft_read_matrix(exe_file)
    dag_heft = heft_read_dag(dag_file, show_dag=False)
    power_dict_heft = heft_read_power(power_file)
    proc_sched_heft, _, _ = heft_schedule(
        dag_heft,
        communication_matrix=comm_matrix_core,
        communication_startup=startup,
        computation_matrix=comp_matrix_heft,
        power_dict=power_dict_heft,
    )
    makespan_heft, _, _ = heft_makespan_idle(proc_sched_heft)
    per_proc_busy_heft, _, _, _ = heft_load_balance(proc_sched_heft)
    avg_busy_heft = (sum(per_proc_busy_heft.values())/len(per_proc_busy_heft)) if per_proc_busy_heft else math.inf
    lb_ratio_heft = makespan_heft / avg_busy_heft if avg_busy_heft>0 else math.inf
    comm_cost_heft = heft_comm_cost(dag_heft, proc_sched_heft, comm_matrix_core, startup)
    wait_heft = heft_wait_time(proc_sched_heft)
    energy_heft = 0.0
    for proc, jobs in proc_sched_heft.items():
        for j in jobs:
            dur = float(j.end)-float(j.start)
            if dur>0:
                try:
                    pw = float(power_dict_heft[j.task][j.proc])
                except Exception:
                    pw = 0.0
                energy_heft += dur*pw
    out_rows.append(['HEFT', rc, makespan_heft, lb_ratio_heft, comm_cost_heft, wait_heft, energy_heft])

    # PEFT
    comm_matrix_peft = peft_read_matrix(bw_file)
    comp_matrix_peft = peft_read_matrix(exe_file)
    dag_peft = peft_read_dag(dag_file, show_dag=False)
    power_dict_peft = peft_read_power(power_file)
    proc_sched_peft, _, _ = peft_schedule(
        dag_peft,
        communication_matrix=comm_matrix_peft,
        computation_matrix=comp_matrix_peft,
        power_dict=power_dict_peft,
    )
    makespan_peft, _, _ = peft_makespan_idle(proc_sched_peft)
    per_proc_busy_peft, _, _, _ = peft_load_balance(proc_sched_peft)
    avg_busy_peft = (sum(per_proc_busy_peft.values())/len(per_proc_busy_peft)) if per_proc_busy_peft else math.inf
    lb_ratio_peft = makespan_peft / avg_busy_peft if avg_busy_peft>0 else math.inf
    comm_cost_peft = peft_comm_cost(dag_peft, proc_sched_peft, comm_matrix_peft)
    wait_peft = peft_wait_time(proc_sched_peft)
    energy_peft = 0.0
    for proc, jobs in proc_sched_peft.items():
        for j in jobs:
            dur = float(j.end)-float(j.start)
            if dur>0:
                try:
                    pw = float(power_dict_peft[j.task][j.proc])
                except Exception:
                    pw = 0.0
                energy_peft += dur*pw
    out_rows.append(['PEFT', rc, makespan_peft, lb_ratio_peft, comm_cost_peft, wait_peft, energy_peft])

out_csv = os.path.join(BASE, 'scenario2_summary.csv')
with open(out_csv, 'w', newline='') as f:
    csv.writer(f).writerows(out_rows)

print('Summary written to', out_csv)
for r in out_rows:
    print(', '.join(map(str,r)))
