import random, csv, numpy as np
random.seed(42)
NUM_TASKS=50
NUM_PROCS=4
EDGE_PROB=0.18  # sparsity control
MIN_COMM=5; MAX_COMM=50
MIN_EXEC=5; MAX_EXEC=120
MIN_BW=5; MAX_BW=20
# Connectivity (upper triangular DAG)
conn=[[0]*NUM_TASKS for _ in range(NUM_TASKS)]
for i in range(NUM_TASKS):
    for j in range(i+1, NUM_TASKS):
        if random.random() < EDGE_PROB:
            conn[i][j]=random.randint(MIN_COMM, MAX_COMM)
# Ensure each non-entry node has at least one predecessor (if empty, connect from random earlier node)
for j in range(1, NUM_TASKS):
    if all(conn[i][j]==0 for i in range(j)):
        i=random.randrange(0,j)
        conn[i][j]=random.randint(MIN_COMM, MAX_COMM)
# Execution times
exec_times=[[random.randint(MIN_EXEC, MAX_EXEC) for _ in range(NUM_PROCS)] for _ in range(NUM_TASKS)]
# Bandwidth matrix symmetric
bw=[[0]*NUM_PROCS for _ in range(NUM_PROCS)]
for i in range(NUM_PROCS):
    for j in range(i+1, NUM_PROCS):
        val=random.randint(MIN_BW, MAX_BW)
        bw[i][j]=bw[j][i]=val
# Write connectivity CSV
with open('graphs/random50_task_connectivity.csv','w', newline='') as f:
    w=csv.writer(f)
    header=['T']+[f'T{k}' for k in range(NUM_TASKS)]
    w.writerow(header)
    for i,row in enumerate(conn):
        w.writerow([f'T{i}']+row)
# Write execution times CSV
with open('graphs/random50_task_exe_time.csv','w', newline='') as f:
    w=csv.writer(f)
    w.writerow(['TP']+[f'P_{p}' for p in range(NUM_PROCS)])
    for i,row in enumerate(exec_times):
        w.writerow([f'T_{i}']+row)
# Write bandwidth CSV
with open('graphs/random50_resource_BW.csv','w', newline='') as f:
    w=csv.writer(f)
    w.writerow(['BP']+[f'P{p}' for p in range(NUM_PROCS)])
    for i,row in enumerate(bw):
        w.writerow([f'P{i}']+row)
print('Generated random50 dataset.')
