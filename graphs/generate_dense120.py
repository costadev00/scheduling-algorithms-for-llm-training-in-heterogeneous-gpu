import random, csv
random.seed(42)
NUM_TASKS=120
NUM_PROCS=4  # keep consistent with other datasets
EDGE_PROB=0.55  # high connectivity
MIN_COMM=5; MAX_COMM=80
MIN_EXEC=5; MAX_EXEC=160
MIN_BW=8; MAX_BW=25
# Build upper-triangular adjacency ensuring DAG
conn=[[0]*NUM_TASKS for _ in range(NUM_TASKS)]
for i in range(NUM_TASKS):
    for j in range(i+1, NUM_TASKS):
        if random.random() < EDGE_PROB:
            conn[i][j]=random.randint(MIN_COMM, MAX_COMM)
# Ensure every non-entry node has >=1 predecessor
for j in range(1, NUM_TASKS):
    preds=[i for i in range(j) if conn[i][j] != 0]
    if not preds:
        i=random.randrange(0,j)
        conn[i][j]=random.randint(MIN_COMM, MAX_COMM)
# Ensure every non-exit node has >=1 successor
for i in range(0, NUM_TASKS-1):
    succs=[j for j in range(i+1, NUM_TASKS) if conn[i][j] != 0]
    if not succs:
        j=random.randrange(i+1, NUM_TASKS)
        conn[i][j]=random.randint(MIN_COMM, MAX_COMM)
# Execution times matrix
exec_times=[[random.randint(MIN_EXEC, MAX_EXEC) for _ in range(NUM_PROCS)] for _ in range(NUM_TASKS)]
# Bandwidth matrix (symmetric)
bw=[[0]*NUM_PROCS for _ in range(NUM_PROCS)]
for i in range(NUM_PROCS):
    for j in range(i+1, NUM_PROCS):
        val=random.randint(MIN_BW, MAX_BW)
        bw[i][j]=bw[j][i]=val
# Write connectivity CSV
with open('graphs/dense120_task_connectivity.csv','w', newline='') as f:
    w=csv.writer(f); w.writerow(['T']+[f'T{k}' for k in range(NUM_TASKS)])
    for i,row in enumerate(conn): w.writerow([f'T{i}']+row)
# Write execution times
with open('graphs/dense120_task_exe_time.csv','w', newline='') as f:
    w=csv.writer(f); w.writerow(['TP']+[f'P_{p}' for p in range(NUM_PROCS)])
    for i,row in enumerate(exec_times): w.writerow([f'T_{i}']+row)
# Write bandwidth matrix
with open('graphs/dense120_resource_BW.csv','w', newline='') as f:
    w=csv.writer(f); w.writerow(['BP']+[f'P{p}' for p in range(NUM_PROCS)])
    for i,row in enumerate(bw): w.writerow([f'P{i}']+row)
print('Generated dense120 dataset.')
