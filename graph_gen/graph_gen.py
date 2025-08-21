#!/usr/bin/env python3
"""
Global graph generator for HEFT/PEFT.
- Reads a simple config file (same format as existing graph.config)
- Writes CSVs into an output folder with a chosen prefix
Usage examples (PowerShell):
  python graph_gen.py --config graph.config --out ..\graphs --prefix mygraph
"""
import numpy as np
import random as rndm
import copy
import csv
import argparse
import os

class graph_node:
    def __init__(self, task_id, level, out_count):
        self.outgoing_edge_count = out_count
        self.outgoing_edge_node = []
        self.outgoing_edge_weight = []
        self.level = level
        self.task_id = task_id
        self.resource_exe_time = []

def split_number(number, parts):
    count_per_part = []
    while len(count_per_part) == 0:
        for i in range(parts - 1):
            tmp1 = (number - sum(count_per_part)) / (parts - i)
            tmp2 = 0
            while tmp2 <= 0:
                tmp2 = int(rndm.normalvariate(tmp1, tmp1 / 2))
            count_per_part.extend([tmp2])
        count_per_part.extend([number - sum(count_per_part)])
        if min(count_per_part) <= 0:
            count_per_part = []
    return count_per_part

parser = argparse.ArgumentParser(description="Generate synthetic DAG and matrices for HEFT/PEFT")
parser.add_argument("--config", type=str, default="graph.config", help="Path to config file")
parser.add_argument("--out", type=str, default=".", help="Output directory for CSVs")
parser.add_argument("--prefix", type=str, default="graph", help="Filename prefix for generated CSVs")
args = parser.parse_args()

config_path = args.config
out_dir = os.path.abspath(args.out)
prefix = args.prefix
os.makedirs(out_dir, exist_ok=True)

# Defaults (overridden by config)
resource_count = 3
graph_height = 6
vertex_count = 20
mean_outdeg = 2
sd_outdeg = 1
comm_2_comp = 2.0
HF = 0.5
edge_weight_range = [1, 100]
bw_range = [10, 100]
SEED = 1000

with open(config_path, "r") as f:
    config = [(x.strip()).split() for x in f.readlines()]

for x in config:
    if len(x) == 0:
        continue
    if x[0] == "RC":
        resource_count = int(x[1])
    elif x[0] == "GH":
        graph_height = int(x[1])
    elif x[0] == "TC":
        vertex_count = int(x[1])
    elif x[0] == "AOD":
        mean_outdeg = int(x[1])
    elif x[0] == "CCR":
        comm_2_comp = float(x[1])
    elif x[0] == "HF":
        HF = float(x[1])
    elif x[0] == "CDR":
        edge_weight_range = [int(float(x[1])), int(float(x[2]))]
    elif x[0] == "LBW":
        bw_range = [int(float(x[1])), int(float(x[2]))]
    elif x[0] == "SEED":
        SEED = int(x[1])

if HF < 0 or HF > 1:
    raise SystemExit("0 <= (Heterogenity Factor(HF)) < 1")

np.random.seed(SEED)
rndm.seed(SEED)
print("SEED=", SEED)

if vertex_count < graph_height:
    raise SystemExit("Number of nodes are smaller than graph height")

resource_com_bw = np.zeros((resource_count, resource_count))
# set communication bandwidth among resources
for i in range(resource_count):
    for j in range(i + 1):
        if i == j:
            continue
        else:
            resource_com_bw[i][j] = rndm.randint(int(bw_range[0]), int(bw_range[1]))
            resource_com_bw[j][i] = resource_com_bw[i][j]

nodes_list = []
# start with one node in the graph
node_count_per_level = [1]
# number of nodes per level
node_count_per_level.extend(split_number(vertex_count - 2, graph_height - 2))
# end with one node in the graph
node_count_per_level.extend([1])

level_nodes_list = []
count = 0
# connect nodes in adjacent level
for level in range(len(node_count_per_level)):
    tmp1 = []
    elem = node_count_per_level[level]
    out_edge_count_to_next_level = []
    if level != len(node_count_per_level) - 1:
        if elem > node_count_per_level[level + 1]:
            out_edge_count_to_next_level = split_number(elem, elem)
        else:
            out_edge_count_to_next_level = split_number(node_count_per_level[level + 1], elem)
    else:
        out_edge_count_to_next_level = list(np.zeros((elem)))
    for i in range(elem):
        tmp1.append(graph_node(count, level, int(out_edge_count_to_next_level[i])))
        count += 1
    nodes_list.extend(tmp1)
    level_nodes_list.append(tmp1)

# make actual connections among the nodes in adjacent levels
for level in range(graph_height):
    l1 = []
    l0 = []
    for elem in nodes_list:
        if elem.level == level + 1:
            l1.append(elem.task_id)
        if elem.level == level:
            l0.append(elem.task_id)
    l1_tmp = copy.deepcopy(l1)
    for elem in l0:
        for _ in range(nodes_list[elem].outgoing_edge_count):
            tmp1 = rndm.choice(l1_tmp)
            nodes_list[elem].outgoing_edge_node.append(tmp1)
            nodes_list[elem].outgoing_edge_weight.append(rndm.randint(int(edge_weight_range[0]), int(edge_weight_range[1])))
            l1_tmp.remove(tmp1)
            if len(l1_tmp) == 0:
                l1_tmp = copy.deepcopy(l1)

# add more edges to connect multiple levels
for elem in nodes_list:
    if elem.level in (graph_height - 1, graph_height - 2, 0):
        continue
    l1 = [n.task_id for n in nodes_list if n.level > elem.level]
    for n in elem.outgoing_edge_node:
        if n in l1:
            l1.remove(n)
    remaining_nodes = len(l1)
    tmp1 = 0
    while (tmp1 <= 0) or (tmp1 > remaining_nodes):
        tmp1 = int(np.random.normal(mean_outdeg, sd_outdeg))
    if elem.outgoing_edge_count >= tmp1:
        continue
    new_nodes_to_connect = []
    for _ in range(tmp1 - elem.outgoing_edge_count):
        tmp2 = rndm.choice(l1)
        l1.remove(tmp2)
        new_nodes_to_connect.append(tmp2)
    elem.outgoing_edge_count += len(new_nodes_to_connect)
    for n in new_nodes_to_connect:
        elem.outgoing_edge_node.append(n)
        elem.outgoing_edge_weight.append(rndm.randint(1, 100))

link_bw = []
for i in range(len(resource_com_bw)):
    for j in range(len(resource_com_bw[i])):
        if i == j:
            break
        else:
            link_bw.append(resource_com_bw[i][j])

# assign computation time to each node on different resources
for elem in nodes_list:
    max_weight = -1
    if elem.outgoing_edge_count == 0:
        average_com_time = float(rndm.randint(int(edge_weight_range[0]), int(edge_weight_range[1]))) / float(
            rndm.randint(int(bw_range[0]), int(bw_range[1]))
        )
    else:
        for tmp in elem.outgoing_edge_weight:
            if tmp > max_weight:
                max_weight = tmp
        com_time = [float(max_weight) / float(bw) for bw in link_bw]
        average_com_time = sum(com_time) / float(len(com_time))
    mean_comp_time = float(average_com_time) / float(comm_2_comp)
    exe_time = []
    if HF == 0:
        exe_time = [mean_comp_time for _ in range(resource_count)]
    else:
        lb = mean_comp_time - (HF * mean_comp_time)
        ub = mean_comp_time + (HF * mean_comp_time)
        exe_time = np.random.uniform(lb, ub, resource_count)
    elem.resource_exe_time = exe_time

# build connectivity matrix
connect_matrix = np.zeros((len(nodes_list), len(nodes_list)))
for elem in nodes_list:
    for idx in range(len(elem.outgoing_edge_node)):
        connect_matrix[elem.task_id][elem.outgoing_edge_node[idx]] = elem.outgoing_edge_weight[idx]

# write resource_BW.csv
rows = [["P"] + [f"P_{i}" for i in range(resource_count)]]
for i in range(resource_count):
    row = [f"P_{i}"] + [resource_com_bw[i][j] for j in range(resource_count)]
    rows.append(row)
with open(os.path.join(out_dir, f"{prefix}_resource_BW.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# write task_connectivity.csv
rows = [["T"] + [f"T_{i}" for i in range(len(connect_matrix))]]
for i in range(len(connect_matrix)):
    row = [f"T_{i}"] + [connect_matrix[i][j] for j in range(len(connect_matrix))]
    rows.append(row)
with open(os.path.join(out_dir, f"{prefix}_task_connectivity.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# write task_exe_time.csv
rows = [["TP"] + [f"P_{i}" for i in range(resource_count)]]
for i in range(len(nodes_list)):
    row = [f"T_{i}"] + [nodes_list[i].resource_exe_time[j] for j in range(resource_count)]
    rows.append(row)
with open(os.path.join(out_dir, f"{prefix}_task_exe_time.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
