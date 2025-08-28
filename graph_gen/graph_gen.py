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
    """Split an integer 'number' into 'parts' positive integers.
    Original version used repeated Gaussian sampling with potential heavy retry loops
    when parts is large (e.g., equal to number). This optimized version provides:
      - Fast path when number == parts: all ones.
      - Fast path when parts == 1.
      - Dirichlet-like random partition using random cut points (O(parts log parts)).
    Guarantees all parts > 0 and sum == number.
    """
    if parts <= 0:
        return []
    if parts == 1:
        return [number]
    if number == parts:
        return [1] * parts
    if parts > number:  # can't have all positive ints otherwise
        # Fallback: allocate 1 to each of number parts, zeros elsewhere (should not happen in current usage)
        return [1]*number + [0]*(parts-number)
    # Use random cut points technique
    # Choose (parts-1) distinct cut positions in [1, number-1]
    cuts = sorted(rndm.sample(range(1, number), parts-1))
    prev = 0
    out = []
    for c in cuts:
        out.append(c - prev)
        prev = c
    out.append(number - prev)
    # All entries guaranteed positive
    return out

parser = argparse.ArgumentParser(description="Generate synthetic DAG and matrices for HEFT/PEFT")
parser.add_argument("--config", type=str, default="graph.config", help="Path to config file")
parser.add_argument("--out", type=str, default=".", help="Output directory for CSVs")
parser.add_argument("--prefix", type=str, default="graph", help="Filename prefix for generated CSVs (no scenario expansion)")
# Optional overrides (if omitted, config values used)
parser.add_argument("--RC", type=int, help="Override resource count")
parser.add_argument("--GH", type=int, help="Override graph height")
parser.add_argument("--TC", type=int, help="Override total task (vertex) count")
parser.add_argument("--AOD", type=int, help="Override average out-degree (mean additional edges)")
parser.add_argument("--CCR", type=float, help="Override communication-to-computation ratio (higher => relatively cheaper compute)")
parser.add_argument("--HF", type=float, help="Override heterogeneity factor [0,1]")
parser.add_argument("--CDR_LOW", type=int, help="Override communication data size range low bound")
parser.add_argument("--CDR_HIGH", type=int, help="Override communication data size range high bound")
parser.add_argument("--LBW_LOW", type=int, help="Override link bandwidth range low bound")
parser.add_argument("--LBW_HIGH", type=int, help="Override link bandwidth range high bound")
parser.add_argument("--SEED", type=int, help="Override base seed")
parser.add_argument("--repeat", type=int, default=1, help="[DEPRECATED] Previously generated multiple scenarios. Ignored; always 1.")
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

"""Apply CLI overrides (if provided) AFTER config values loaded"""
if args.RC is not None: resource_count = args.RC
if args.GH is not None: graph_height = args.GH
if args.TC is not None: vertex_count = args.TC
if args.AOD is not None: mean_outdeg = args.AOD
if args.CCR is not None: comm_2_comp = args.CCR
if args.HF is not None: HF = args.HF
if args.CDR_LOW is not None: edge_weight_range[0] = int(args.CDR_LOW)
if args.CDR_HIGH is not None: edge_weight_range[1] = int(args.CDR_HIGH)
if args.LBW_LOW is not None: bw_range[0] = int(args.LBW_LOW)
if args.LBW_HIGH is not None: bw_range[1] = int(args.LBW_HIGH)
if args.SEED is not None: SEED = args.SEED

if HF < 0 or HF > 1:
    raise SystemExit("0 <= (Heterogeneity Factor(HF)) < 1")

def generate_one(seed, px):
    np.random.seed(seed)
    rndm.seed(seed)
    print("SEED=", seed, "PREFIX=", px)

    if vertex_count < graph_height:
        raise SystemExit("Number of nodes are smaller than graph height")

    resource_com_bw = np.zeros((resource_count, resource_count))
    for i in range(resource_count):
        for j in range(i + 1):
            if i == j:
                continue
            resource_com_bw[i][j] = rndm.randint(int(bw_range[0]), int(bw_range[1]))
            resource_com_bw[j][i] = resource_com_bw[i][j]

    nodes_list = []
    node_count_per_level = [1]
    node_count_per_level.extend(split_number(vertex_count - 2, graph_height - 2))
    node_count_per_level.extend([1])

    level_nodes_list = []
    count = 0
    for level in range(len(node_count_per_level)):
        tmp1 = []
        elem = node_count_per_level[level]
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
            link_bw.append(resource_com_bw[i][j])

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
        if HF == 0:
            exe_time = [mean_comp_time for _ in range(resource_count)]
        else:
            lb = mean_comp_time - (HF * mean_comp_time)
            ub = mean_comp_time + (HF * mean_comp_time)
            exe_time = np.random.uniform(lb, ub, resource_count)
        elem.resource_exe_time = exe_time

    connect_matrix = np.zeros((len(nodes_list), len(nodes_list)))
    for elem in nodes_list:
        for idx in range(len(elem.outgoing_edge_node)):
            connect_matrix[elem.task_id][elem.outgoing_edge_node[idx]] = elem.outgoing_edge_weight[idx]

    # Write outputs
    rows = [["P"] + [f"P_{i}" for i in range(resource_count)]]
    for i in range(resource_count):
        row = [f"P_{i}"] + [resource_com_bw[i][j] for j in range(resource_count)]
        rows.append(row)
    with open(os.path.join(out_dir, f"{px}_resource_BW.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    rows = [["T"] + [f"T_{i}" for i in range(len(connect_matrix))]]
    for i in range(len(connect_matrix)):
        row = [f"T_{i}"] + [connect_matrix[i][j] for j in range(len(connect_matrix))]
        rows.append(row)
    with open(os.path.join(out_dir, f"{px}_task_connectivity.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    rows = [["TP"] + [f"P_{i}" for i in range(resource_count)]]
    for i in range(len(nodes_list)):
        row = [f"T_{i}"] + [nodes_list[i].resource_exe_time[j] for j in range(resource_count)]
        rows.append(row)
    with open(os.path.join(out_dir, f"{px}_task_exe_time.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

# Deprecated multi-scenario logic removed. Always generate a single set of files.
if args.repeat and args.repeat > 1:
    print(f"[WARN] --repeat={args.repeat} requested but multi-scenario generation is disabled. Generating a single graph only.")

# Sanitize any legacy {i} placeholder in prefix
if "{i}" in prefix:
    print("[INFO] Removing '{i}' placeholder from prefix for single generation mode.")
    prefix = prefix.replace("{i}", "")

generate_one(SEED, prefix)
