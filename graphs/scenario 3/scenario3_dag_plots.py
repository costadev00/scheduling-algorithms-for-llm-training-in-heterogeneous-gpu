import os, sys
import networkx as nx
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path so 'heft' / 'peft' packages import without installation.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from heft.heft.heft import readDagMatrix

BASE = os.path.join('graphs','scenario 3')
PREFIXES = ['scen3_rc2','scen3_rc4','scen3_rc8']

for px in PREFIXES:
    dag_path = os.path.join(BASE, f'{px}_task_connectivity.csv')
    if not os.path.exists(dag_path):
        print('Missing', dag_path)
        continue
    dag = readDagMatrix(dag_path, show_dag=False)
    try:
        pos = nx.nx_pydot.graphviz_layout(dag, prog='dot')
        layout_used = 'dot'
    except Exception as e:
        pos = nx.spring_layout(dag, seed=42)
        layout_used = f'spring (fallback: {e})'
    plt.figure(figsize=(8,10))
    nx.draw(dag, pos=pos, node_size=300, font_size=6, with_labels=True, arrowsize=8)
    plt.title(f'Scenario 3 DAG {px} (layout={layout_used})')
    out_file = os.path.join(BASE, f'{px}_dag.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print('Wrote', out_file)

print('Done generating DAG images.')
