import numpy as np
import ot
import edlib
import networkx as nx
import matplotlib.pyplot as plt

# 仮想のコンティグを生成
contigs = [
    "ACTG", 
]
# assembly result:
# CTGACTGAC?

# edit distance を計算する関数
def edit_distance(seq1, seq2):
    result = edlib.align(seq1, seq2, task="distance")
    return result["editDistance"]

# 配列アラインメントを計算する関数。match, mismatch とか.
def alignment(seq1, seq2, params):
    # Using biopython
    from Bio import pairwise2
    result = pairwise2.align.globalms(seq1, seq2, params["match"], params["mismatch"], params["gap_open"], params["gap_extend"])
    return result

# コスト行列の作成
n = len(contigs)
cost_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            cost_matrix[i, j] = edit_distance(contigs[i], contigs[j])

print(f"Cost matrix:\n{cost_matrix}")


# Gromov-Wasserstein距離の計算
p = np.ones(n) / n
q = np.ones(n) / n
gw_dist, log = ot.gromov.gromov_wasserstein(cost_matrix, cost_matrix, p, q, 'square_loss', log=True)
print(f"Gromov-Wasserstein distance: {gw_dist}")
print(f"Optimal transport plan:\n{log}")
# 最適なマッチングの取得
optimal_transport_plan = gw_dist

# グラフの作成
G = nx.Graph()

# ノードの追加
for i, contig in enumerate(contigs):
    G.add_node(i, label=contig)

# エッジの追加
for i in range(n):
    for j in range(n):
        if optimal_transport_plan[i, j] > 0:
            G.add_edge(i, j, weight=optimal_transport_plan[i, j], label=f"{cost_matrix[i, j]:.2f}")

# ノードラベルとエッジラベルの設定
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')
edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges}

# グラフの描画
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Genome Assembly Visualization with Gromov-Wasserstein Distance")
plt.savefig("/large/otgk/OT_assembly/figures/GW_GenomeAssembly.png")