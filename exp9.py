import networkx as nx
import matplotlib.pyplot as plt

graph = {'User1': ['ProductA', 'ProductB'],
         'User2': ['ProductB', 'ProductC'],
         'User3': ['ProductC', 'ProductA'],
         'User4': ['ProductA', 'ProductD'],
         'User5': ['ProductD', 'ProductB']}

G = nx.DiGraph([(u, v) for u, links in graph.items() for v in links])
nx_pagerank_scores = nx.pagerank(G)

print("\nPageRank Scores:")
for k, v in sorted(nx_pagerank_scores.items(), key=lambda x: -x[1]):
    print(f"{k}: {v:.4f}")

nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight="bold", arrows=True)
plt.title("User-Product Interaction Graph (PageRank)")
plt.show()
