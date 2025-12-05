import random

num_nodes = 300
num_edges = 100  # you can adjust the density
max_weight = 100
min_weight = 1

# Generate node names
nodes = [f"N{i}" for i in range(1, num_nodes + 1)]

# Generate random edges
edges = []
for _ in range(num_edges):
    n1, n2 = random.sample(nodes, 2)  # choose 2 distinct nodes
    weight = random.randint(min_weight, max_weight)   # random weight between 1 and 100
    edges.append((n1, n2, weight))

# Save to a text file
with open(f"./data/graph_{num_nodes}_{num_edges}.txt", "w") as f:
    for u, v, w in edges:
        f.write(f"{u} {v} {w}\n")

print(f"Generated graph with {num_nodes} nodes and {len(edges)} edges.")
