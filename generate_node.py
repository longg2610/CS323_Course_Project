import random

num_nodes = 10
num_edges = 10  # must be >= num_nodes
max_weight = 10
min_weight = 1

nodes = [f"{i}" for i in range(1, num_nodes + 1)]

edges = []                # full edge list (with weights)
used_pairs = set()        # store only (u,v) pairs to avoid duplicates

# --- Step 1: Ensure each node has at least one edge ---
for u in nodes:
    v = random.choice(nodes)
    while v == u or tuple(sorted((u, v))) in used_pairs:
        v = random.choice(nodes)

    pair = tuple(sorted((u, v)))
    used_pairs.add(pair)

    w = random.randint(min_weight, max_weight)
    edges.append((u, v, w))

# --- Step 2: Add more edges up to num_edges ---
while len(edges) < num_edges:
    u, v = random.sample(nodes, 2)
    pair = tuple(sorted((u, v)))

    if pair in used_pairs:
        continue

    used_pairs.add(pair)

    w = random.randint(min_weight, max_weight)
    edges.append((u, v, w))

# --- Save to file ---
with open(f"./data/graph_{num_nodes}_{num_edges}.txt", "w") as f:
    for u, v, w in edges:
        f.write(f"{u} {v} {w}\n")

print(f"Generated graph with {num_nodes} nodes and {len(edges)} edges.")
