import random

INF = float('inf')

def generate_graph(n, A, edge_prob):
    # Parameters: runtime changes with n/edge_bound proportion
    # Generate adjacency matrix with random weights and missing edges
    G = [[INF for _ in range(n)] for _ in range(n)]

    for u in range(n):
        for v in range(n):
            if u != v:
                if random.random() < edge_prob:  # chance of edge existing
                    G[u][v] = G[v][u] = random.randint(1, A)
                else:
                    G[u][v] = G[v][u] = INF

    return G

# num_vertices = 16
# max_weight = 200
# edge_probability = 0.90

# 10, 15, 20, 25, 30
# A = 200
# edge_prob = 0.2
# epsilon = 0.1

# A = 100, 500, 1000, 2000, 3000
# n = 25
# edge_prob = 0.6
# epsilon = 0.1

num_vertices = 25
max_weight = 3000
edge_probability = 0.6


graph = generate_graph(num_vertices, max_weight, edge_probability)

# Dump the graph adj matrix to a text file for easy loading later
with open(f"data/graph_{num_vertices}_{max_weight}_{edge_probability}.txt", "w") as f:
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] == INF:
                f.write("INF ")
            else:
                f.write(f"{graph[i][j]} ")
        f.write("\n")
