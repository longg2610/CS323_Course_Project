"""
Error is maximum difference between an entry in the output matrix and that entry approximated value
R: mean of an exponential distribution, R = (n) / (A*epsilon*n)**(2-sqrt(2))
T = (n) / (A*epsilon*n)**(sqrt(2)-1)
A: bound of edge weight
todo: handle G[u][v] = G[v][u], need to be assigned in the same line and not reasssigned
** be careful of shallow copies **
"""

import math
import copy
import numpy as np
import random
import heapq

random.seed(36)

"""
JAMES
return the ball of radius r around vertex v in graph G. the ball is a set of vertices
G should be a graph with the following property: if there is an edge between u and v in the original graph,
G[u][v] = 1, otherwise G[u][v] = 0
"""
def ball(G, v, r):
    ball = set()
    
    # BFS to find all vertices within distance r from v
    n = len(G) 
    queue = [(v, 0)]  # (current_vertex, current_distance)
    visited = [False] * n
    visited[v] = True
    
    while queue:
        current, dist = queue.pop(0)
        ball.add(current)
        
        if dist < r:
            for neighbor in range(n):
                if G[current][neighbor] == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
    
    return ball

def add_edge(G, u, v, value):
    G[u][v] = value
    G[v][u] = value

"""
return graph G with vertices in V peeled off (G.V-V)
"""
def peel(G, V):
    n = len(G)
    subgraph = [[_ for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u in V or v in V:
                add_edge(subgraph, u, v, -1)
            else:
                add_edge(subgraph, u, v, G[u][v])
    return subgraph


"""
return the subgraph of G with only vertices in SET V. note that V has to be subset of G.V
pruning vertices by setting their entries to -1
"""
def subgraph(G, V):
    n = len(G)
    pruned_set = []
    for i in range(n):
        if i not in V:
            pruned_set.append(i)

    subgraph = [[_ for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u in pruned_set or v in pruned_set:
                add_edge(subgraph, u, v, -1)
            else:
                add_edge(subgraph, u, v, G[u][v])
    return subgraph

"""
return the matrix containing entrywise median of all matrices in the input
n is size of each matrix
"""
def matrix_median(matrices, n):
    L = len(matrices)   # number of matrices
    mat_median = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(L):
        mat = matrices[i]
        for u in range(n):
            for v in range(u+1, n):
                add_edge(mat_median, u, v, mat_median[u][v] + mat[u][v])
    
    for u in range(n):
        for v in range(n):
            add_edge(mat_median, u, v, mat_median[u][v] / L)

    return mat_median


"""
recursive DFS used to enumerate all paths
"""
def dfs(G, u, dst, visited, path, paths, edge_color, r, b, g, rbound, bbound, gbound):
    if r > rbound or b > bbound or g > gbound:      
        return

    if u == dst:
        paths.append(path.copy())
        return

    n = len(G)
    visited[u] = True
    for v in range(n):
        if G[u][v] != -1 and not visited[v]:
            if edge_color[u,v] == "red":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r + 1, b, g, rbound, bbound, gbound)
            elif edge_color[u,v] == "blue":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r, b + 1, g, rbound, bbound, gbound)
            elif edge_color[u,v] == "green":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r, b, g + 1, rbound, bbound, gbound)
            
    visited[u] = False  # backtrack
    return

"""
compute total distance of a path in graph G
"""
def compute_dist(G, path):
    dist = 0
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        dist += G[u][v]
    return dist

"""
magical function that returns shortest path distance from src to dst in G. 
the path uses at most rbound red edges, bbound blue edges, and gbound green edges
"""
def constrained_shortest_path(G, edge_color, src, dst, rbound, bbound, gbound):
    n = len(G)
    visited = [False] * n
    paths = []

    # get all paths that match constraint
    dfs(G, src, dst, visited, [src], paths, edge_color, 0, 0, 0, rbound, bbound, gbound)

    # find the shortest one
    shortest_dist = float('inf')
    shortest_path = []
    for path in paths:
        dist = compute_dist(G, path)
        if dist <= shortest_dist:
            shortest_dist = dist
            shortest_path = path
    return shortest_dist, shortest_path
    
"""
JAMES
return shortest path from u to v i.e. dist(u, v). (Dijsktra)
return -1 if v is not reachable from u
"""
def shortest_path(G, u, v):
    
    n = len(G)
    INF = float('inf')

    dist = [INF] * n
    dist[u] = 0

    pq = [(0, u)]  # (distance, node)

    while pq:
        d, node = heapq.heappop(pq)

        # Early stop
        if node == v:
            return d

        if d > dist[node]:
            continue

        # Explore neighbors
        for j in range(n):
            w = G[node][j]

            if w == -1:     # no edge
                continue

            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))

    # v unreachable
    return -1


"""
return H, the unweighted graph topology of G
if there is an edge between u and v in G (including ones with 0 weight), the corresponding edge in H is 1
if there is no edge between u and v in G (G[u][v] == -1), the corresponding edge in H is 0
"""
def get_topology(G):
    n = len(G)
    H = [[_ for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if G[u][v] >= 0:
                add_edge(H, u, v, 1)
            else:               # G[u][v] == -1
                add_edge(H, u, v, 0)
    return H

"""
output apsp matrix for graph G
G: matrix reprentation of the graph. G[u][v] is weight of the edge connecting vertices u and v. 
G[u][v] = -1 if there's no edge between u and v
"""
def bounded_weights(G, epsilon):
    V = list(range(len(G)))
    n = len(V)       # number of vertices
    A = max(map(max, G))        # maximum edge weight i.e. bound of weights

    K = 100 * math.log(n)
    T = n / ((A * epsilon * n) ** ((math.sqrt(17) - 3)/4))
    R = n / ((A * epsilon * n) ** ((5 - math.sqrt(17))/2))
    L = K * (n/T)

    H = get_topology(G)    # H is G but unweighted. H is public

    iterations = []     # contains result of each of the k iterations, used to find entrywise median to improve success probability
    for k in range(1, K + 1):
        H_prime = copy.deepcopy(H)
        t = 0

        while True:
            v_t = -1
            for i in range(n):    # for each vertex in H_prime
                if len(ball(H_prime, i, 100*R* math.log(n))) <= T:       # if the size of the ball around i less than T
                    v_t = i                                                 # take v_t
                    break

            if v_t == -1:   # could not find satisfying v_t
                break
                
            r_t = np.random.exponential(scale=1/R)      # r_t = Exp(R)

            balls = []
            B_t = ball(H_prime, v_t, r_t)       # set of vertices that are within distance r_t of v_t in H′
            balls.append(B_t)

            H_prime = peel(H_prime, B_t)        # Peel off ball around v_t
            t += 1
        
        medians = []            # list of M_t's
        for B_t in balls:
            matrices = []       # len(matrices) = K
            for l in range(1, K + 1):
                M_t_l = bounded_weights(subgraph(G, B_t), epsilon / (3*(K**2)))
                matrices.append(M_t_l)
            M_t = matrix_median(matrices, n)
            medians.append(M_t)

        S = np.random.choice(V, size=L, replace=False)  # Construct hitting set
        G_til = [[-1 for _ in range(n)] for _ in range(n)]      # Create a weighted multigraph ˜G with vertex set V and initially no edges

        edge_color = {}
        # for each pair (u, v) ∈ E  (can be optimized by maintaining edge list)
        for u in range(n):
            for v in range(u+1, n):
                if G[u][v] != -1:
                    noise = np.random.laplace(0.0, scale = (3*K)/epsilon)
                    add_edge(G_til, u, v, G[u][v] + noise)  # add a red edge to G_til, input perturbation
                    edge_color[u,v] = edge_color[v,u] = "red"
        
        # for all pairs u, v ∈ S (u = S[i], v = S[j]). Note: using indices is easier to enumerate pairs
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                dist_u_v = shortest_path(G, S[i], S[j])  # actual shortest path (use Dijkstra's or any SSSP algo)   
                # blue edge replaces sum of every edges from u to v. it will be as if there's one single edge from u to v     
                if dist_u_v != -1:     
                    noise = np.random.laplace(0.0, scale = (10*K*L*L)/epsilon)   
                    add_edge(G_til, u, v, dist_u_v + noise) # add a blue edge to G_til, output-perturbation
                    edge_color[u,v] = edge_color[v,u] = "blue"
        
        # for each B_t
        for t in range(len(balls)):
            B_t = balls[t]
            # for all pairs u, v ∈ B_t
            for i in range(len(B_t)):
                for j in range(i+1, len(B_t)):
                    u = B_t[i]
                    v = B_t[j]
                    M_t = medians[t]
                    add_edge(G_til, u, v, M_t[u][v])    # add a green edge to G_til
                    edge_color[u,v] = edge_color[v,u] = "green"

        output_apsp = [[_ for _ in range(n)] for _ in range(n)]
        # for all pairs u, v ∈ V 
        for u in range(n):
            for v in range(n):
                add_edge(output_apsp, u, v, constrained_shortest_path(G_til, edge_color, u, v, (100*R*math.log(n)) + (100*T)/R, 1, (100*T)/R)) # dark magic
        iterations.append(output_apsp)
    
    return matrix_median(iterations, n)    # return median over K iterations



"""
TESTING
"""

# Parameters
n = 5  # number of vertices
colors = ["red", "blue", "green"]

# Generate adjacency matrix with random weights and missing edges
G = [[-1 for _ in range(n)] for _ in range(n)]
edge_color = {}

for u in range(n):
    for v in range(n):
        if u != v and (u,v) not in edge_color and (v,u) not in edge_color:
            if random.random() < 0.9:  # 90% chance of edge existing
                G[u][v] = G[v][u] = random.randint(1, 10)
                color = random.choice(colors)
                edge_color[(u, v)] = edge_color[(v, u)] = color
            else:
                G[u][v] = G[v][u] = -1

# Print graph
print("Adjacency matrix (weights):")
for row in G:
    print(row)

print("\nEdge colors:")
for (u, v), c in edge_color.items():
    print(f"({u} -> {v}): {c}")

print(constrained_shortest_path(G, edge_color, 0, 4, 0, 0, 2))   # r, b, g
