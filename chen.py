"""
Written by Long Pham & James Bui
Private All Pairs Shortest Path by Chen et. al
"""

import math
import copy
import numpy as np
import heapq

INF = float('inf')

"""
add edges symmetrically to graph
"""
def add_edge(G, u, v, value):
    G[u][v] = value
    G[v][u] = value

"""
return the matrix containing entrywise median of all matrices in the input
n is size of each matrix
"""
def matrix_median(matrices, n):
    L = len(matrices)   # number of matrices
    mat_median = [[0 for _ in range(n)] for _ in range(n)]
    arr = np.array(matrices, dtype=float)   # shape (L, n, n)

    for u in range(n):
        for v in range(u+1, n):
            # extract the L values for (u, v)
            vals = arr[:, u, v]

            # ignore both infinity and -1
            # mask = np.isfinite(vals) & (vals != -1)
            mask = np.isfinite(vals)
            finite_vals = vals[mask]

            if len(finite_vals) == 0:
                med = INF
            else:
                med = float(np.median(finite_vals))

            mat_median[u][v] = mat_median[v][u] = med

    return mat_median


"""
return H, the unweighted graph topology of G
if there is an edge between u and v in G (including ones with 0 weight), the corresponding edge in H is 1
if there is no edge between u and v in G (G[u][v] == INF), the corresponding edge in H is 0
"""
def get_topology(G):
    n = len(G)
    H = [[0 for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if G[u][v] != INF:
                add_edge(H, u, v, 1)
            else:      
                add_edge(H, u, v, 0)
    return H

"""
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


"""
return graph G with vertices in V peeled off (G.V-V). only used in context of 
a topology representation graph i.e. 1 if there's an edge, 0 otherwise
"""
def peel(G, V):
    assert isinstance(V, set), "V must be a set"
    n = len(G)
    peeled = [[0 for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if u in V or v in V:
                add_edge(peeled, u, v, 0)
            else:
                add_edge(peeled, u, v, G[u][v])
    return peeled


"""
return the subgraph of G with only vertices in SET V. note that V has to be subset of G.V
pruning vertices by setting their entries to INF
"""
def subgraph(G, V):
    assert isinstance(V, set), "V must be a set"

    n = len(G)
    pruned_set = []
    for i in range(n):
        if i not in V:
            pruned_set.append(i)

    subgraph = [[INF for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if u in pruned_set or v in pruned_set:
                add_edge(subgraph, u, v, INF)
            else:
                add_edge(subgraph, u, v, G[u][v])
    return subgraph


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
        if G[u][v] != INF and not visited[v]:
            if edge_color[u,v] == "red":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r + 1, b, g, rbound, bbound, gbound)
            elif edge_color[u,v] == "blue":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r, b + 1, g, rbound, bbound, gbound)
            elif edge_color[u,v] == "green":
                dfs(G, v, dst, visited, path + [v], paths, edge_color, r, b, g + 1, rbound, bbound, gbound)
            
    visited[u] = False  # backtrack
    return


"""
returns shortest path distance from src to dst in G. 
the path uses at most rbound red edges, bbound blue edges, and gbound green edges
"""
def constrained_shortest_path(G, edge_color, src, dst, rbound, bbound, gbound):
    n = len(G)
    visited = [False] * n
    paths = []

    # get all paths that match constraint
    dfs(G, src, dst, visited, [src], paths, edge_color, 0, 0, 0, rbound, bbound, gbound)

    # find the shortest one
    shortest_dist = INF
    shortest_path = []
    for path in paths:
        dist = compute_dist(G, path)
        if dist <= shortest_dist:
            shortest_dist = dist
            shortest_path = path
    return shortest_dist, shortest_path
    
"""
return shortest path from u to v i.e. dist(u, v). (Dijsktra)
return INF if v is not reachable from u
"""
def shortest_path(G, u, v):
    
    n = len(G)

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

            if w == INF:     # no edge
                continue

            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))

    # v unreachable
    return INF


"""
output apsp matrix for graph G
G: matrix reprentation of the graph. G[u][v] is weight of the edge connecting vertices u and v. 
G[u][v] = INF if there's no edge between u and v
"""
def bounded_weights(G, V, A, epsilon):

    assert isinstance(V, set), "V must be a set"

    n = len(G)       # number of vertices in the original graph 
    V_len = len(V)      

    if len(V) == 0:
        raise Exception("Vertex set should not be empty")

    if len(V) == 1:         # BASE CASE, return INF distances
        return [[INF for _ in range(n)] for _ in range(n)]

    # K = math.floor(100 * math.log(V_len, 10))     # actual value used in the paper
    K = math.floor(1 * math.log(V_len, 10))
    T = V_len / ((A * epsilon * V_len) ** ((math.sqrt(17) - 3)/4))
    R = V_len / ((A * epsilon * V_len) ** ((5 - math.sqrt(17))/2))
    # L = math.floor(100 * math.log(n, 10) * ((A * epsilon * V_len) ** ((math.sqrt(17) - 3)/4)))        # actual value used in the paper
    L = math.floor(1 * math.log(n, 10) * ((A * epsilon * V_len) ** ((math.sqrt(17) - 3)/4)))

    H = get_topology(G)    # H is G but unweighted. H (the graph architecture) is public

    iterations = []     # contains result of each of the k iterations, used to find entrywise median to improve success probability
    for k in range(1, K+1):
        H_prime = copy.deepcopy(H)
        t = 0

        balls = []
        for v_t in V:    # for each vertex in H_prime
            ball_radius = len(ball(H_prime, v_t, 1*R* math.log(V_len, 10)))
            # if the size of the ball around v_t less than T and the ball is nonempty (H_prime needs to shrink), take v_t
            if ball_radius == 0:    # ball_radius == 0 should never happen because will at least contain v_t
                raise Exception("Ball should never be empty")
            if ball_radius > V_len:         
                raise Exception("Ball should not be larger than the graph itself")
            if ball_radius > T:  
                continue
                
            r_t = np.random.exponential(scale=1/R)      # r_t = Exp(R)
            B_t = ball(H_prime, v_t, r_t)       # set of vertices that are within distance r_t of v_t in H′
            
            if len(B_t) == 0:              
                raise Exception("Ball should never be empty")   
            if len(B_t) > V_len:       
                raise Exception("Ball should not be larger than the graph itself")

            # if the ball is everything, all nodes are peeled off and the ball is appended to balls
            # and processed in the next recursion depth WITH THE SAME SET OF NODES
            # thus the ball to be peeled off must be strictly smaller than the set of vertices
            if len(B_t) > 1 and len(B_t) == V_len:
                continue
                                      
            balls.append(B_t)
            H_prime = peel(H_prime, B_t)        # Peel off ball around v_t
            t += 1

        medians = []            # list of M_t's
        for B_t in balls:
            matrices = []       # len(matrices) = K
            for l in range(1, K + 1):
                subgr = subgraph(G, B_t)
                M_t_l = bounded_weights(subgr, B_t, A, epsilon / (3*((1 * math.log(V_len, 10))**2)))        # recursively run on smaller balls
                matrices.append(M_t_l)

            M_t = matrix_median(matrices, n)   # entrywise median of M_t_l over K     
            medians.append(M_t)

        if L > len(V):
            raise Exception(f"Sample size larger than the graph: L needs {L}, V has {len(V)}")
        
        S = np.random.choice(list(V), size=L, replace=False)  # Construct hitting set
        G_til = [[INF for _ in range(n)] for _ in range(n)]      # Create a weighted multigraph ˜G with vertex set V and initially no edges

        edge_color = {}
        # for each pair (u, v) ∈ E 
        V_list = list(V)
        for i in range(V_len):
            for j in range(i+1, V_len):
                u = V_list[i]
                v = V_list[j]
                if G[u][v] != INF:
                    noise = np.random.laplace(0.0, scale = (3*K)/epsilon)
                    add_edge(G_til, u, v, G[u][v] + noise)  # add a red edge to G_til, input perturbation
                    edge_color[u,v] = edge_color[v,u] = "red"
        
        # for all pairs u, v ∈ S
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                u = S[i]
                v = S[j]
                dist_u_v = shortest_path(G, u, v)  # actual shortest path (use Dijkstra's or any SSSP algo), INF if unreachable   
                # blue edge replaces sum of every edges from u to v. it will be as if there's one single edge from u to v     
                if dist_u_v != INF:   
                    noise = np.random.laplace(0.0, scale = (10*K*L*L)/epsilon)   
                    add_edge(G_til, u, v, dist_u_v + noise) # add a blue edge to G_til, output-perturbation
                    edge_color[u,v] = edge_color[v,u] = "blue"
        
        # for each B_t
        for t in range(len(balls)):
            B_t = list(balls[t])
            # for all pairs u, v ∈ B_t
            for i in range(len(B_t)):
                for j in range(i+1, len(B_t)):
                    u = B_t[i]
                    v = B_t[j]
                    M_t = medians[t]
                    add_edge(G_til, u, v, M_t[u][v])    # add a green edge to G_til
                    edge_color[u,v] = edge_color[v,u] = "green"

        output_apsp = [[INF for _ in range(n)] for _ in range(n)]      # INF means unreacheable

        # for all pairs u, v ∈ V 
        V_list = list(V)
        for i in range(V_len):
            for j in range(i + 1, V_len):
                u = V_list[i]
                v = V_list[j]
                add_edge(output_apsp, u, v, constrained_shortest_path(G_til, edge_color, u, v, 
                                                            math.floor(1*(R*math.log(n, 10) + T/R)), 1, math.floor(1*(T/R))) [0]) 
        iterations.append(output_apsp)
    return matrix_median(iterations, n)    # return median over K iterations
