"""
Error is maximum difference between an entry in the output matrix and that entry approximated value
R: mean of an exponential distribution, R = (n) / (A*epsilon*n)**(2-sqrt(2))
T = (n) / (A*epsilon*n)**(sqrt(2)-1)
A: bound of edge weight

** be careful of shallow copies **
"""

import math
import copy
import numpy as np

"""
JAMES
return the ball of radius r around vertex v in graph G. the ball is a set of vertices
G should be a graph with the following property: if there is an edge between u and v in the original graph,
G[u][v] = 1, otherwise G[u][v] = 0
"""
def ball(G, v, r):
    ball = set()


    return ball

"""
return graph G with vertices in V peeled off (G.V\V)
"""
def peel(G, V):
    n = len(G)
    subgraph = [[_ for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u in V or v in V:
                subgraph[u][v] = -1
            else:
                subgraph[u][v] = G[u][v]
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
                subgraph[u][v] = -1
            else:
                subgraph[u][v] = G[u][v]
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
            for v in range(n):
                mat_median[u][v] += mat[u][v]
    
    for u in range(n):
        for v in range(n):
            mat_median[u][v] /= L

    return mat_median

"""
magical function that returns shortest path distance from u to v in G. 
the path uses at most r red edges, b blue edges, and g green edges
"""
def constrained_shortest_path(G, u, v, r, b, g):

    return -1

"""
JAMES
return shortest path from u to v i.e. dist(u, v). (Dijsktra)
return -1 if v is not reachable from u
"""
def shortest_path(G, u, v):



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
                H[u][v] = 1
            else:               # G[u][v] == -1
                H[u][v] = 0
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

        # for each pair (u, v) ∈ E  (can be optimized by maintaining edge list)
        for u in range(n):
            for v in range(n):
                if G[u][v] != -1:
                    G_til[u][v] = G[u][v] + np.random.laplace(0.0, scale = (3*K)/epsilon)   # add a red edge to G_til, input perturbation
        
        # for all pairs u, v ∈ S (u = S[i], v = S[j]). Note: using indices is easier to enumerate pairs
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                dist_u_v = shortest_path(G, S[i], S[j])  # actual shortest path (use Dijkstra's or any SSSP algo)        
                if dist_u_v != -1:                # blue edge replaces sum of every edges from u to v. it will be as if there's one single edge from u to v
                    G_til[u][v] = dist_u_v + np.random.laplace(0.0, scale = (10*K*L*L)/epsilon) # add a blue edge to G_til, output-perturbation
        
        # for each B_t
        for t in range(len(balls)):
            B_t = balls[t]
            # for all pairs u, v ∈ B_t
            for i in range(len(B_t)):
                for j in range(i+1, len(B_t)):
                    u = B_t[i]
                    v = B_t[j]
                    M_t = medians[t]
                    G_til[u][v] = M_t[u][v]     # add a green edge to G_til

        output_apsp = [[_ for _ in range(n)] for _ in range(n)]
        # for all pairs u, v ∈ V 
        for u in range(n):
            for v in range(n):
                output_apsp[u][v] = constrained_shortest_path(G_til, u, v, (100*R*math.log(n)) + (100*T)/R, 1, (100*T)/R)   # dark magic
        iterations.append(output_apsp)
    
    return matrix_median(iterations, n)    # return median over K iterations
