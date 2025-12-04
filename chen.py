"""
Error is maximum difference between an entry in the output matrix and that entry approximated value
R: mean of an exponential distribution, R = (n) / (A*epsilon*n)**(2-sqrt(2))
T = (n) / (A*epsilon*n)**(sqrt(2)-1)
A: bound of edge weight


Need:
- Function to get the ball of a vertex given a radius B(v, r)
- Function to compute entrywise median of K matrices
"""


import math
import numpy as np

def ball(G, v, r):
    return

"""
return graph A with nodes in B peeled off (A\B)
"""
def peel(A, B):
    return

"""
return the subgraph of G with only vertices in set V. note that V has to be subset of G.V
"""
def subgraph(G, V):
    return

"""
return the entrywise median of all matrices in the input
"""
def matrix_median(matrices):
    return

"""
return shortest path from u to v i.e. dist(u, v)
"""
def shortest_path(G, u, v):
    return

def bounded_weights(G, epsilon):
    V = list(range(len(G[0])))
    n = len(V)       # number of vertices
    A = max(map(max, G))        # maximum edge weight i.e. bound of weights

    K = 100 * math.log(n)
    T = n / ((A * epsilon * n) ** ((math.sqrt(17) - 3)/4))
    R = n / ((A * epsilon * n) ** ((5 - math.sqrt(17))/2))
    L = K * (n/T)

    H = G # H is G but unweighted

    for k in range(1, K + 1):
        H_prime = H
        t = 0

        while True:
            v_t = -1
            for i in range(len(H_prime[0])):    # for each vertex in H_prime
                if len(ball(H_prime, v_t, 100 * R * math.log(n))) <= T:       # if the size of the ball around v_t less than T
                    v_t = i                                                 # take v_t
                    break

            if v_t == -1:   # could not find satisfying v_t
                break
                
            r_t = np.random.exponential(scale=1/R)      # r_t = Exp(R)

            balls = []
            B_t = ball(H_prime, v_t, r_t)       # set of vertices that are within distance r_t of v_t in H′
            balls.append(B_t)

            H_prime = peel(H_prime, B_t)
            t += 1
        

        for B_t in balls:
            matrices = []       # len(matrices) = K
            for l in range(1, K + 1):
                M_t_l = bounded_weights(subgraph(G, B_t), epsilon / (3*(K**2)))
                matrices.append(M_t_l)
            M_t = matrix_median(matrices)

        S = np.random.choice(V, size=L, replace=False)  # Construct hitting set
        G_til = [[-1 for _ in range(n)] for _ in range(n)]      # Create a weighted multigraph ˜G with vertex set V and initially no edges

        # for each pair (u, v) ∈ E
        for u in range(n):
            for v in range(n):
                if G[u][v] != -1:
                    G_til[u][v] = G[u][v] + np.random.laplace(0.0, scale = (3*K)/epsilon)   # add a red edge to G_til
        
        # for all pairs u, v ∈ S
        for u in range(len(S)):
            for v in range(u+1, len(S)):
                dist_u_v = shortest_path(G, u, v)
                if dist_u_v != -1:
                    G_til[u][v] = dist_u_v + np.random.laplace(0.0, scale = (10*K*L*L)/epsilon) # add a blue edge to G_til
        
        # for each B_t
        

