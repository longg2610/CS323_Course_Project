"""
James Bui, Long Pham
December 2025
dp_floyd_warshall.py
This file contains the implementation of a simple Differentially Private Floyd-Warshall Algorithm on a weighted graph.
"""

import numpy as np
import math

INF = float("inf")


def dp_floyd_warshall_input_perturbation(graph, eps = 1.0):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm,
    with input perturbation (noise added to each edge).
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j,
               or INF if no direct edge.
    Returns:
        distances: 2D list of shortest distances between all pairs
    """
    num_vertices = len(graph)
    
    # Initialize distances and paths with copy
    distances = [row[:] for row in graph]
    
    # sensitivity = 3 * (math.log(num_vertices, 10))        # sensitivity comparable to Chen et al. (3K)
    sensitivity = 1
    # Add noise to each existing edge (not INF), excluding self-loops
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and distances[i][j] != INF:
                noise = np.random.laplace(0, sensitivity/eps)
                distances[i][j] += noise
    
    # Floyd-Warshall algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distances[i][k] == INF or distances[k][j] == INF:
                    continue
                new_d = distances[i][k] + distances[k][j]
                if distances[i][j] == INF or new_d < distances[i][j]:
                    distances[i][j] = new_d
    
    return distances


def dp_floyd_warshall_output_perturbation(graph, eps = 1.0):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm,
    then adds noise to each final distance entry (output perturbation).
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j,
               or INF if no direct edge.
    Returns:
        distances: 2D list of shortest distances between all pairs (with noise added),
                   INF if no path exists between i and j.
    """
    num_vertices = len(graph)
    
    distances = [row[:] for row in graph]
    
    # sensitivity = 3 * (math.log(num_vertices, 10))        # sensitivity comparable to Chen et al. (3K)
    sensitivity = 1
    # Floyd-Warshall
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distances[i][k] == INF or distances[k][j] == INF:
                    continue
                new_d = distances[i][k] + distances[k][j]
                if distances[i][j] == INF or new_d < distances[i][j]:
                    distances[i][j] = new_d
    
    # Add noise to each finite distance (excluding self-loops, optionally)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and distances[i][j] != INF:
                noise = np.random.laplace(0, sensitivity/eps)
                distances[i][j] += noise
    
    return distances
