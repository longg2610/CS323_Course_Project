"""
James Bui, Long Pham
December 2025
dp_floyd_warshall.py
This file contains the implementation of a simple Differentially Private Floyd-Warshall Algorithm on a weighted graph.
"""

import numpy as np


# Input pertubation - add noise to each edge weight

def dp_floyd_warshall_input_pertubation(graph, eps = 1.0):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j
        
    Returns:
        distances: 2D list of shortest distances between all pairs
    """
    
    num_vertices = len(graph) 
    
    # Initialize distances and paths
    distances = [row[:] for row in graph]  # Copy of the distance matrix
    
    for i in range(num_vertices):
        for j in range(num_vertices):
            if distances[i][j] != -1 and i != j:
                noise = np.random.laplace(0, 1/eps)
                distances[i][j] += noise
    
    # Floyd-Warshall algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distances[i][k] == -1 or distances[k][j] == -1:
                    continue
                
                if  distances[i][j] == -1:
                    distances[i][j] = distances[i][k] + distances[k][j]
                    
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    
    
    return distances


# Output pertubation - add noise to each entry in the distance matrix at the end

def dp_floyd_warshall_output_pertubation(graph, eps = 1.0):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j
        
    Returns:
        distances: 2D list of shortest distances between all pairs
    """
    
    num_vertices = len(graph) 
    
    # Initialize distances and paths
    distances = [row[:] for row in graph]  # Copy of the distance matrix
    
    # Floyd-Warshall algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distances[i][k] == -1 or distances[k][j] == -1:
                    continue
                
                if  distances[i][j] == -1:
                    distances[i][j] = distances[i][k] + distances[k][j]
                
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    for i in range(num_vertices):
        for j in range(num_vertices):
            if distances[i][j] != -1 and i != j:
                noise = np.random.laplace(0, 1/eps)
                distances[i][j] += noise
    
    
    return distances