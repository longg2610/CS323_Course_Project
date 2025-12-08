"""
James Bui, Long Pham
December 2025
floyd_warshall.py
This file contains the implementation of Floyd-Warshall Algorithm on a weighted graph.
"""


def floyd_warshall(graph):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j
        
    Returns:
        distances: 2D list of shortest distances between all pairs
        paths: 2D list where paths[i][j] contains the sequence of vertices in the shortest path from i to j
    """
    
    num_vertices = len(graph)
    
    # Initialize distances and paths
    distances = [row[:] for row in graph]  # Copy of the distance matrix
    
    # Floyd-Warshall algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                # if distances[i][k] == -1 or distances[k][j] == -1:
                #     continue
                
                # if  distances[i][j] == -1:
                #     distances[i][j] = distances[i][k] + distances[k][j]
                    
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    return distances






