"""
James Bui, Long Pham
December 2025
floyd_warshall.py
This file contains the implementation of Floyd-Warshall Algorithm on a weighted graph.
"""

INF = float("inf")

def floyd_warshall(graph):
    """
    Computes shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
    
    Args:
        graph: Distance matrix (2D list) where graph[i][j] is the distance from i to j,
               or INF if no direct edge.
        
    Returns:
        distances: 2D list of shortest distances between all pairs
    """
    
    num_vertices = len(graph)
    
    # Copy original distance matrix
    distances = [row[:] for row in graph]
    
    # Floyd–Warshall
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                # Skip if either side is unreachable
                if distances[i][k] == INF or distances[k][j] == INF:
                    continue
                
                new_d = distances[i][k] + distances[k][j]
                
                # If no previous path or shorter path found
                if distances[i][j] == INF or new_d < distances[i][j]:
                    distances[i][j] = new_d
    
    return distances