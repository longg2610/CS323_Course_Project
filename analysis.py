"""
James Bui, Long Pham
December 2025
analysis.py
This file contains the analysis of different all-pair shortest distance on a weighted graph.
"""
from floyd_warshall import floyd_warshall
from dp_floyd_warshall import dp_floyd_warshall_input_pertubation, dp_floyd_warshall_output_pertubation
# from chen import chen_algorithm
import time
import numpy as np
import matplotlib.pyplot as plt

def run_shortest_paths(graph, algorithm):
    if algorithm == "floyd_warshall":
        return floyd_warshall(graph)
    
    elif algorithm == "dp_floyd_warshall_input":
        return dp_floyd_warshall_input_pertubation(graph)
    
    elif algorithm == "dp_floyd_warshall_output":
        return dp_floyd_warshall_output_pertubation(graph)
    
    # elif algorithm == "chen":
        # return chen_algorithm(graph)
    
    else:
        raise ValueError("Unsupported algorithm specified.")
    
    
def compute_accuracy(true_distances, calculated_distances):
    accuracy = np.mean(np.abs(np.array(true_distances) - np.array(calculated_distances)))
    max_accuracy = np.max(np.abs(np.array(true_distances) - np.array(calculated_distances)))    
    return accuracy, max_accuracy


def compute_time(graph, algorithm):
    start = time.time()
    distances = run_shortest_paths(graph, algorithm)
    end = time.time()
    

    return end - start, distances


def read_graph_from_file(file_path):
    
    edges = []

    # Read the file
    with open(file_path, "r") as f:
        for line in f:
            u, v, w = line.split()
            edges.append((u, v, int(w)))

    # Build set of nodes
    nodes = sorted({u for u, v, w in edges} | {v for u, v, w in edges})
    index = {node: i for i, node in enumerate(nodes)}

    # Initialize adjacency matrix (0 = no edge)
    n = len(nodes)
    adj = [[0 for _ in range(n)] for _ in range(n)]

    # Fill matrix (undirected)
    for u, v, w in edges:
        i, j = index[u], index[v]
        adj[i][j] = w
        adj[j][i] = w  # remove if directed

    # Print results
    print("Nodes:", nodes)
    print("Adjacency matrix:")
    for row in adj:
        print(row)
        
    return adj


def main():
    
    num_runs = 5
    graph = read_graph_from_file("")
    
    # Floyd-Warshall 
        
    floyd_warshall_runtimes = []
    floyd_warshall_accuracies = []
    true_distances = floyd_warshall(graph)
    
    for _ in range(num_runs):
        runtime, distances = compute_time(graph, "floyd_warshall")
        accuracy = compute_accuracy(true_distances, distances)
        floyd_warshall_runtimes.append(runtime)
        floyd_warshall_accuracies.append(accuracy)
    
    
    # DP Floyd-Warshall with Input Perturbation
    dp_floyd_warshall_input_runtimes = []
    dp_floyd_warshall_input_accuracies = []
    
    for _ in range(num_runs):
        runtime, distances = compute_time(graph, "dp_floyd_warshall_input")
        accuracy = compute_accuracy(true_distances, distances)
        dp_floyd_warshall_input_runtimes.append(runtime)
        dp_floyd_warshall_input_accuracies.append(accuracy)
    
    # DP Floyd-Warshall with Output Perturbation
    dp_floyd_warshall_output_runtimes = []
    dp_floyd_warshall_output_accuracies = []
    
    for _ in range(num_runs):
        runtime, distances = compute_time(graph, "dp_floyd_warshall_output")
        accuracy = compute_accuracy(true_distances, distances)
        dp_floyd_warshall_output_runtimes.append(runtime)
        dp_floyd_warshall_output_accuracies.append(accuracy)
    
    # # Chen's Algorithm
    # chen_runtimes = []
    # chen_accuracies = []
    
    # for _ in range(num_runs):
    #     runtime, distances, paths = compute_time(graph, "chen")
    #     accuracy = compute_accuracy(distances, paths)
    #     chen_runtimes.append(runtime)
    #     chen_accuracies.append(accuracy)

    # Runtime plot
    algorithms = ["floyd_warshall", "dp_floyd_warshall_input", "dp_floyd_warshall_output"]
    avg_runtimes = [
        np.mean(floyd_warshall_runtimes),
        np.mean(dp_floyd_warshall_input_runtimes),
        np.mean(dp_floyd_warshall_output_runtimes),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, avg_runtimes)
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Runtime Comparison of Algorithms")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    # Accuracy plot (Mean vs Max)
    mean_errors = [
        np.mean([a[0] for a in floyd_warshall_accuracies]),
        np.mean([a[0] for a in dp_floyd_warshall_input_accuracies]),
        np.mean([a[0] for a in dp_floyd_warshall_output_accuracies]),
    ]

    max_errors = [
        np.mean([a[1] for a in floyd_warshall_accuracies]),
        np.mean([a[1] for a in dp_floyd_warshall_input_accuracies]),
        np.mean([a[1] for a in dp_floyd_warshall_output_accuracies]),
    ]

    x = np.arange(len(algorithms))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, mean_errors, width=width, label="Mean Error")
    plt.bar(x + width/2, max_errors, width=width, label="Max Error")

    plt.xticks(x, algorithms, rotation=20)
    plt.ylabel("Error")
    plt.title("Accuracy Comparison of Algorithms")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()