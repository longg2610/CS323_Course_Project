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

def run_shortest_paths(graph, algorithm, eps=1.0):
    if algorithm == "floyd_warshall":
        return floyd_warshall(graph)
    
    elif algorithm == "dp_floyd_warshall_input":
        return dp_floyd_warshall_input_pertubation(graph, eps)
    
    elif algorithm == "dp_floyd_warshall_output":
        return dp_floyd_warshall_output_pertubation(graph, eps)
    
    # elif algorithm == "chen":
        # return chen_algorithm(graph)
    
    else:
        raise ValueError("Unsupported algorithm specified.")
    
    
def compute_accuracy(true_distances, calculated_distances):
    accuracy = np.mean(np.abs(np.array(true_distances) - np.array(calculated_distances)))
    max_accuracy = np.max(np.abs(np.array(true_distances) - np.array(calculated_distances)))    
    return accuracy, max_accuracy


def compute_time(graph, algorithm, eps=1.0):
    start = time.time()
    distances = run_shortest_paths(graph, algorithm, eps)
    end = time.time()
    
    return end - start, distances


def read_graph_from_file(file_path):
    
    edges = []

    # Read the file
    with open(file_path, "r") as f:
        for line in f:
            u, v, w = line.split()
            edges.append((u, v, float(w)))

    # Build set of nodes
    nodes = sorted({u for u, v, w in edges} | {v for u, v, w in edges})
    index = {node: i for i, node in enumerate(nodes)}

    # Initialize adjacency matrix (0 = no edge)
    n = len(nodes)
    adj = [[0.0 for _ in range(n)] for _ in range(n)]

    # Fill matrix (undirected)
    for u, v, w in edges:
        i, j = index[u], index[v]
        adj[i][j] = float(w)
        adj[j][i] = float(w)  # remove if directed
        
    return adj


def main():
    
    num_runs = 5
    epsilon_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    graph = read_graph_from_file("./data/graph_300_100.txt")
    
    # Get true distances (non-private Floyd-Warshall)
    true_distances = floyd_warshall(graph)
    
    # Storage for results across epsilon values
    floyd_warshall_results = {'mean_error': [], 'max_error': [], 'runtime': []}
    dp_input_results = {'mean_error': [], 'max_error': [], 'runtime': []}
    dp_output_results = {'mean_error': [], 'max_error': [], 'runtime': []}
    
    # Run experiments for each epsilon value
    for eps in epsilon_values:
        print(f"\nRunning experiments for epsilon = {eps}")
        
        # Floyd-Warshall (baseline - epsilon doesn't affect it but we pass it for consistency)
        fw_runtimes = []
        fw_mean_errors = []
        fw_max_errors = []
        
        for _ in range(num_runs):
            runtime, distances = compute_time(graph, "floyd_warshall", eps)
            mean_err, max_err = compute_accuracy(true_distances, distances)
            fw_runtimes.append(runtime)
            fw_mean_errors.append(mean_err)
            fw_max_errors.append(max_err)
        
        floyd_warshall_results['mean_error'].append(np.mean(fw_mean_errors))
        floyd_warshall_results['max_error'].append(np.mean(fw_max_errors))
        floyd_warshall_results['runtime'].append(np.mean(fw_runtimes))
        
        # DP Floyd-Warshall with Input Perturbation
        dp_input_runtimes = []
        dp_input_mean_errors = []
        dp_input_max_errors = []
        
        for _ in range(num_runs):
            runtime, distances = compute_time(graph, "dp_floyd_warshall_input", eps)
            mean_err, max_err = compute_accuracy(true_distances, distances)
            dp_input_runtimes.append(runtime)
            dp_input_mean_errors.append(mean_err)
            dp_input_max_errors.append(max_err)
        
        dp_input_results['mean_error'].append(np.mean(dp_input_mean_errors))
        dp_input_results['max_error'].append(np.mean(dp_input_max_errors))
        dp_input_results['runtime'].append(np.mean(dp_input_runtimes))
        
        # DP Floyd-Warshall with Output Perturbation
        dp_output_runtimes = []
        dp_output_mean_errors = []
        dp_output_max_errors = []
        
        for _ in range(num_runs):
            runtime, distances = compute_time(graph, "dp_floyd_warshall_output", eps)
            mean_err, max_err = compute_accuracy(true_distances, distances)
            dp_output_runtimes.append(runtime)
            dp_output_mean_errors.append(mean_err)
            dp_output_max_errors.append(max_err)
        
        dp_output_results['mean_error'].append(np.mean(dp_output_mean_errors))
        dp_output_results['max_error'].append(np.mean(dp_output_max_errors))
        dp_output_results['runtime'].append(np.mean(dp_output_runtimes))
    
    # Create the three line plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Epsilon vs Mean Error
    axes[0].plot(epsilon_values, floyd_warshall_results['mean_error'], 
                 marker='o', label='Floyd-Warshall', linewidth=2)
    axes[0].plot(epsilon_values, dp_input_results['mean_error'], 
                 marker='s', label='DP Floyd-Warshall (Input)', linewidth=2)
    axes[0].plot(epsilon_values, dp_output_results['mean_error'], 
                 marker='^', label='DP Floyd-Warshall (Output)', linewidth=2)
    axes[0].set_xlabel('Epsilon (ε)', fontsize=12)
    axes[0].set_ylabel('Mean Error', fontsize=12)
    axes[0].set_title('Epsilon vs Mean Error', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon vs Max Error
    axes[1].plot(epsilon_values, floyd_warshall_results['max_error'], 
                 marker='o', label='Floyd-Warshall', linewidth=2)
    axes[1].plot(epsilon_values, dp_input_results['max_error'], 
                 marker='s', label='DP Floyd-Warshall (Input)', linewidth=2)
    axes[1].plot(epsilon_values, dp_output_results['max_error'], 
                 marker='^', label='DP Floyd-Warshall (Output)', linewidth=2)
    axes[1].set_xlabel('Epsilon (ε)', fontsize=12)
    axes[1].set_ylabel('Max Error', fontsize=12)
    axes[1].set_title('Epsilon vs Max Error', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon vs Runtime
    axes[2].plot(epsilon_values, floyd_warshall_results['runtime'], 
                 marker='o', label='Floyd-Warshall', linewidth=2)
    axes[2].plot(epsilon_values, dp_input_results['runtime'], 
                 marker='s', label='DP Floyd-Warshall (Input)', linewidth=2)
    axes[2].plot(epsilon_values, dp_output_results['runtime'], 
                 marker='^', label='DP Floyd-Warshall (Output)', linewidth=2)
    axes[2].set_xlabel('Epsilon (ε)', fontsize=12)
    axes[2].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[2].set_title('Epsilon vs Runtime', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for i, eps in enumerate(epsilon_values):
        print(f"\nEpsilon = {eps}")
        print(f"  Floyd-Warshall: Mean Error = {floyd_warshall_results['mean_error'][i]:.4f}, "
              f"Max Error = {floyd_warshall_results['max_error'][i]:.4f}, "
              f"Runtime = {floyd_warshall_results['runtime'][i]:.4f}s")
        print(f"  DP Input:       Mean Error = {dp_input_results['mean_error'][i]:.4f}, "
              f"Max Error = {dp_input_results['max_error'][i]:.4f}, "
              f"Runtime = {dp_input_results['runtime'][i]:.4f}s")
        print(f"  DP Output:      Mean Error = {dp_output_results['mean_error'][i]:.4f}, "
              f"Max Error = {dp_output_results['max_error'][i]:.4f}, "
              f"Runtime = {dp_output_results['runtime'][i]:.4f}s")


if __name__ == "__main__":
    main()