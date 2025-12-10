"""
James Bui, Long Pham
December 2025
analysis.py
This file contains the analysis of different all-pair shortest distance on a weighted graph.
"""
from floyd_warshall import floyd_warshall
from dp_floyd_warshall import dp_floyd_warshall_input_perturbation, dp_floyd_warshall_output_perturbation
from chen import bounded_weights
import time
import numpy as np
import matplotlib.pyplot as plt

def run_shortest_paths(graph, algorithm, eps=1.0):
    if algorithm == "floyd_warshall":
        return floyd_warshall(graph)
    
    elif algorithm == "dp_floyd_warshall_input":
        return dp_floyd_warshall_input_perturbation(graph, eps)
    
    elif algorithm == "dp_floyd_warshall_output":
        return dp_floyd_warshall_output_perturbation(graph, eps)
    
    elif algorithm == "chen":
        return bounded_weights(graph, eps)
    
    else:
        raise ValueError("Unsupported algorithm specified.")
    
    
def compute_accuracy(true_distances, calculated_distances):
    A = np.array(true_distances, dtype=float)
    B = np.array(calculated_distances, dtype=float)

    # Mask only entries where both distances are finite (reachable)
    mask = np.isfinite(A) & np.isfinite(B)

    if not np.any(mask):
        return 0.0, 0.0

    diff = np.abs(A[mask] - B[mask])

    mean_error = diff.mean()
    max_error = diff.max()

    return mean_error, max_error


def compute_time(graph, algorithm, eps=1.0):
    start = time.time()
    distances = run_shortest_paths(graph, algorithm, eps)
    end = time.time()
    
    return end - start, distances


def read_graph_from_file(file_path):
    adj = []

    # --- Read edges ---
    with open(file_path, "r") as f:
        for line in f:
            row = []
            for val in line.strip().split():
                if val == "INF":
                    row.append(float("inf"))
                else:
                    row.append(float(val))
            adj.append(row)

    print(f"Graph loaded from {file_path} with {len(adj)} vertices.")
    
    return adj


def run_experiment(graph, epsilon_values, num_runs=1):
    # Floyd-Warshall baseline (runtime and distances)
    fw_runtimes = []
    fw_distances = None
    for _ in range(num_runs):
        t, d = compute_time(graph, "floyd_warshall")
        fw_runtimes.append(t)
        fw_distances = d

    fw_runtime_avg = np.mean(fw_runtimes)
    print(f"Floyd-Warshall average runtime over {num_runs} run(s): {fw_runtime_avg:.6f} s")

    # Storage for DP algorithms
    results = {
        "floyd_runtime": fw_runtime_avg,
        "dp_in_mean": [], "dp_in_max": [], "dp_in_runtime": [],
        "dp_out_mean": [], "dp_out_max": [], "dp_out_runtime": [],
        "chen_mean": [], "chen_max": [], "chen_runtime": [],
    }

    # Sweep epsilon values
    for eps in epsilon_values:
        print(f"\nEpsilon = {eps}")

        # DP Input perturbation
        dp_in_runtimes = []
        dp_in_mean_errors = []
        dp_in_max_errors = []

        for _ in range(num_runs):
            t, d = compute_time(graph, "dp_floyd_warshall_input", eps)
            m, M = compute_accuracy(fw_distances, d)
            dp_in_runtimes.append(t)
            dp_in_mean_errors.append(m)
            dp_in_max_errors.append(M)

        # Store averages
        results["dp_in_runtime"].append(np.mean(dp_in_runtimes))
        results["dp_in_mean"].append(np.mean(dp_in_mean_errors))
        results["dp_in_max"].append(np.mean(dp_in_max_errors))
        print(f"DP Input:    Avg Runtime = {np.mean(dp_in_runtimes):.6f}s, "
              f"Mean Error = {np.mean(dp_in_mean_errors):.6f}, "
              f"Max Error = {np.mean(dp_in_max_errors):.6f}")

        # DP Output perturbation
        dp_out_runtimes = []
        dp_out_mean_errors = []
        dp_out_max_errors = []

        for _ in range(num_runs):
            t, d = compute_time(graph, "dp_floyd_warshall_output", eps)
            m, M = compute_accuracy(fw_distances, d)
            dp_out_runtimes.append(t)
            dp_out_mean_errors.append(m)
            dp_out_max_errors.append(M)

        results["dp_out_runtime"].append(np.mean(dp_out_runtimes))
        results["dp_out_mean"].append(np.mean(dp_out_mean_errors))
        results["dp_out_max"].append(np.mean(dp_out_max_errors))
        print(f"DP Output:   Avg Runtime = {np.mean(dp_out_runtimes):.6f}s, "
              f"Mean Error = {np.mean(dp_out_mean_errors):.6f}, "
              f"Max Error = {np.mean(dp_out_max_errors):.6f}")
        
        # Chen's Algorithm
        chen_runtimes = []
        chen_mean_errors = []
        chen_max_errors = []

        for _ in range(num_runs):
            t, d = compute_time(graph, "dp_floyd_warshall_output", eps)
            m, M = compute_accuracy(fw_distances, d)
            chen_runtimes.append(t)
            chen_mean_errors.append(m)
            chen_max_errors.append(M)

        results["chen_runtime"].append(np.mean(chen_runtimes))
        results["chen_mean"].append(np.mean(chen_mean_errors))
        results["chen_max"].append(np.mean(chen_max_errors))
        print(f"Chen et al:   Avg Runtime = {np.mean(chen_runtimes):.6f}s, "
              f"Mean Error = {np.mean(chen_mean_errors):.6f}, "
              f"Max Error = {np.mean(chen_max_errors):.6f}")
    return results

def plot_results(epsilon_values, results):

    # --- Mean error ---
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_values, results["dp_in_mean"], label="DP Input Mean Error")
    plt.plot(epsilon_values, results["dp_out_mean"], label="DP Output Mean Error")
    plt.plot(epsilon_values, results["chen_mean"], label="Chen et al Mean Error")
    plt.xlabel("epsilon")
    plt.ylabel("Mean Error")
    plt.title("Mean Error vs epsilon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Max error ---
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_values, results["dp_in_max"], label="DP Input Max Error")
    plt.plot(epsilon_values, results["dp_out_max"], label="DP Output Max Error")
    plt.plot(epsilon_values, results["chen_max"], label="Chen et al Max Error")
    plt.xlabel("epsilon")
    plt.ylabel("Max Error")
    plt.title("Max Error vs epsilon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Runtime comparison ---
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_values, results["dp_in_runtime"], label="DP Input Runtime", marker="o")
    plt.plot(epsilon_values, results["dp_out_runtime"], label="DP Output Runtime", marker="o")
    plt.plot(epsilon_values, results["chen_runtime"], label="Chen et al. Runtime", marker="o")
    plt.axhline(
        y=results["floyd_runtime"],
        linestyle="--",
        color="r",
        label="Floyd-Warshall"
    )
    plt.xlabel("epsilon")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs epsilon — DP vs Floyd-Warshall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file = "./data/graph_10_200_0.2.txt"
    graph = read_graph_from_file(file)
    epsilon_values = [0.2, 0.5, 1, 2, 5, 10]

    results = run_experiment(graph, epsilon_values, num_runs=5)

    plot_results(epsilon_values, results)