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
import os
    
    
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


def compute_time(graph, function, *args, **kwargs):
    start = time.time()
    distances = function(graph, *args, **kwargs)
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


def run_experiment(graph, epsilon_values, vertices, max_weight, num_runs=1):
    # Floyd-Warshall baseline (runtime and distances)
    fw_runtimes = []
    fw_distances = None
    for _ in range(num_runs):
        t, d = compute_time(graph, floyd_warshall)
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
            t, d = compute_time(graph, dp_floyd_warshall_input_perturbation, eps)
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
            t, d = compute_time(graph, dp_floyd_warshall_output_perturbation, eps)
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
            t, d = compute_time(graph, bounded_weights, epsilon=eps, A=max_weight, V=vertices)
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

def plot_results(epsilon_values, results, filename="results.png"):

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.tight_layout(pad=4.0)

    # --- Mean error ---
    ax = axes[0]
    ax.plot(epsilon_values, results["dp_in_mean"], label="DP Input Mean Error")
    ax.plot(epsilon_values, results["dp_out_mean"], label="DP Output Mean Error")
    ax.plot(epsilon_values, results["chen_mean"], label="Chen et al Mean Error")
    ax.set_yscale('log')
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Mean Error")
    ax.set_title("Mean Error vs epsilon")
    ax.grid(True)
    ax.legend()

    # --- Max error ---
    ax = axes[1]
    ax.plot(epsilon_values, results["dp_in_max"], label="DP Input Max Error")
    ax.plot(epsilon_values, results["dp_out_max"], label="DP Output Max Error")
    ax.plot(epsilon_values, results["chen_max"], label="Chen et al Max Error")
    ax.set_yscale('log')
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Max Error")
    ax.set_title("Max Error vs epsilon")
    ax.grid(True)
    ax.legend()

    # --- Runtime comparison ---
    ax = axes[2]
    ax.plot(epsilon_values, results["dp_in_runtime"], label="DP Input Runtime", marker="o")
    ax.plot(epsilon_values, results["dp_out_runtime"], label="DP Output Runtime", marker="o")
    ax.plot(epsilon_values, results["chen_runtime"], label="Chen et al. Runtime", marker="o")
    ax.axhline(
        y=results["floyd_runtime"],
        linestyle="--",
        color="r",
        label="Floyd-Warshall"
    )
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs epsilon — DP vs Floyd-Warshall")
    ax.grid(True)
    ax.legend()

    # Save to file
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    # Show
    plt.show()

# Helper to plot mean error, max error, and runtime for a dictionary of results
def plot_three_metrics(x_values, all_results, x_label, filename_prefix):
    # Convert dict-of-results into lists sorted by the x-values
    xs = sorted(x_values)

    dp_in_mean = [all_results[x]["dp_in_mean"][0] for x in xs]
    dp_out_mean = [all_results[x]["dp_out_mean"][0] for x in xs]
    chen_mean   = [all_results[x]["chen_mean"][0]   for x in xs]

    dp_in_max = [all_results[x]["dp_in_max"][0] for x in xs]
    dp_out_max = [all_results[x]["dp_out_max"][0] for x in xs]
    chen_max   = [all_results[x]["chen_max"][0]   for x in xs]

    dp_in_rt = [all_results[x]["dp_in_runtime"][0] for x in xs]
    dp_out_rt = [all_results[x]["dp_out_runtime"][0] for x in xs]
    chen_rt   = [all_results[x]["chen_runtime"][0]   for x in xs]
    fw_rt     = [all_results[x]["floyd_runtime"]     for x in xs]

    # --------------------------
    # Mean Error
    # --------------------------
    plt.figure(figsize=(8,5))
    plt.plot(xs, dp_in_mean, marker='o', label="DP Input Mean Error")
    plt.plot(xs, dp_out_mean, marker='o', label="DP Output Mean Error")
    plt.plot(xs, chen_mean, marker='o', label="Chen et al Mean Error")
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel("Mean Error")
    plt.title(f"Mean Error vs {x_label}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename_prefix}_mean_error.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------
    # Max Error
    # --------------------------
    plt.figure(figsize=(8,5))
    plt.plot(xs, dp_in_max, marker='o', label="DP Input Max Error")
    plt.plot(xs, dp_out_max, marker='o', label="DP Output Max Error")
    plt.plot(xs, chen_max, marker='o', label="Chen et al Max Error")
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel("Max Error")
    plt.title(f"Max Error vs {x_label}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename_prefix}_max_error.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------
    # Runtime
    # --------------------------
    plt.figure(figsize=(8,5))
    plt.plot(xs, dp_in_rt, marker='o', label="DP Input Runtime")
    plt.plot(xs, dp_out_rt, marker='o', label="DP Output Runtime")
    plt.plot(xs, chen_rt, marker='o', label="Chen et al Runtime")
    plt.plot(xs, fw_rt, marker='o', label="Floyd-Warshall Runtime", linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Algorithm Runtime vs {x_label}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename_prefix}_runtime.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_vary_vertices():
    print("Run varying number of vertices")

    test_files = [
        "graph_10_200_0.2.txt",
        "graph_15_200_0.2.txt",
        "graph_20_200_0.2.txt",
        "graph_25_200_0.2.txt",
        "graph_30_200_0.2.txt",
    ]

    epsilon_values = [0.2]
    max_weight = 200

    all_results = {}

    for fname in test_files:
        path = os.path.join("./data", fname)
        graph = read_graph_from_file(path)
        n = len(graph)
        vertices = set(range(n))

        print(f"\n--- Running experiment for {fname} (n={n}) ---")
        results = run_experiment(graph, epsilon_values, vertices=vertices,
                                 max_weight=max_weight, num_runs=3)
        all_results[n] = results

    # Plot mean/max error & runtime
    plot_three_metrics(
        x_values=list(all_results.keys()),
        all_results=all_results,
        x_label="Number of Vertices",
        filename_prefix="results_num_vertices_comparison"
    )


# -----------------------------------
# TEST 2: Vary max weight (A parameter)
# -----------------------------------

def run_vary_max_weight():
    print("Run varying max weight")

    test_files = [
        "graph_25_100_0.6.txt",
        "graph_25_500_0.6.txt",
        "graph_25_1000_0.6.txt",
        "graph_25_2000_0.6.txt",
        "graph_25_3000_0.6.txt",
    ]

    epsilon_values = [0.1]

    all_results = {}
    max_weights = []

    for fname in test_files:
        path = os.path.join("./data", fname)

        parts = fname.split("_")
        maxW = int(parts[2])
        max_weights.append(maxW)

        graph = read_graph_from_file(path)
        vertices = set(range(len(graph)))

        print(f"\n--- Running experiment for {fname} (maxW={maxW}) ---")
        results = run_experiment(graph, epsilon_values, vertices=vertices,
                                 max_weight=maxW, num_runs=3)
        all_results[maxW] = results

    # Plot results
    plot_three_metrics(
        x_values=max_weights,
        all_results=all_results,
        x_label="Max Weight",
        filename_prefix="results_max_weight_comparison"
    )


if __name__ == "__main__":
    # file = "./data/graph_16_200_0.9.txt"
    # graph = read_graph_from_file(file)
    # epsilon_values = [0.1, 0.2, 0.5, 0.75, 1]
    # n = len(graph)
    # vertices = set(range(n))
    # max_weight = 200 
    # results = run_experiment(graph, epsilon_values, vertices=vertices, max_weight=max_weight, num_runs=5)

    # filename = f"results.png"
    # plot_results(epsilon_values, results)
    
    run_vary_vertices()
    run_vary_max_weight()