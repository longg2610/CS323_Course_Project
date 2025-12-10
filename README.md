# All-Pair Shortest Distance Algorithms Benchmark
### CS 323: Data Privacy Course Project
Authors: James Bui and Long Pham

## Project Overview

This project benchmarks several all-pairs shortest distance algorithms on weighted graphs, including the standard Floyd–Warshall algorithm, two differentially private variants (DP Input Perturbation and DP Output Perturbation), and the algorithm proposed by Chen et al. (2022). The goal is to compare their accuracy, runtime, and behavior under different privacy and approximation settings. Our experiments focus on small-sized graphs and analyze how perturbation affects both error and computational efficiency.

## How to Run the Code

### 1. Install dependencies
```bash
pip install numpy matplotlib
```

### 2. Generate graphs
To generate graphs, go to `generate_node.py` and edit the parameters (num_vertices, max_weight, edge_probability) to generate desirable graphs. The output graphs will be in `data` directory. 

```bash
python generate_node.py
```

### 3. Run experiments and plot results

```bash
python analysis.py 
```

### 5. View outputs

Plots will be saved in the project main directory and runtime/error statistics will be printed to the console.