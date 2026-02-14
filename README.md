# Graph Convolutional Network with Random Weights (GCNRW)

This project implements a **Graph Convolutional Network with Random Weights (GCNRW)**, a high-speed graph neural network architecture that utilizes fixed random hidden weights and an analytical closed-form solution for the output layer. This approach, inspired by Extreme Learning Machines (ELM), significantly reduces training time by avoiding iterative backpropagation.

## ## Technical Features

* **Random Hidden Layer:** Uses a fixed weight matrix  initialized with a uniform distribution .
* **Second-Order Smoothing:** Employs the squared adjacency matrix () to aggregate features from two-hop neighborhoods.
* **Analytical Solver:** Computes output weights () using Tikhonov regularization () to solve the linear system.
* **Efficiency:** Automatically selects between primal and dual solutions based on the ratio of hidden units to training samples.

## ## Performance Benchmarks

Evaluated on the **Planetoid** datasets with the following results:

| Dataset | Hidden Units | Lambda () | Train Acc | Test Acc |
| --- | --- | --- | --- | --- |
| **Cora** | 8,000 | 15 | 98.15% | 79.33% |
| **Citeseer** | 10,000 | 100 | 98.40% | 71.25% |
| **PubMed** | 7,000 | 10 | 89.65% | 80.24% |

---

## ## Feasibility for IEEE TPAMI

Publishing a Random Weight GCN in a top-tier journal like **IEEE TPAMI** is feasible but requires addressing the following high-standard criteria:

* **Theoretical Foundations:** TPAMI demands more than empirical success. You must provide theoretical analysis, such as the **Universal Approximation Property** for graph signals or bounds related to the **Johnson-Lindenstrauss Lemma** regarding the random projections.
* **Algorithmic Novelty:** Since ELM-based GCNs exist in literature (e.g., Random-GCN), you must highlight a unique contribution, such as a novel regularization technique, a specific solution for **graph heterophily**, or a new way to handle **large-scale graph spectral filtering**.
* **Comprehensive Evaluation:** Beyond the Planetoid datasets (Cora, Citeseer, PubMed), a TPAMI-level paper requires testing on diverse benchmarks, including large-scale OGB (Open Graph Benchmark) datasets and heterophilic graphs.
* **Comparative Complexity:** Detailed analysis comparing the computational complexity and energy efficiency of GCNRW against state-of-the-art iterative models like GATv2 or Graph Transformers is essential.

