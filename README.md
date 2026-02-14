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

---Transformers is essential.

