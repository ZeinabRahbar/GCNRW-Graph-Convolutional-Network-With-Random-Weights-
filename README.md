# Graph Convolutional Network with Random Weights (GCNRW)

This project aims to reproduce the experimental results reported in the IEEE TPAMI paper “Are Graph Convolutional Networks with Random Weights Feasible?” We follow the original experimental protocol, hyperparameter settings, and data splits provided by the authors.

## Mathematical Foundations & Core Thesis

### Core Mathematical Thesis
Traditional Graph Convolutional Networks (GCNs) require training multiple weight layers using full-batch gradient descent across the entire graph structure at each iteration. This poses a major computational bottleneck for large-scale graph structures. 

To overcome this, the paper introduces **Graph Convolutional Networks with Random Weights (GCN-RW)**. The foundational thesis is that projecting graph-smoothed feature spaces into a randomized hidden layer allows the learning objective to be reformulated as a regularized least squares problem (Tikhonov regularization) rather than a non-linear optimization problem. This transformation allows parameters to be solved analytically in a single closed-form step, bypassing backpropagation completely.

Formally, given an undirected graph $G=\{V,\mathcal{E},A\}$ and a node feature matrix $X \in \mathbb{R}^{N \times m}$, GCN-RW uses a renormalized graph convolutional operator $\hat{A} = \hat{D}^{-1/2}(A + I)\hat{D}^{-1/2}$. The continuous mapping function $Z$ for the single-layer architecture is written as:

$$Z = f(X, \hat{A}) = \text{sigmoid}(\hat{A}^2 X W)\beta$$

where:
- $W \in \mathbb{R}^{m \times L}$ is a fixed hidden layer filter matrix whose entries are drawn randomly and independently from a uniform distribution space $\mathcal{U}(-\theta, \theta)$.
- $\text{sigmoid}(\cdot)$ serves as the non-linear activation function.
- $\beta \in \mathbb{R}^{L \times C}$ represents the learnable output weight matrix.

The learning objective minimizes the regularized least squares loss over the set of labeled training vertices $\mathcal{Y}$:

$$\min_{\beta} \| H\beta - Y \|_F^2 + \lambda \|\beta\|_F^2$$

where $H = [\text{sigmoid}(\hat{A}^2 X W)]|_{\mathcal{Y}} \in \mathbb{R}^{|\mathcal{Y}| \times L}$ represents the hidden output state matrix mapped to the training nodes, $Y \in \mathbb{R}^{|\mathcal{Y}| \times C}$ is the ground-truth label matrix, and $\lambda > 0$ is the regularization parameter.

The paper proves **Theorem 1 (Universal Approximation Upper Bound)**, establishing that for any bounded and continuous target function $f$ on the graph, the expected approximation error is probabilistically upper-bounded by the size of the hidden layer:

$$d_V(f, \hat{f}_L) \le \frac{C}{\sqrt{L}}$$

This proves that GCN-RW behaves as a universal approximator in a probability sense when $L$ is sufficiently large.

### Why a Public Implementation Was Missing
Although the original paper published in *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* established rigorous theoretical guarantees for GCNs with random weights, an official, publicly accessible repository was not open-sourced by the authors. 

Historically, randomized architectures like Random Vector Functional-Link (RVFL) networks and Extreme Learning Machines (ELM) have experienced a gap between academic theory and reusable public software libraries. Furthermore, standard graph deep learning libraries (such as PyTorch Geometric and DGL) are inherently designed around backpropagation and automatic differentiation rather than analytical, closed-form matrix solvers. This repository fills that gap by providing an independent implementation that accurately matches the paper's original experimental protocol, parameter selections, and mathematical routines.

### Architectural Breakdown
The GCN-RW architectural pipeline executes through five distinct, decoupled stages:

1. **Second-Order Graph Smoothing**: Unlike vanilla GCNs that perform first-order neighborhood propagation per layer, GCN-RW aggregates structural graph properties across a two-hop neighborhood up front:
   $$\bar{X} = \hat{A}(\hat{A}X) \in \mathbb{R}^{N \times m}$$
   Because this step contains no learnable parameters, it can be entirely pre-computed during data preprocessing to eliminate redundant feature aggregation during training.
2. **Random Projections**: The graph-smoothed node feature matrix $\bar{X}$ is projected into a higher-dimensional space using the fixed, randomly generated weights $W$:
   $$H_{\text{raw}} = \bar{X}W \in \mathbb{R}^{N \times L}$$
3. **Non-linear Activation**: The randomized hidden states are transformed using the element-wise sigmoid function to generate non-linear embeddings:
   $$H_{\text{full}} = \text{sigmoid}(H_{\text{raw}}) \in \mathbb{R}^{N \times L}$$
4. **Training Node Selection**: The row components matching unlabeled nodes are masked out to form the specialized training sub-matrix:
   $$H = H_{\text{full}}[\mathcal{Y}, :]$$
5. **Analytical Solver Optimization**: To compute the optimal parameter configuration $\beta^*$, the model bypasses gradient descent and automatically toggles between a Primal or Dual solution based on the ratio of hidden nodes $L$ to labeled items $|\mathcal{Y}|$ to maximize efficiency:
   - **Primal Solution** (when $|\mathcal{Y}| \ge L$):
     $$\beta^* = (H^T H + \lambda I_L)^{-1}H^T Y$$
   - **Dual Solution** (when $|\mathcal{Y}| < L$):
     $$\beta^* = H^T (H H^T + \lambda I_{|\mathcal{Y}|})^{-1}Y$$

---

## Technical Features

* **Random Hidden Layer**: Uses a fixed weight matrix initialized with a uniform distribution $\mathcal{U}(-\theta, \theta)$.
* **Second-Order Smoothing**: Employs the squared adjacency matrix to aggregate features from two-hop neighborhoods.
* **Analytical Solver**: Computes output weights using Tikhonov regularization to solve the linear system.
* **Efficiency**: Automatically selects between primal and dual solutions based on the ratio of hidden units to training samples.

## Performance Benchmarks

Evaluated on the Planetoid datasets with the following results:

| Dataset | Hidden Units | Lambda ($\lambda$) | Train Acc | Test Acc |
| :--- | :---: | :---: | :---: | :---: |
| **Cora** | 8,000 | 15 | 98.15% | 79.33% |
| **Citeseer** | 10,000 | 100 | 98.40% | 71.25% |
| **PubMed** | 7,000 | 10 | 89.65% | 80.24% |

## Citation

This repository is an independent implementation of the following paper (I am not the author). If you use this code in your research, please cite the original paper:

```bibtex
@article{huang2023graph,
  title={Are Graph Convolutional Networks With Random Weights Feasible?},
  author={Huang, Changqin and Li, Ming and Cao, Feilong and Fujita, Hamido and Li, Zhao and Wu, Xindong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={3},
  pages={2751--2768},
  year={2023},
  month={Mar},
  doi={10.1109/TPAMI.2022.3183143},
  pmid={35704541}
}
