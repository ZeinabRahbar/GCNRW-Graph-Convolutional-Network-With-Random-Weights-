# Graph Convolutional Network with Random Weights (GCNRW)

This project aims to reproduce the experimental results reported in the IEEE TPAMI paper “Are Graph Convolutional Networks with Random Weights Feasible?” We follow the original experimental protocol, hyperparameter settings, and data splits provided by the authors.

## Mathematical Foundations & Core Thesis

### Core Mathematical Thesis
[cite_start]Traditional Graph Convolutional Networks (GCNs) require training multiple weight layers using full-batch gradient descent across the entire graph structure at each iteration[cite: 418, 536]. [cite_start]This poses a major computational bottleneck for large-scale graph structures[cite: 418, 536]. 

[cite_start]To overcome this, the paper introduces **Graph Convolutional Networks with Random Weights (GCN-RW)**[cite: 420, 449]. [cite_start]The foundational thesis is that projecting graph-smoothed feature spaces into a randomized hidden layer allows the learning objective to be reformulated as a regularized least squares problem (Tikhonov regularization) rather than a non-linear optimization problem[cite: 420, 450, 542]. [cite_start]This transformation allows parameters to be solved analytically in a single closed-form step, bypassing backpropagation completely [cite: 450, 549-552].

[cite_start]Formally, given an undirected graph $G=\{V,\mathcal{E},A\}$ and a node feature matrix $X \in \mathbb{R}^{N \times m}$ [cite: 502-504][cite_start], GCN-RW uses a renormalized graph convolutional operator $\hat{A} = \hat{D}^{-1/2}(A + I)\hat{D}^{-1/2}$ [cite: 525-527]. [cite_start]The continuous mapping function $Z$ for the single-layer architecture is written as[cite: 538, 539]:

$$Z = f(X, \hat{A}) = \text{sigmoid}(\hat{A}^2 X W)\beta$$

where:
- [cite_start]$W \in \mathbb{R}^{m \times L}$ is a fixed hidden layer filter matrix whose entries are drawn randomly and independently from a uniform distribution space $\mathcal{U}(-\theta, \theta)$[cite: 541, 584].
- [cite_start]$\text{sigmoid}(\cdot)$ serves as the non-linear activation function[cite: 539, 560].
- [cite_start]$\beta \in \mathbb{R}^{L \times C}$ represents the learnable output weight matrix[cite: 541].

[cite_start]The learning objective minimizes the regularized least squares loss over the set of labeled training vertices $\mathcal{Y}$ [cite: 542-544]:

$$\min_{\beta} \| H\beta - Y \|_F^2 + \lambda \|\beta\|_F^2$$

where $H = [\text{sigmoid}(\hat{A}^2 X W)]|_{\mathcal{Y}} \in \mathbb{R}^{|\mathcal{Y}| [cite_start]\times L}$ represents the hidden output state matrix mapped to the training nodes[cite: 558], $Y \in \mathbb{R}^{|\mathcal{Y}| [cite_start]\times C}$ is the ground-truth label matrix [cite: 548][cite_start], and $\lambda > 0$ is the regularization parameter[cite: 548].

[cite_start]The paper proves **Theorem 1 (Universal Approximation Upper Bound)**, establishing that for any bounded and continuous target function $f$ on the graph, the expected approximation error is probabilistically upper-bounded by the size of the hidden layer[cite: 590, 611, 612]:

$$d_V(f, \hat{f}_L) \le \frac{C}{\sqrt{L}}$$

[cite_start]This proves that GCN-RW behaves as a universal approximator in a probability sense when $L$ is sufficiently large[cite: 616].

### Why a Public Implementation Was Missing
[cite_start]Although the original paper published in *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* established rigorous theoretical guarantees for GCNs with random weights[cite: 413, 421], an official, publicly accessible repository was not open-sourced by the authors. 

[cite_start]Historically, randomized architectures like Random Vector Functional-Link (RVFL) networks and Extreme Learning Machines (ELM) have experienced a gap between academic theory and reusable public software libraries[cite: 54, 57]. Furthermore, standard graph deep learning libraries (such as PyTorch Geometric and DGL) are inherently designed around backpropagation and automatic differentiation rather than analytical, closed-form matrix solvers. This repository fills that gap by providing an independent implementation that accurately matches the paper's original experimental protocol, parameter selections, and mathematical routines.

### Architectural Breakdown
[cite_start]The GCN-RW architectural pipeline executes through five distinct, decoupled stages [cite: 583-585]:

1. [cite_start]**Second-Order Graph Smoothing**: Unlike vanilla GCNs that perform first-order neighborhood propagation per layer[cite: 445], GCN-RW aggregates structural graph properties across a two-hop neighborhood up front:
   $$\bar{X} = \hat{A}(\hat{A}X) \in \mathbb{R}^{N \times m}$$
   [cite_start]Because this step contains no learnable parameters, it can be entirely pre-computed during data preprocessing to eliminate redundant feature aggregation during training[cite: 566, 568].
2. [cite_start]**Random Projections**: The graph-smoothed node feature matrix $\bar{X}$ is projected into a higher-dimensional space using the fixed, randomly generated weights $W$[cite: 541, 584]:
   $$H_{\text{raw}} = \bar{X}W \in \mathbb{R}^{N \times L}$$
3. [cite_start]**Non-linear Activation**: The randomized hidden states are transformed using the element-wise sigmoid function to generate non-linear embeddings[cite: 539, 560]:
   $$H_{\text{full}} = \text{sigmoid}(H_{\text{raw}}) \in \mathbb{R}^{N \times L}$$
4. [cite_start]**Training Node Selection**: The row components matching unlabeled nodes are masked out to form the specialized training sub-matrix[cite: 558, 584]:
   $$H = H_{\text{full}}[\mathcal{Y}, :]$$
5. [cite_start]**Analytical Solver Optimization**: To compute the optimal parameter configuration $\beta^*$, the model bypasses gradient descent and automatically toggles between a Primal or Dual solution based on the ratio of hidden nodes $L$ to labeled items $|\mathcal{Y}|$ to maximize efficiency [cite: 549-552]:
   - **Primal Solution** (when $|\mathcal{Y}| \ge L$)[cite: 550]:
     $$\beta^* = (H^T H + \lambda I_L)^{-1}H^T Y$$
   - [cite_start]**Dual Solution** (when $|\mathcal{Y}| < L$)[cite: 552]:
     $$\beta^* = H^T (H H^T + \lambda I_{|\mathcal{Y}|})^{-1}Y$$

---

## Technical Features

* [cite_start]**Random Hidden Layer**: Uses a fixed weight matrix initialized with a uniform distribution $\mathcal{U}(-\theta, \theta)$[cite: 541, 584].
* [cite_start]**Second-Order Smoothing**: Employs the squared adjacency matrix to aggregate features from two-hop neighborhoods[cite: 539, 583].
* [cite_start]**Analytical Solver**: Computes output weights using Tikhonov regularization to solve the linear system [cite: 542, 549-552].
* [cite_start]**Efficiency**: Automatically selects between primal and dual solutions based on the ratio of hidden units to training samples [cite: 550-552].

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
  author={Huang, Changqin and Li, Ming & Cao, Feilong & Fujita, Hamido & Li, Zhao & Wu, Xindong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={3},
  pages={2751--2768},
  year={2023},
  month={Mar},
  doi={10.1109/TPAMI.2022.3183143},
  pmid={35704541}
}
