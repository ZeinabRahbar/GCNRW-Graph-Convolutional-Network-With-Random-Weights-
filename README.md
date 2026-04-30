![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)
![Stars](https://img.shields.io/github/stars/ZeinabRahbar/GCNRW-Graph-Convolutional-Network-With-Random-Weights-)

# Graph Convolutional Network with Random Weights (GCNRW)

This project aims to reproduce the experimental results reported in the IEEE TPAMI paper **“Are Graph Convolutional Networks with Random Weights Feasible?”** We follow the original experimental protocol, hyperparameter settings, and data splits provided by the authors.

##  Technical Features

* **Random Hidden Layer:** Uses a fixed weight matrix  initialized with a uniform distribution .
* **Second-Order Smoothing:** Employs the squared adjacency matrix to aggregate features from two-hop neighborhoods.
* **Analytical Solver:** Computes output weights using Tikhonov regularization to solve the linear system.
* **Efficiency:** Automatically selects between primal and dual solutions based on the ratio of hidden units to training samples.

##  Performance Benchmarks

Evaluated on the **Planetoid** datasets with the following results:

| Dataset | Hidden Units | Lambda | Train Acc | Test Acc |
| --- | --- | --- | --- | --- |
| **Cora** | 8,000 | 15 | 98.15% | 79.33% |
| **Citeseer** | 10,000 | 100 | 98.40% | 71.25% |
| **PubMed** | 7,000 | 10 | 89.65% | 80.24% |


## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{zhang2022are,
  title={Are Graph Convolutional Networks with Random Weights Feasible?},
  author={Zhang, Shun and Wang, Xueliang and Zhang, Zhen and Liu, Yong and Xiong, Hui and Zhou, Xingshe},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
