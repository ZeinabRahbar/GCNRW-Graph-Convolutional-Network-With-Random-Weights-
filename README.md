This Jupyter notebook implements a **Graph Convolutional Network with Random Weights (GCNRW)**. This approach is similar to Extreme Learning Machines (ELM) but applied to graph-structured data, where the hidden layer weights are fixed and randomly initialized, and only the output weights are analytically computed.

---

## ## Project Overview

The **GCNRW** model leverages the structural information of a graph by using an adjacency matrix and node features. It employs a closed-form solution for training, which significantly reduces training time compared to traditional backpropagation-based Graph Convolutional Networks (GCNs).

### ### Key Features

* **Random Weights:** The hidden layer projection uses a uniform random distribution, avoiding expensive iterative training for the internal parameters.
* **Analytical Training:** The output weights are calculated using the Moore-Penrose pseudoinverse with Tikhonov regularization ().
* **Graph Smoothing:** The model utilizes  (the squared adjacency matrix with self-loops) to aggregate information from two-hop neighborhoods.

---

## ## Technical Implementation

### ### Model Architecture

The model is defined in the `GCNRW` class with the following logic:

1. **Adjacency Processing:** Adds self-loops and computes  for deeper spatial feature aggregation.
2. **Hidden Layer:** Computes , where  is random.
3. **Output Weights ():** Solved using the regularized least squares formula:
* If : 
* If : 



### ### Dependencies

* `torch` & `torch_geometric`
* `scikit-learn`
* `numpy` / `scipy`

---

## ## Performance Benchmarks

The notebook evaluates the model on three standard citation datasets from `Planetoid`.

| Dataset | Hidden Units | Lambda () | Train Acc | Test Acc |
| --- | --- | --- | --- | --- |
| **Cora** | 8,000 | 15 | 98.15% | 79.33% |
| **Citeseer** | 10,000 | 100 | 98.40% | 71.25% |
| **PubMed** | 7,000 | 10 | 89.65% | 80.24% |

---

## ## Usage

1. **Install Requirements:** Run `!pip install torch_geometric`.
2. **Initialize Model:** ```python
model = GCNRW(n_features=num_features, n_classes=num_classes, n_hidden=8000, lambda_=15)
```

```


3. **Train (Fit):**
```python
model.fit(A_train, X_train, Y_train)

```


4. **Predict:**
```python
predictions = model.predict(A_test, X_test)
```


```

Would you like me to help you refactor the code into a standalone Python script or perhaps optimize the memory
