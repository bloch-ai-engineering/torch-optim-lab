# Torch Optim Lab

**Optimization from first principles in PyTorch: logistic regression trained with hand-written SGD, numerically stable losses, L1 sparsity, and GD / Momentum / AdaGrad / Adam implemented from scratch and raced on convex and non-convex surfaces.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-ee4c2c) ![Dataset](https://img.shields.io/badge/data-SST--2-yellow)

## What's inside

### Part 1: Logistic regression from scratch on SST-2 sentiment
- **Text pipeline:** cleaning, tokenization and a 10k-word Bag-of-Words vocabulary built on the training split only.
- **Numerical stability:** binary vs. softmax cross-entropy, with a stable log-sum-exp loss so `exp` can't overflow.
- **Model and training loop:** a `LogisticRegression` module built from raw tensors (zero and random init), trained with a hand-written mini-batch SGD loop that logs train and validation loss and accuracy each epoch.
- **Hyperparameter grid:** 5 learning rates × 3 batch sizes, shown as heatmaps of final train and validation accuracy.
- **L1 regularization:** a λ sweep from 0 to 0.1 that tracks accuracy, loss and weight sparsity.

### Part 2: Optimizers on easy vs. hard landscapes
GD, Momentum, AdaGrad and Adam, implemented by hand and run on:
- **Convex bowl:** f(x, y) = x² + 4y²
- **Six-hump camel function:** non-convex, with 2 global and 4 local minima

The notebook plots loss curves and 2-D optimization trajectories for each optimizer.

### Bonus
A short derivation of why L1 drives weights to exactly zero, and how proximal gradient descent (soft-thresholding) fixes plain SGD's failure to do so.

## Key findings
- **Learning rate vs. batch size:** a higher learning rate converges faster but becomes unstable. Smaller batches give more updates per epoch at the cost of noisier convergence.
- **Initialization:** zero and random init reach the same optimum, as expected for a convex loss.
- **L1 on SST-2:** increasing λ monotonically *hurt* accuracy. The BoW model wasn't overfitting, so the penalty only removed signal. Plain SGD also doesn't produce exact zeros, which motivates the proximal step.
- **Optimizers:** all four reach (0, 0) on the bowl. On the camel function, GD, Momentum and Adam fall into the same local minimum from (−2, −1.5). Geometry and the starting point matter more than the optimizer.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/optimization_and_regularization.ipynb
```

The notebook was developed on Google Colab and runs on CPU.

## Context

This was homework 1 of the **LLM Architectures** module in **Nebius Academy: AI Performance Engineering**. The task formulations are kept inline in the notebook, as the course required.

## Authors

**Nathanael Bloch** · **Tomer Porat**
