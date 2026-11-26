# Forward Pass and Backpropagation — From First Principles

Implementation of the full neural-network training loop — **forward pass → loss → backward pass → parameter update** — at three levels of abstraction: analytical derivation by hand, a from-scratch NumPy MLP, and PyTorch autograd.

## Contents

| Part | Topic | Key result |
|---|---|---|
| 1 | Manual derivation of a 2-2-1 ReLU regression network | ŷ = 3.0, L = 1.0; dead ReLU unit identified |
| 2 | NumPy MLP for XOR with numerical gradient check | Max abs gradient error ≈ 2.6e-11 |
| 3 | PyTorch autograd validation and penguin classifier | Autograd matches manual gradients; 99.03% test accuracy |
| 4 | Applied engineering scenarios | Linear collapse, dying ReLU, activation caching, feature scaling, divergence |
| Bonus | 3-class MLP in pure NumPy | 99.03% test accuracy (target > 95%) |

## Results

### Gradient verification

The analytical backward pass was validated two ways: against a central-difference numerical gradient (max abs difference 2.6e-11) and against PyTorch autograd (exact match on all parameters).

### XOR — NumPy MLP

| Experiment | Finding |
|---|---|
| Learning rate (2,000 epochs) | lr = 0.1 fails to converge; lr ≥ 0.5 solves XOR |
| Hidden width (5,000 epochs) | n_h = 1 collapses to OR; **n_h ≥ 2** required |
| ReLU vs. sigmoid | ReLU reaches ~10× lower loss at every learning rate; sensitive to initialization (dead units) |

### Palmer Penguins — PyTorch (test accuracy)

| hidden \ lr | 0.01 | 0.1 | 1.0 |
|---|---|---|---|
| 2 | 0.4369 | 0.8155 | 0.9806 |
| 8 | 0.7961 | **0.9903** | 0.9903 |
| 32 | 0.8641 | 0.9903 | 0.9903 |

Selected configuration: **8 hidden units, lr = 0.1** — same accuracy as the 32-unit model with 4× fewer parameters in the hidden layer.

An ablation removing `optimizer.zero_grad()` shows gradient accumulation: training loss collapses to zero while the model becomes over-confident, despite unchanged test accuracy on this split.

## Repository structure

```
ML-Homework-53/
├── ML_Homework_53.ipynb    # Full solution with executed outputs
├── images/                 # Handwritten derivation for Part 1
├── requirements.txt
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook ML_Homework_53.ipynb
```

All experiments use fixed seeds (`numpy` seed 1 / 42, `torch.manual_seed(42)`, `random_state=42`), so results are deterministic.

## Stack

Python · NumPy · PyTorch · scikit-learn · seaborn
