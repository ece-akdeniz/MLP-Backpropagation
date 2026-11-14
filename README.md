# Multilayer Perceptron from Scratch

A two-layer multilayer perceptron with explicit forward and backward passes, implemented using PyTorch tensor operations for the AIN3002 deep learning course.

The goal is to understand backpropagation directly. The implementation calculates parameter gradients itself; the accompanying scripts compare those gradients with PyTorch autograd.

## What is implemented

- Two configurable fully connected layers.
- ReLU, sigmoid, and identity activations.
- Mean squared error, binary cross entropy, and multiclass cross entropy, including derivatives.
- Cached intermediate values, explicit gradient storage, and a method to clear gradients and cached tensors.

## Quick start

Create a fresh Python environment and install PyTorch:

```bash
git clone https://github.com/ece-akdeniz/deep-learning-assignments.git
cd deep-learning-assignments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Run the four gradient comparison scripts:

```bash
python test1.py
python test2.py
python test3.py
python test4.py
```

Each script prints four Boolean tensor values, one for each weight or bias gradient. A value of `True` means its difference from the autograd reference has norm below `1e-3`. These scripts print comparisons rather than raising test assertions.

## Files

- [mlp.py](mlp.py): network, activation handling, manual backpropagation, and loss functions.
- [test1.py](test1.py): ReLU / identity network with mean squared error.
- [test2.py](test2.py): sigmoid / sigmoid network with binary cross entropy.
- [test3.py](test3.py): ReLU / ReLU network with mean squared error.
- [test4.py](test4.py): ReLU / identity network with multiclass cross entropy.
- [assignment1.pdf](assignment1.pdf): assignment specification.

## Scope

This is an educational gradient implementation. It does not include an optimizer, a dataset loader, a complete training loop, or benchmark results. Binary cross entropy operates on probabilities directly and does not clamp them away from zero and one.

## Author

[Ece Akdeniz](https://github.com/ece-akdeniz)
