# nnfc

A neural network framework built from scratch in C++ with zero external dependencies.

Everything lives in a single file (`src/main.cpp`): a custom garbage collector for tensor memory, matrix operations, layers (Linear, ReLU, Sigmoid), backpropagation, and SGD — all implemented from the ground up using only the C++ standard library.

## Features

- **Custom memory management** — arena-style garbage collector for tensor allocations
- **Matrix operations** — dot product, transpose, element-wise ops, bias addition
- **Layer types** — Linear (with bias), ReLU, Sigmoid
- **Training** — forward/backward pass, MSE and BCE loss, SGD optimizer
- **Flexible architecture** — compose arbitrary layer stacks via `LayerSpec`

## Build & Run

```bash
make
./build/main
```

## Example

The included example learns `y = 1.7*x1 + x2` using a small MLP (2 → 3 → 3 → 3 → 1) trained for 100k steps.
