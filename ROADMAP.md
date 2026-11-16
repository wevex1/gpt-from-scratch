# Roadmap

## Layer 1: Autograd (NumPy)
- [ ] Tensor/Value class holding data and grad
- [ ] Forward ops: add, mul, pow, neg, sub, div
- [ ] Backward pass with topological ordering
- [ ] Broadcasting-aware gradients
- [ ] Matmul, sum, mean, reshape, transpose
- [ ] Activations: relu, tanh, exp, log
- [ ] Gradient check against numerical gradients
- [ ] Layer 1 quiz
- [ ] Layer 1 rebuild from memory

## Layer 2: NN library (NumPy)
- [ ] Module base class and parameters
- [ ] Linear layer
- [ ] Loss functions: MSE, cross-entropy (stable softmax)
- [ ] Optimizers: SGD, Adam
- [ ] Training loop on a toy dataset (e.g. XOR, MNIST subset)
- [ ] Layer 2 quiz
- [ ] Layer 2 rebuild from memory

## Layer 3: CNN (PyTorch)
- [ ] Port core pieces to PyTorch tensors
- [ ] Conv2d, pooling
- [ ] Train a small CNN on MNIST
- [ ] Layer 3 quiz
- [ ] Layer 3 rebuild from memory

## Layer 4: Transformer / GPT (PyTorch)
- [ ] Data and character-level tokenizer
- [ ] Batching: context windows and shifted targets
- [ ] Bigram baseline and training loop
- [ ] Self-attention with causal mask
- [ ] Multi-head attention, feed-forward, residuals, LayerNorm
- [ ] Full GPT block stack
- [ ] Generation with temperature and top-k
- [ ] Optional: BPE tokenizer
- [ ] Layer 4 quiz
- [ ] Layer 4 rebuild from memory
