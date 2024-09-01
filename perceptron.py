import triton
import triton.language as tl

import torch

@triton.jit
def mlp_kernel(X, W1, B1, W2, B2, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Load input
    x = tl.load(X + block_start * K + tl.arange(0, BLOCK_SIZE)[:, None] * K + tl.arange(0, K)[None, :])

    # First layer: X @ W1 + B1
    w1 = tl.load(W1 + tl.arange(0, K)[:, None] * M + tl.arange(0, M)[None, :])
    b1 = tl.load(B1 + tl.arange(0, M))
    y1 = tl.dot(x, w1) + b1

    # Activation function (ReLU)
    y1 = tl.where(y1 > 0, y1, 0)

    # Second layer: y1 @ W2 + B2
    w2 = tl.load(W2 + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    b2 = tl.load(B2 + tl.arange(0, N))
    y2 = tl.dot(y1, w2) + b2

    # Store output
    tl.store(Y + block_start * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, N)[None, :], y2)

def mlp(X, W1, B1, W2, B2):
    Y = torch.empty((BLOCK_SIZE, N), dtype=torch.float32)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE), )

    mlp_kernel[grid](X, W1, B1, W2, B2, Y, M,N,K,BLOCK_SIZE)

if __name__ == "__main__":
    # Parameters
    M = 128  # Size of the first hidden layer
    N = 64   # Size of the output layer
    K = 256  # Size of the input layer
    BLOCK_SIZE = 128

    # Allocate memory for inputs and outputs
    X = torch.rand((BLOCK_SIZE, K), dtype=torch.float32)
    W1 = torch.rand((K, M), dtype=torch.float32)
    B1 = torch.rand((M,), dtype=torch.float32)
    W2 = torch.rand((M, N), dtype=torch.float32)
    B2 = torch.rand((N,), dtype=torch.float32)

    mlp(X,W1,B1,W2,B2)