import triton
import triton.language as tl

import torch

@triton.jit
def mlp_forward_kernel(
    X, W1, B1, W2, B2, A1, MASK, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input
    x = tl.load(X + offsets[:, None] * K + tl.arange(0, K)[None, :], mask=offsets[:, None] < M)  # Adjust mask as needed
    
    # First layer
    w1 = tl.load(W1 + tl.arange(0, K)[:, None] * M + tl.arange(0, M)[None, :])
    b1 = tl.load(B1 + tl.arange(0, M))
    y1 = tl.dot(x, w1) + b1
    
    # ReLU and mask
    mask = y1 > 0
    a1 = tl.where(mask, y1, 0.0)
    tl.store(A1 + offsets[:, None] * M + tl.arange(0, M)[None, :], a1)
    tl.store(MASK + offsets[:, None] * M + tl.arange(0, M)[None, :], mask)
    
    # Second layer
    w2 = tl.load(W2 + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    b2 = tl.load(B2 + tl.arange(0, N))
    y2 = tl.dot(a1, w2) + b2
    tl.store(Y + offsets[:, None] * N + tl.arange(0, N)[None, :], y2)

def mlp_forward(X, W1, B1, W2, B2):
    B, K = X.shape
    M, N = W1.shape[1], W2.shape[1]
    A1 = torch.empty((B, M), device='cuda')
    MASK = torch.empty((B, M), device='cuda', dtype=torch.bool)
    Y = torch.empty((B, N), device='cuda')
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE']), )
    mlp_forward_kernel[grid](X, W1, B1, W2, B2, A1, MASK, Y, M, N, K, BLOCK_SIZE=32)
    return Y, A1, MASK



# Backward pass kernels with proper tiling
@triton.jit
def compute_dW2_kernel(
    A1, dY, dW2,
    B, M, N,
    stride_am, stride_ak,
    stride_dyn, stride_dyk,
    stride_dwm, stride_dwn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A1 + (offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am)
    dy_ptrs = dY + (offs_k[:, None] * stride_dyk + offs_n[None, :] * stride_dyn)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(B, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[:, None] < B, other=0.0)
        dy = tl.load(dy_ptrs, mask=offs_k[:, None] < B, other=0.0)
        acc += tl.dot(a, dy, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        dy_ptrs += BLOCK_SIZE_K * stride_dyk
    
    tl.store(dW2 + offs_m[:, None] * stride_dwm + offs_n[None, :] * stride_dwn, acc)

@triton.jit
def compute_dW1_kernel(
    X, dy1, dW1,
    B, K, M,
    stride_xk, stride_xb,
    stride_dy1m, stride_dy1b,
    stride_dwk, stride_dwm,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    x_ptrs = X + (offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk)
    dy_ptrs = dy1 + (offs_b[:, None] * stride_dy1b + offs_m[None, :] * stride_dy1m)
    
    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), dtype=tl.float32)
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        x = tl.load(x_ptrs, mask=offs_b[:, None] < B, other=0.0)
        dy = tl.load(dy_ptrs, mask=offs_b[:, None] < B, other=0.0)
        acc += tl.dot(x, dy, allow_tf32=False)
        x_ptrs += BLOCK_SIZE_B * stride_xb
        dy_ptrs += BLOCK_SIZE_B * stride_dy1b
    
    tl.store(dW1 + offs_k[:, None] * stride_dwk + offs_m[None, :] * stride_dwm, acc)

def mlp_backward(dY, X, A1, MASK, W1, W2):
    B, K = X.shape
    M, N = W1.shape[1], W2.shape[1]
    
    # Compute dW2 with proper block sizes
    dW2 = torch.zeros_like(W2)
    grid = lambda meta: (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    compute_dW2_kernel[grid](
        A1, dY, dW2, B, M, N,
        A1.stride(1), A1.stride(0),
        dY.stride(1), dY.stride(0),
        dW2.stride(1), dW2.stride(0),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )
    
    # Compute dB2
    dB2 = torch.sum(dY, dim=0)
    
    # Compute dW1 with proper block sizes
    dW1 = torch.zeros_like(W1)
    grid = lambda meta: (triton.cdiv(K, 32) * triton.cdiv(M, 32),)
    compute_dW1_kernel[grid](
        X, (A1 * MASK), dW1, B, K, M,
        X.stride(1), X.stride(0),
        (A1 * MASK).stride(1), (A1 * MASK).stride(0),
        dW1.stride(1), dW1.stride(0),
        BLOCK_SIZE_K=32, BLOCK_SIZE_M=32, BLOCK_SIZE_B=32
    )
    
    # Remaining calculations stay the same
    dB1 = torch.sum(A1 * MASK, dim=0)
    dX = (A1 * MASK) @ W1.T
    
    return dX, dW1, dB1, dW2, dB2

if __name__ == "__main__":
    # Configuration
    M, N, K = 64, 64, 256
    BLOCK_SIZE, epochs, lr = 32, 100, 0.01
    

    scale_factor = 0.1


    # Initialize parameters
    W1 = scale_factor * torch.randn(K, M, device='cuda', requires_grad=False)
    B1 = torch.zeros(M, device='cuda', requires_grad=False)
    W2 = scale_factor * torch.randn(M, N, device='cuda', requires_grad=False)
    B2 = torch.zeros(N, device='cuda', requires_grad=False)
    
    # Generate data
    X = scale_factor * torch.randn(BLOCK_SIZE, K, device='cuda')
    Y_true = scale_factor * torch.randn(BLOCK_SIZE, N, device='cuda')
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        Y_pred, A1, MASK = mlp_forward(X, W1, B1, W2, B2)
        
        # Compute loss
        loss = torch.mean((Y_pred - Y_true) ** 2)
        
        # Backward pass
        dY = 2 * (Y_pred - Y_true) / BLOCK_SIZE
        dX, dW1, dB1, dW2, dB2 = mlp_backward(dY, X, A1, MASK, W1, W2)
        
        # Update parameters
        W1 -= lr * dW1
        B1 -= lr * dB1
        W2 -= lr * dW2
        B2 -= lr * dB2
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
