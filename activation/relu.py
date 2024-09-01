import triton
import triton.language as tl
import pdb
import torch

def next_power_of_2(n):
    if n <= 0:
        return 1
    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n
    # Find the next power of 2
    power = 1
    while power < n:
        power <<= 1
    return power

@triton.jit
def relu_kernel(x_ptr, out_ptr, N: tl.constexpr, block_size: tl.constexpr):
    #pdb.set_trace()

    # Get the index of the current thread
    pid = tl.program_id(0)
    block_start = pid*block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    # Load elements from global memory
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute linear layer
    result = tl.where(x >= 0, x, 0.0)

    # Write result to global memory
    if pid == 0:
       tl.store(out_ptr+offsets, result, mask=mask)

def relu(x):
    # Prepare output tensor
    out = torch.empty_like(x, dtype=torch.float32, device=x.device)
    N = out.numel()

    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # Calculate the number of blocks needed
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE), )

    relu_kernel[grid](x, out, N, BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    # Define two 1D tensors
    x = torch.tensor([1.0, -1.0, 0, 3.0], dtype=torch.float32)

    # Compute the relu layer using Triton kernel
    output_triton = relu(x)
    print(f"Triton result: {output_triton}")

    output_torch = torch.relu(x)

    print(f"Torch result: {output_torch}")
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

