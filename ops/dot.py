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
def dot_product_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, block_size: tl.constexpr):
    #pdb.set_trace()

    # Get the index of the current thread
    pid = tl.program_id(0)
    block_start = pid*block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    # Load elements from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute dot product
    result = tl.sum(x * y, axis=0)

    # Write result to global memory
    if pid == 0:
       tl.store(out_ptr, result)

def dot_product(x, y):
    # Ensure x and y are 1D tensors
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Both input tensors must be 1-dimensional")
    
    if x.size(0) != y.size(0):
        raise ValueError("Input tensors must be of the same size")

    N = next_power_of_2(x.size(0))
    block_size = 1024

    # Prepare output tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)
    
    # Launch Triton kernel
    grid = (1,)
    print(x,y)

    dot_product_kernel[grid](x, y, out, N, block_size)
    
    return out.item()

# Example usage
if __name__ == "__main__":
    # Define two 1D tensors
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

    # Compute the dot product using Triton kernel
    output_triton = dot_product(x, y)
    print(f"Triton result: {output_triton}")

    output_torch = torch.dot(x,y)

    print(f"Torch result: {output_torch}")
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

