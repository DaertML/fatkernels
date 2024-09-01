import triton
import triton.language as tl
import pdb
import torch

@triton.jit
def dot_product_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    #pdb.set_trace()

    # Get the index of the current thread
    pid = tl.program_id(0)
    
    # Compute the index for the elements
    index = pid * 1 + tl.arange(0, 1)
    
    # Load elements from global memory
    x = tl.load(x_ptr + index)
    y = tl.load(y_ptr + index)
    
    print(x,y)
    print()

    # Compute dot product
    result = tl.sum(x * y, axis=0)
    print(result)
    # Write result to global memory
    tl.store(out_ptr, result)

def dot_product(x, y):
    # Ensure x and y are 1D tensors
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Both input tensors must be 1-dimensional")
    
    if x.size(0) != y.size(0):
        raise ValueError("Input tensors must be of the same size")

    N = x.size(0)

    # Prepare output tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)
    
    # Launch Triton kernel
    grid = (1,)
    print(x,y)

    dot_product_kernel[grid](x, y, out, N)
    
    return out.item()

# Example usage
if __name__ == "__main__":
    # Define two 1D tensors
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

    # Compute the dot product using Triton kernel
    result = dot_product(x, y)
    print(f"Dot product result: {result}")

    torchres = torch.dot(x,y)
    print(torchres)
