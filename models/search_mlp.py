import triton
import triton.language as tl
import torch
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class KernelConfig:
    block_size: int
    num_warps: int
    num_stages: int
    performance: float = 0.0  # GFLOPS
    
    def __hash__(self):
        return hash((self.block_size, self.num_warps, self.num_stages))

# Fused kernel: performs matmul + bias + relu in one kernel
@triton.jit
def fused_linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused Linear + ReLU kernel
    out = relu(x @ weight.T + bias)
    x: (batch_size, in_features)
    weight: (out_features, in_features) 
    bias: (out_features,)
    out: (batch_size, out_features)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    x_ptrs = x_ptr + offs_m[:, None] * in_features + offs_k[None, :]
    w_ptrs = weight_ptr + offs_n[:, None] * in_features + offs_k[None, :]
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, in_features, BLOCK_K):
        mask_m = offs_m[:, None] < batch_size
        mask_n = offs_n[:, None] < out_features
        mask_k = (k + offs_k[None, :]) < in_features
        
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)
        w = tl.load(w_ptrs, mask=mask_n & mask_k, other=0.0)
        
        acc += tl.dot(x, tl.trans(w))
        
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    mask_n = offs_n < out_features
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    
    # ReLU
    acc = tl.where(acc >= 0, acc, 0.0)
    
    # Store result
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_out_m[:, None] * out_features + offs_out_n[None, :]
    mask = (offs_out_m[:, None] < batch_size) & (offs_out_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def fused_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused Linear kernel (no activation)
    out = x @ weight.T + bias
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * in_features + offs_k[None, :]
    w_ptrs = weight_ptr + offs_n[:, None] * in_features + offs_k[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_K):
        mask_m = offs_m[:, None] < batch_size
        mask_n = offs_n[:, None] < out_features
        mask_k = (k + offs_k[None, :]) < in_features
        
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)
        w = tl.load(w_ptrs, mask=mask_n & mask_k, other=0.0)
        
        acc += tl.dot(x, tl.trans(w))
        
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    
    bias_ptrs = bias_ptr + offs_n
    mask_n = offs_n < out_features
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_out_m[:, None] * out_features + offs_out_n[None, :]
    mask = (offs_out_m[:, None] < batch_size) & (offs_out_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=mask)


class MLPModel(torch.nn.Module):
    """PyTorch MLP for comparison"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TritonMLPOptimizer:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, device='cuda'):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.best_config = None
        self.all_configs = []
        
    def fused_mlp_forward(self, x, w1, b1, w2, b2, w3, b3, config):
        """Forward pass using fused Triton kernels"""
        batch_size = x.shape[0]
        
        # Layer 1: Linear + ReLU
        h1 = torch.empty((batch_size, self.hidden_dim1), dtype=torch.float32, device=x.device)
        grid = lambda meta: (
            triton.cdiv(batch_size, config.block_size),
            triton.cdiv(self.hidden_dim1, config.block_size)
        )
        fused_linear_relu_kernel[grid](
            x, w1, b1, h1,
            batch_size, self.input_dim, self.hidden_dim1,
            BLOCK_M=config.block_size, BLOCK_N=config.block_size, BLOCK_K=config.block_size,
            num_warps=config.num_warps, num_stages=config.num_stages
        )
        
        # Layer 2: Linear + ReLU
        h2 = torch.empty((batch_size, self.hidden_dim2), dtype=torch.float32, device=x.device)
        grid = lambda meta: (
            triton.cdiv(batch_size, config.block_size),
            triton.cdiv(self.hidden_dim2, config.block_size)
        )
        fused_linear_relu_kernel[grid](
            h1, w2, b2, h2,
            batch_size, self.hidden_dim1, self.hidden_dim2,
            BLOCK_M=config.block_size, BLOCK_N=config.block_size, BLOCK_K=config.block_size,
            num_warps=config.num_warps, num_stages=config.num_stages
        )
        
        # Layer 3: Linear (no activation)
        out = torch.empty((batch_size, self.output_dim), dtype=torch.float32, device=x.device)
        grid = lambda meta: (
            triton.cdiv(batch_size, config.block_size),
            triton.cdiv(self.output_dim, config.block_size)
        )
        fused_linear_kernel[grid](
            h2, w3, b3, out,
            batch_size, self.hidden_dim2, self.output_dim,
            BLOCK_M=config.block_size, BLOCK_N=config.block_size, BLOCK_K=config.block_size,
            num_warps=config.num_warps, num_stages=config.num_stages
        )
        
        return out
    
    def benchmark_config(self, config: KernelConfig, x: torch.Tensor, 
                        w1, b1, w2, b2, w3, b3, 
                        num_iterations=50, warmup=5):
        """Benchmark a specific kernel configuration"""
        # Warmup
        for _ in range(warmup):
            _ = self.fused_mlp_forward(x, w1, b1, w2, b2, w3, b3, config)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            out = self.fused_mlp_forward(x, w1, b1, w2, b2, w3, b3, config)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed_time = (end - start) / num_iterations
        
        # Calculate GFLOPS
        batch_size = x.shape[0]
        flops_per_sample = (2 * self.input_dim * self.hidden_dim1 + 
                           2 * self.hidden_dim1 * self.hidden_dim2 + 
                           2 * self.hidden_dim2 * self.output_dim)
        total_flops = flops_per_sample * batch_size
        gflops = total_flops / (elapsed_time * 1e9)
        
        return gflops, out
    
    def grid_search(self, x, w1, b1, w2, b2, w3, b3, verbose=True):
        """Grid search over possible configurations"""
        block_sizes = [32, 64, 128]
        num_warps_options = [2, 4, 8]
        num_stages_options = [2, 3, 4]
        
        best_perf = 0
        best_config = None
        
        if verbose:
            print("\n=== GRID SEARCH OPTIMIZATION ===")
            print(f"Total configurations to test: {len(block_sizes) * len(num_warps_options) * len(num_stages_options)}")
        
        for bs in block_sizes:
            for nw in num_warps_options:
                for ns in num_stages_options:
                    try:
                        config = KernelConfig(block_size=bs, num_warps=nw, num_stages=ns)
                        perf, _ = self.benchmark_config(config, x, w1, b1, w2, b2, w3, b3)
                        config.performance = perf
                        self.all_configs.append(config)
                        
                        if verbose:
                            print(f"Config: BS={bs:4d}, Warps={nw}, Stages={ns} -> {perf:.2f} GFLOPS")
                        
                        if perf > best_perf:
                            best_perf = perf
                            best_config = config
                    except Exception as e:
                        if verbose:
                            print(f"Config: BS={bs:4d}, Warps={nw}, Stages={ns} -> FAILED ({str(e)[:80]})")
        
        self.best_config = best_config
        if verbose and best_config:
            print(f"\nBest Grid Search Config: BS={best_config.block_size}, Warps={best_config.num_warps}, Stages={best_config.num_stages}")
            print(f"Best Performance: {best_perf:.2f} GFLOPS")
        
        return best_config
    
    def genetic_algorithm(self, x, w1, b1, w2, b2, w3, b3, 
                         population_size=12, generations=5, 
                         mutation_rate=0.3, verbose=True):
        """Genetic algorithm to find optimal configuration"""
        block_sizes = [32, 64, 128]
        num_warps_options = [2, 4, 8]
        num_stages_options = [2, 3, 4]
        
        def create_random_config():
            return KernelConfig(
                block_size=random.choice(block_sizes),
                num_warps=random.choice(num_warps_options),
                num_stages=random.choice(num_stages_options)
            )
        
        def mutate(config: KernelConfig):
            new_config = KernelConfig(
                block_size=config.block_size,
                num_warps=config.num_warps,
                num_stages=config.num_stages
            )
            if random.random() < mutation_rate:
                new_config.block_size = random.choice(block_sizes)
            if random.random() < mutation_rate:
                new_config.num_warps = random.choice(num_warps_options)
            if random.random() < mutation_rate:
                new_config.num_stages = random.choice(num_stages_options)
            return new_config
        
        def crossover(parent1: KernelConfig, parent2: KernelConfig):
            return KernelConfig(
                block_size=random.choice([parent1.block_size, parent2.block_size]),
                num_warps=random.choice([parent1.num_warps, parent2.num_warps]),
                num_stages=random.choice([parent1.num_stages, parent2.num_stages])
            )
        
        if verbose:
            print("\n=== GENETIC ALGORITHM OPTIMIZATION ===")
            print(f"Population size: {population_size}, Generations: {generations}")
        
        population = [create_random_config() for _ in range(population_size)]
        
        best_ever = None
        best_ever_perf = 0
        
        for gen in range(generations):
            if verbose:
                print(f"\nGeneration {gen + 1}/{generations}")
            
            for config in population:
                if config.performance == 0:
                    try:
                        perf, _ = self.benchmark_config(config, x, w1, b1, w2, b2, w3, b3, num_iterations=30)
                        config.performance = perf
                    except:
                        config.performance = 0.0
            
            population.sort(key=lambda c: c.performance, reverse=True)
            
            if verbose and len(population) > 0 and population[0].performance > 0:
                print(f"Best in generation: BS={population[0].block_size}, Warps={population[0].num_warps}, Stages={population[0].num_stages} -> {population[0].performance:.2f} GFLOPS")
            
            if len(population) > 0 and population[0].performance > best_ever_perf:
                best_ever = population[0]
                best_ever_perf = population[0].performance
            
            elite_size = max(2, population_size // 4)
            elite = [p for p in population if p.performance > 0][:elite_size]
            
            if len(elite) == 0:
                elite = [create_random_config()]
            
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        if verbose and best_ever:
            print(f"\nBest GA Config: BS={best_ever.block_size}, Warps={best_ever.num_warps}, Stages={best_ever.num_stages}")
            print(f"Best Performance: {best_ever_perf:.2f} GFLOPS")
        
        return best_ever


def benchmark_pytorch_mlp(model, x, num_iterations=50, warmup=5):
    """Benchmark PyTorch MLP"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            out = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
    
    elapsed_time = (end - start) / num_iterations
    
    # Calculate GFLOPS
    batch_size = x.shape[0]
    input_dim = model.fc1.in_features
    hidden_dim1 = model.fc1.out_features
    hidden_dim2 = model.fc2.out_features
    output_dim = model.fc3.out_features
    
    flops_per_sample = (2 * input_dim * hidden_dim1 + 
                       2 * hidden_dim1 * hidden_dim2 + 
                       2 * hidden_dim2 * output_dim)
    total_flops = flops_per_sample * batch_size
    gflops = total_flops / (elapsed_time * 1e9)
    
    return gflops, out


def compare_mlp_implementations(x, model, optimizer, w1, b1, w2, b2, w3, b3):
    """Compare all implementations"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON - FUSED MLP")
    print("="*70)
    
    # PyTorch baseline
    print("\nBenchmarking PyTorch MLP...")
    pytorch_gflops, pytorch_out = benchmark_pytorch_mlp(model, x)
    print(f"PyTorch Performance: {pytorch_gflops:.2f} GFLOPS")
    
    # Grid search optimized
    print("\nRunning Grid Search Optimization...")
    gs_config = optimizer.grid_search(x, w1, b1, w2, b2, w3, b3, verbose=True)
    
    gs_out = None
    if gs_config:
        gs_gflops = gs_config.performance
        print(f"\nGrid Search Performance: {gs_gflops:.2f} GFLOPS")
        print(f"Speedup vs PyTorch: {gs_gflops/pytorch_gflops:.2f}x")
        gs_out = optimizer.fused_mlp_forward(x, w1, b1, w2, b2, w3, b3, gs_config)
    
    # Genetic algorithm optimized
    print("\nRunning Genetic Algorithm Optimization...")
    ga_config = optimizer.genetic_algorithm(x, w1, b1, w2, b2, w3, b3, verbose=True)
    
    ga_out = None
    if ga_config:
        ga_gflops = ga_config.performance
        print(f"\nGenetic Algorithm Performance: {ga_gflops:.2f} GFLOPS")
        print(f"Speedup vs PyTorch: {ga_gflops/pytorch_gflops:.2f}x")
        ga_out = optimizer.fused_mlp_forward(x, w1, b1, w2, b2, w3, b3, ga_config)
    
    # Verification
    print("\n" + "="*70)
    print("CORRECTNESS VERIFICATION")
    print("="*70)
    if gs_out is not None:
        print(f"Max diff (Grid Search vs PyTorch): {torch.max(torch.abs(gs_out - pytorch_out)):.6e}")
        print(f"Mean diff (Grid Search vs PyTorch): {torch.mean(torch.abs(gs_out - pytorch_out)):.6e}")
    if ga_out is not None:
        print(f"Max diff (GA vs PyTorch): {torch.max(torch.abs(ga_out - pytorch_out)):.6e}")
        print(f"Mean diff (GA vs PyTorch): {torch.mean(torch.abs(ga_out - pytorch_out)):.6e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Method':<30} {'Performance (GFLOPS)':<25} {'Speedup vs PyTorch':<15}")
    print("-"*70)
    print(f"{'PyTorch':<30} {pytorch_gflops:<25.2f} {1.0:<15.2f}")
    if gs_config:
        print(f"{'Triton (Grid Search)':<30} {gs_config.performance:<25.2f} {gs_config.performance/pytorch_gflops:<15.2f}")
    if ga_config:
        print(f"{'Triton (Genetic Algorithm)':<30} {ga_config.performance:<25.2f} {ga_config.performance/pytorch_gflops:<15.2f}")
    print("="*70)


if __name__ == "__main__":
    print("Fused MLP Triton Kernel Auto-Optimization")
    print("="*70)
    
    # Network architecture
    input_dim = 128
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 64
    
    print(f"\nMLP Architecture: {input_dim} -> {hidden_dim1} -> {hidden_dim2} -> {output_dim}")
    print("\nNote: Fused kernels combine Linear + ReLU operations to reduce memory transfers")
    
    # Test with different batch sizes
    batch_sizes = [64, 512, 2048]
    
    for batch_size in batch_sizes:
        print(f"\n\n{'='*70}")
        print(f"Testing with batch size: {batch_size}")
        print(f"{'='*70}")
        
        # Create test data
        x = torch.randn(batch_size, input_dim, dtype=torch.float32, device='cuda')
        
        # Create PyTorch model
        model = MLPModel(input_dim, hidden_dim1, hidden_dim2, output_dim).cuda()
        
        # Extract weights for Triton kernel
        w1 = model.fc1.weight.contiguous()  # (hidden_dim1, input_dim)
        b1 = model.fc1.bias.contiguous()
        w2 = model.fc2.weight.contiguous()  # (hidden_dim2, hidden_dim1)
        b2 = model.fc2.bias.contiguous()
        w3 = model.fc3.weight.contiguous()  # (output_dim, hidden_dim2)
        b3 = model.fc3.bias.contiguous()
        
        # Initialize optimizer
        optimizer = TritonMLPOptimizer(input_dim, hidden_dim1, hidden_dim2, output_dim, device='cuda')
        
        # Compare implementations
        compare_mlp_implementations(x, model, optimizer, w1, b1, w2, b2, w3, b3)
    
    print("\n\n" + "="*70)
    print("Optimization complete!")
    print("="*70)
