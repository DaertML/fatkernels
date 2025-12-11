import triton
import triton.language as tl
import torch
import time
import json
from pathlib import Path
from transformers import AutoTokenizer
from dataclasses import dataclass
import random

@dataclass
class KernelConfig:
    block_size_m: int
    block_size_n: int
    block_size_k: int
    num_warps: int
    num_stages: int
    performance: float = 0.0
    
    def __hash__(self):
        return hash((self.block_size_m, self.block_size_n, self.block_size_k, 
                    self.num_warps, self.num_stages))


@triton.jit
def fused_linear_kernel(
    input_ptr, weight_ptr, output_ptr,
    n_rows, in_features, out_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Fused matrix multiplication kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    input_ptrs = input_ptr + offs_m[:, None] * in_features + offs_k[None, :]
    weight_ptrs = weight_ptr + offs_n[:, None] * in_features + offs_k[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_K):
        mask_m = offs_m[:, None] < n_rows
        mask_n = offs_n[:, None] < out_features
        mask_k = (k + offs_k[None, :]) < in_features
        
        inp = tl.load(input_ptrs, mask=mask_m & mask_k, other=0.0)
        w = tl.load(weight_ptrs, mask=mask_n & mask_k, other=0.0)
        
        acc += tl.dot(inp, tl.trans(w))
        
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K
    
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = output_ptr + offs_out_m[:, None] * out_features + offs_out_n[None, :]
    mask = (offs_out_m[:, None] < n_rows) & (offs_out_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=mask)


class LlamaTritonOptimizer:
    def __init__(self, model_path="./llama32", device='cuda'):
        self.device = device
        self.model_path = model_path
        
        # Check if path exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path '{model_path}' does not exist. Please provide correct path.")
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model weights
        model_file = Path(model_path) / "consolidated.00.pth"
        if not model_file.exists():
            model_file = Path(model_path) / "pytorch_model.bin"
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found in {model_path}")
        
        print(f"Loading model weights from {model_file}...")
        self.model = torch.load(model_file, map_location='cpu')
        
        # Move model weights to device
        print(f"Moving model weights to {device}...")
        for key in self.model.keys():
            if isinstance(self.model[key], torch.Tensor):
                self.model[key] = self.model[key].to(device)
        
        # Load config - try multiple possible locations/names
        config_file = None
        possible_configs = [
            Path(model_path) / "params.json",
            Path(model_path) / "config.json",
        ]
        
        for cfg_path in possible_configs:
            if cfg_path.exists():
                config_file = cfg_path
                break
        
        if config_file is None:
            raise FileNotFoundError(
                f"Config file not found. Tried: {[str(p) for p in possible_configs]}\n"
                f"Please ensure your model directory contains a config file."
            )
        
        print(f"Loading config from {config_file}...")
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Handle both LLaMA format (params.json) and HuggingFace format (config.json)
        self.dim = config.get("dim") or config.get("hidden_size")
        self.n_layers = config.get("n_layers") or config.get("num_hidden_layers")
        self.n_heads = config.get("n_heads") or config.get("num_attention_heads")
        self.n_kv_heads = config.get("n_kv_heads") or config.get("num_key_value_heads", self.n_heads)
        self.vocab_size = config.get("vocab_size")
        self.norm_eps = config.get("norm_eps") or config.get("rms_norm_eps", 1e-6)
        self.rope_theta = config.get("rope_theta") or config.get("rope_theta", 10000.0)
        self.head_dim = self.dim // self.n_heads
        
        self.best_config = None
        self.all_configs = []
        
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads, dim={self.dim}")
    
    def rms_norm(self, tensor, norm_weights):
        """RMS Normalization"""
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)) * norm_weights
    
    def apply_rotary_embeddings(self, x, freqs_cis):
        """Apply rotary position embeddings"""
        # x shape: (seq_len, n_heads, head_dim)
        # freqs_cis shape: (seq_len, head_dim // 2)
        # Need to reshape x to (seq_len, n_heads, head_dim // 2, 2) for complex view
        x_shape = x.shape
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # x_complex shape: (seq_len, n_heads, head_dim // 2)
        
        # Reshape freqs_cis to broadcast: (seq_len, 1, head_dim // 2)
        freqs_cis = freqs_cis.unsqueeze(1)
        
        x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
        return x_rotated.type_as(x).reshape(x_shape)
    
    def create_freqs_cis(self, seq_len):
        """Create frequency tensor for rotary embeddings"""
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(seq_len, device=self.device).float()
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def triton_linear(self, x, weight, config):
        """Linear layer using Triton"""
        n_rows = x.shape[0]
        in_features = x.shape[1]
        out_features = weight.shape[0]
        
        output = torch.empty((n_rows, out_features), dtype=torch.float32, device=x.device)
        
        grid = lambda meta: (
            triton.cdiv(n_rows, config.block_size_m),
            triton.cdiv(out_features, config.block_size_n)
        )
        
        fused_linear_kernel[grid](
            x, weight, output,
            n_rows, in_features, out_features,
            BLOCK_M=config.block_size_m,
            BLOCK_N=config.block_size_n,
            BLOCK_K=config.block_size_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
        return output
    
    def forward_layer_triton(self, x, layer_idx, freqs_cis, config):
        """Single transformer layer using Triton kernels for linear ops"""
        # Attention
        attn_norm = self.rms_norm(
            x, 
            self.model[f"model.layers.{layer_idx}.input_layernorm.weight"]
        )
        
        # Q, K, V projections with Triton
        q = self.triton_linear(
            attn_norm,
            self.model[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            config
        ).to(torch.bfloat16)
        k = self.triton_linear(
            attn_norm,
            self.model[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            config
        ).to(torch.bfloat16)
        v = self.triton_linear(
            attn_norm,
            self.model[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            config
        ).to(torch.bfloat16)
        
        # Reshape for multi-head attention
        seq_len = x.shape[0]
        q = q.view(seq_len, self.n_heads, self.head_dim)
        k = k.view(seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = self.apply_rotary_embeddings(q, freqs_cis)
        k = self.apply_rotary_embeddings(k, freqs_cis)
        
        # Multi-head attention
        attn_outputs = []
        for h in range(self.n_heads):
            kv_head = h // (self.n_heads // self.n_kv_heads)
            q_h = q[:, h, :]
            k_h = k[:, kv_head, :]
            v_h = v[:, kv_head, :]
            
            scores = torch.matmul(q_h, k_h.T) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            scores = scores + mask
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_h)
            attn_outputs.append(attn_out)
        
        attn_output = torch.stack(attn_outputs, dim=1).reshape(seq_len, -1)
        
        # Output projection with Triton
        attn_output = self.triton_linear(
            attn_output,
            self.model[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            config
        ).to(torch.bfloat16)
        
        # Residual connection
        x = x + attn_output
        
        # FFN
        ffn_norm = self.rms_norm(
            x,
            self.model[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        )
        
        # Use Triton for FFN linear layers
        w1_out = self.triton_linear(
            ffn_norm,
            self.model[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            config
        ).to(torch.bfloat16)
        w1_out = torch.nn.functional.silu(w1_out)
        
        w3_out = self.triton_linear(
            ffn_norm,
            self.model[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            config
        ).to(torch.bfloat16)
        
        ffn_output = self.triton_linear(
            w1_out * w3_out,
            self.model[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            config
        ).to(torch.bfloat16)
        
        # Residual connection
        x = x + ffn_output
        
        return x
    
    def forward_triton(self, tokens, config):
        """Full forward pass using Triton kernels for matmuls"""
        # Embedding
        embedding_layer = torch.nn.Embedding(self.vocab_size, self.dim)
        embedding_layer.weight.data.copy_(self.model["model.embed_tokens.weight"])
        embedding_layer = embedding_layer.to(self.device)
        x = embedding_layer(tokens).to(torch.bfloat16)
        
        # Create position embeddings
        freqs_cis = self.create_freqs_cis(len(tokens))
        
        # Process all layers
        for layer_idx in range(self.n_layers):
            x = self.forward_layer_triton(x, layer_idx, freqs_cis, config)
        
        # Final norm
        x = self.rms_norm(x, self.model["model.norm.weight"])
        
        # Output projection (use PyTorch for final layer)
        logits = torch.matmul(x[-1], self.model["model.embed_tokens.weight"].T)
        
        return logits
    
    def forward_pytorch(self, tokens):
        """PyTorch reference implementation"""
        embedding_layer = torch.nn.Embedding(self.vocab_size, self.dim)
        embedding_layer.weight.data.copy_(self.model["model.embed_tokens.weight"])
        embedding_layer = embedding_layer.to(self.device)
        x = embedding_layer(tokens).to(torch.bfloat16)
        
        freqs_cis = self.create_freqs_cis(len(tokens))
        
        for layer_idx in range(self.n_layers):
            # Attention
            attn_norm = self.rms_norm(
                x,
                self.model[f"model.layers.{layer_idx}.input_layernorm.weight"]
            )
            
            q = torch.matmul(attn_norm, self.model[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].T)
            k = torch.matmul(attn_norm, self.model[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].T)
            v = torch.matmul(attn_norm, self.model[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].T)
            
            seq_len = x.shape[0]
            q = q.view(seq_len, self.n_heads, self.head_dim)
            k = k.view(seq_len, self.n_kv_heads, self.head_dim)
            v = v.view(seq_len, self.n_kv_heads, self.head_dim)
            
            q = self.apply_rotary_embeddings(q, freqs_cis)
            k = self.apply_rotary_embeddings(k, freqs_cis)
            
            attn_outputs = []
            for h in range(self.n_heads):
                kv_head = h // (self.n_heads // self.n_kv_heads)
                q_h = q[:, h, :]
                k_h = k[:, kv_head, :]
                v_h = v[:, kv_head, :]
                
                scores = torch.matmul(q_h, k_h.T) / (self.head_dim ** 0.5)
                mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
                scores = scores + mask
                attn_weights = torch.nn.functional.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_weights, v_h)
                attn_outputs.append(attn_out)
            
            attn_output = torch.stack(attn_outputs, dim=1).reshape(seq_len, -1)
            attn_output = torch.matmul(attn_output, self.model[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].T)
            
            x = x + attn_output
            
            # FFN
            ffn_norm = self.rms_norm(
                x,
                self.model[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
            )
            
            w1_out = torch.matmul(ffn_norm, self.model[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].T)
            w1_out = torch.nn.functional.silu(w1_out)
            w3_out = torch.matmul(ffn_norm, self.model[f"model.layers.{layer_idx}.mlp.up_proj.weight"].T)
            ffn_output = torch.matmul(w1_out * w3_out, self.model[f"model.layers.{layer_idx}.mlp.down_proj.weight"].T)
            
            x = x + ffn_output
        
        x = self.rms_norm(x, self.model["model.norm.weight"])
        logits = torch.matmul(x[-1], self.model["model.embed_tokens.weight"].T)
        
        return logits
    
    def benchmark_config(self, config, tokens, num_iterations=10, warmup=2):
        """Benchmark a specific configuration"""
        try:
            # Warmup
            for _ in range(warmup):
                _ = self.forward_triton(tokens, config)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                logits = self.forward_triton(tokens, config)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            elapsed_time = (end - start) / num_iterations
            return 1.0 / elapsed_time, logits  # Return inferences/sec
        except Exception as e:
            return 0.0, None
    
    def grid_search(self, tokens, verbose=True):
        """Grid search for optimal configuration"""
        block_sizes_m = [32, 64]
        block_sizes_n = [64, 128]
        block_sizes_k = [32, 64]
        num_warps_options = [4, 8]
        num_stages_options = [2, 3]
        
        best_perf = 0
        best_config = None
        
        if verbose:
            print("\n=== GRID SEARCH OPTIMIZATION ===")
        
        total_configs = (len(block_sizes_m) * len(block_sizes_n) * 
                        len(block_sizes_k) * len(num_warps_options) * 
                        len(num_stages_options))
        print(f"Testing {total_configs} configurations...")
        
        tested = 0
        for bm in block_sizes_m:
            for bn in block_sizes_n:
                for bk in block_sizes_k:
                    for nw in num_warps_options:
                        for ns in num_stages_options:
                            config = KernelConfig(
                                block_size_m=bm,
                                block_size_n=bn,
                                block_size_k=bk,
                                num_warps=nw,
                                num_stages=ns
                            )
                            
                            perf, _ = self.benchmark_config(config, tokens)
                            config.performance = perf
                            self.all_configs.append(config)
                            
                            tested += 1
                            if verbose and perf > 0:
                                print(f"[{tested}/{total_configs}] BM={bm}, BN={bn}, BK={bk}, W={nw}, S={ns} -> {perf:.3f} inf/s")
                            
                            if perf > best_perf:
                                best_perf = perf
                                best_config = config
        
        self.best_config = best_config
        if verbose and best_config:
            print(f"\nBest Grid Search Config:")
            print(f"  Block M={best_config.block_size_m}, Block N={best_config.block_size_n}, Block K={best_config.block_size_k}")
            print(f"  Warps={best_config.num_warps}, Stages={best_config.num_stages}")
            print(f"  Performance: {best_perf:.3f} inferences/sec")
        
        return best_config
    
    def genetic_algorithm(self, tokens, population_size=10, generations=5, 
                         mutation_rate=0.3, verbose=True):
        """Genetic algorithm optimization"""
        block_sizes_m = [32, 64]
        block_sizes_n = [64, 128]
        block_sizes_k = [32, 64]
        num_warps_options = [4, 8]
        num_stages_options = [2, 3]
        
        def create_random_config():
            return KernelConfig(
                block_size_m=random.choice(block_sizes_m),
                block_size_n=random.choice(block_sizes_n),
                block_size_k=random.choice(block_sizes_k),
                num_warps=random.choice(num_warps_options),
                num_stages=random.choice(num_stages_options)
            )
        
        if verbose:
            print("\n=== GENETIC ALGORITHM OPTIMIZATION ===")
            print(f"Population: {population_size}, Generations: {generations}")
        
        population = [create_random_config() for _ in range(population_size)]
        best_ever = None
        best_ever_perf = 0
        
        for gen in range(generations):
            if verbose:
                print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate population
            for config in population:
                if config.performance == 0:
                    perf, _ = self.benchmark_config(config, tokens, num_iterations=5)
                    config.performance = perf
            
            # Sort by performance
            population.sort(key=lambda c: c.performance, reverse=True)
            
            if verbose and population[0].performance > 0:
                print(f"  Best: {population[0].performance:.3f} inf/s "
                      f"(BM={population[0].block_size_m}, BN={population[0].block_size_n}, "
                      f"BK={population[0].block_size_k})")
            
            if population[0].performance > best_ever_perf:
                best_ever = population[0]
                best_ever_perf = population[0].performance
            
            # Selection and reproduction
            elite_size = max(2, population_size // 4)
            elite = [p for p in population if p.performance > 0][:elite_size]
            
            if len(elite) == 0:
                elite = [create_random_config()]
            
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                child = KernelConfig(
                    block_size_m=random.choice([parent1.block_size_m, parent2.block_size_m]),
                    block_size_n=random.choice([parent1.block_size_n, parent2.block_size_n]),
                    block_size_k=random.choice([parent1.block_size_k, parent2.block_size_k]),
                    num_warps=random.choice([parent1.num_warps, parent2.num_warps]),
                    num_stages=random.choice([parent1.num_stages, parent2.num_stages])
                )
                
                # Mutation
                if random.random() < mutation_rate:
                    child.block_size_m = random.choice(block_sizes_m)
                if random.random() < mutation_rate:
                    child.block_size_n = random.choice(block_sizes_n)
                if random.random() < mutation_rate:
                    child.block_size_k = random.choice(block_sizes_k)
                if random.random() < mutation_rate:
                    child.num_warps = random.choice(num_warps_options)
                if random.random() < mutation_rate:
                    child.num_stages = random.choice(num_stages_options)
                
                new_population.append(child)
            
            population = new_population
        
        if verbose and best_ever:
            print(f"\nBest GA Config:")
            print(f"  Block M={best_ever.block_size_m}, Block N={best_ever.block_size_n}, Block K={best_ever.block_size_k}")
            print(f"  Warps={best_ever.num_warps}, Stages={best_ever.num_stages}")
            print(f"  Performance: {best_ever_perf:.3f} inferences/sec")
        
        return best_ever
    
    def generate_text(self, prompt, config, num_tokens=50):
        """Generate text using the optimized config"""
        tokens = torch.tensor(self.tokenizer.encode(prompt), device=self.device)
        
        print(f"\nGenerating {num_tokens} tokens with optimized Triton kernels...")
        print(f"Prompt: '{prompt}'")
        print("="*70)
        
        generated_tokens = []
        
        for i in range(num_tokens):
            logits = self.forward_triton(tokens, config)
            next_token = torch.argmax(logits, dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])
            generated_tokens.append(next_token.item())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i+1}/{num_tokens} tokens...")
        
        generated_text = self.tokenizer.decode(tokens.tolist())
        print("\n" + "="*70)
        print("GENERATED TEXT:")
        print("="*70)
        print(generated_text)
        print("="*70)
        
        return generated_text


def main():
    import sys
    
    print("="*70)
    print("Fused LLaMA 3.2 Triton Kernel Auto-Optimization")
    print("="*70)
    
    # Get model path from command line or use default
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./llama32"
    
    print(f"\nModel path: {model_path}")
    
    # Check if path exists
    if not Path(model_path).exists():
        print(f"\nERROR: Model path '{model_path}' does not exist!")
        print("\nUsage: python script.py [model_path]")
        print("Example: python script.py ./llama32")
        print("\nMake sure your model directory contains:")
        print("  - Tokenizer files")
        print("  - Model weights (consolidated.00.pth or pytorch_model.bin)")
        print("  - Config file (params.json or config.json)")
        return
    
    # Initialize optimizer
    try:
        optimizer = LlamaTritonOptimizer(model_path=model_path)
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        return
    
    # Test prompt
    prompt = "python hello world:"
    tokens = torch.tensor(optimizer.tokenizer.encode(prompt), device='cuda')
    
    print(f"\nTest prompt: '{prompt}'")
    print(f"Tokens: {tokens.tolist()}")
    print(f"Sequence length: {len(tokens)}")
    
    # Benchmark PyTorch baseline
    print("\n" + "="*70)
    print("PYTORCH BASELINE BENCHMARK")
    print("="*70)
    
    warmup = 2
    num_iter = 10
    
    print("Warming up...")
    for _ in range(warmup):
        _ = optimizer.forward_pytorch(tokens)
    
    print("Benchmarking PyTorch implementation...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        pytorch_out = optimizer.forward_pytorch(tokens)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    pytorch_time = (end - start) / num_iter
    pytorch_perf = 1.0 / pytorch_time
    print(f"PyTorch Performance: {pytorch_perf:.3f} inferences/sec ({pytorch_time*1000:.2f} ms/inference)")
    
    # Grid search optimization
    print("\n" + "="*70)
    print("OPTIMIZATION PHASE")
    print("="*70)
    
    gs_config = optimizer.grid_search(tokens, verbose=True)
    
    # Genetic algorithm optimization
    ga_config = optimizer.genetic_algorithm(tokens, verbose=True)
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Method':<30} {'Performance (inf/s)':<20} {'Speedup':<15}")
    print("-"*70)
    print(f"{'PyTorch (Baseline)':<30} {pytorch_perf:<20.3f} {'1.00x':<15}")
    
    if gs_config and gs_config.performance > 0:
        speedup = gs_config.performance / pytorch_perf
        print(f"{'Triton (Grid Search)':<30} {gs_config.performance:<20.3f} {f'{speedup:.2f}x':<15}")
    
    if ga_config and ga_config.performance > 0:
        speedup = ga_config.performance / pytorch_perf
        print(f"{'Triton (Genetic Algo)':<30} {ga_config.performance:<20.3f} {f'{speedup:.2f}x':<15}")
    
    print("="*70)
    
    # Correctness verification
    print("\n" + "="*70)
    print("CORRECTNESS VERIFICATION")
    print("="*70)
    
    best_config = ga_config if (ga_config and ga_config.performance > (gs_config.performance if gs_config else 0)) else gs_config
    
    if best_config:
        triton_out = optimizer.forward_triton(tokens, best_config)
        diff = torch.abs(triton_out - pytorch_out)
        print(f"Max difference: {torch.max(diff):.6e}")
        print(f"Mean difference: {torch.mean(diff):.6e}")
        print(f"Outputs match: {torch.allclose(triton_out, pytorch_out, rtol=1e-2, atol=1e-2)}")
    
    # Text generation with best config
    if best_config:
        print("\n" + "="*70)
        print("TEXT GENERATION WITH OPTIMIZED CONFIG")
        print("="*70)
        optimizer.generate_text(prompt, best_config, num_tokens=50)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
