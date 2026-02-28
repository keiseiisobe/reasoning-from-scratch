# Hardware and Software Performance Comparison

This experiment compares the performance of the **qwen3-0.6B** model across different hardware backends (CPU vs. GPU) and software optimizations (**Non-cached** vs. **KV Cache**).

## 📊 Results

The following results were obtained by running `generate_performance_stats.py`:

```text
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
Time on CPU: 66.26 sec
1 tokens / sec

Time on CPU with KV Cache: 2.95 sec
33 tokens / sec

Time on GPU: 7.35 sec
13 tokens / sec

Time on GPU with KV Cache: 3.44 sec
29 tokens / sec
```

## 🔍 Analysis

### 1. The Power of KV Caching
The most significant performance gain comes from the **KV Cache**. 
- On the CPU, switching from non-cached to cached generation resulted in a **33x speedup** (1 token/sec to 33 tokens/sec).
- This confirms that for autoregressive generation, reusing mathematical states is more important than raw compute power.

### 2. CPU vs. GPU (Apple Silicon)
The results show an interesting behavior specific to small models (0.6B) on unified memory architectures:
- **Parallel Tasks (Non-cached):** The GPU is much faster (**13x**) because it handles the growing sequence length in parallel more effectively.
- **Serial Tasks (KV Cache):** The **CPU actually outperformed the GPU** (33 t/s vs 29 t/s). This is likely because, for a very small model, the overhead of sending single-token kernels to the GPU (dispatch latency) outweighs the GPU's compute advantage. The CPU can process the single-token "next-step" logic with extremely low latency.

## 📝 Generated Output
> Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.<|endoftext|>Human language is a complex and dynamic system that has evolved over millions of years to enable effective communication and social interaction. Large language models, such as GPT-3 and GPT-4, are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform

## 📂 Related Scripts
- **[Benchmark Script](./scripts/generate_performance_stats.py):** Measures the speed difference between hardware backends and optimizations.
- **[Visualization Script](./scripts/visualize_inference.py):** Generates graphs explaining the terminology distinction between NN and statistical inference.

---
*Results observed on Apple Silicon hardware (M-series).*
