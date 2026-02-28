# GPU vs. CPU Performance Comparison

This experiment measures the performance difference between running the **qwen3-0.6B** model on a CPU versus an Apple Silicon GPU (using the `mps` backend).

## 📊 Results

The following results were obtained by running `generate_performance_stats.py`:

```text
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
Time on CPU: 64.07 sec
1 tokens / sec
Time on GPU: 7.34 sec
13 tokens / sec
```

### Analysis
The GPU implementation is approximately **8.7x faster** than the CPU implementation (13 tokens/sec vs 1 token/sec). This demonstrates the significant advantage of utilizing specialized hardware for the matrix multiplications required by Transformer models.

## 📝 Generated Output
> Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.<|endoftext|>Human language is a complex and dynamic system that has evolved over millions of years to enable effective communication and social interaction...

## 📂 Related Scripts
- **[Benchmark Script](./scripts/generate_performance_stats.py):** Measures the speed difference between hardware backends.
- **[Visualization Script](./scripts/visualize_inference.py):** Generates graphs explaining the terminology distinction between NN and statistical inference.

---
*Results observed on Apple Silicon hardware (M-series).*
