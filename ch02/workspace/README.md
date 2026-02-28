# Workspace: Reasoning From Scratch (Chapter 2)

This workspace contains technical documentation, performance benchmarks, and theoretical insights gained while exploring the **Qwen3 0.6B** reasoning model and autoregressive text generation.

## 📂 Directory Structure

### 🧠 [Theory](./theory/)
- **[Inference Terminology](./theory/terminology.md):** Distinguishing between Neural Network Inference and Statistical Inference.
- **[Generation Efficiency](./theory/generation_efficiency.md):** Deep dive into KV Caching, Redundancy, and why GPUs process sequences the way they do.

### 🏗️ [Architecture](./architecture/)
- **[Qwen3 0.6B Specs](./architecture/qwen3_specs.md):** Detailed technical reference for the model architecture, including GQA, RoPE, and input/output shapes.

### 🧪 [Experiments](./experiments/)
- **[Performance Results](./experiments/performance_results.md):** Comparison of CPU vs. Apple Silicon GPU (MPS) performance.
- **[Scripts](./experiments/scripts/):** Python scripts used for benchmarks and visualizations.
- **[Assets](./experiments/assets/):** Generated images and visualization output.

---
*Created during a deep-dive session into LLM fundamentals and optimization.*
