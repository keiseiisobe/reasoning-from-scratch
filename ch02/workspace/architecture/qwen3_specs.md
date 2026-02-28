# Qwen3 0.6B Model Architecture

This document provides a technical reference for the **Qwen3 0.6B** architecture implemented in this project.

## 1. Key Specifications

| Feature | Specification | Description |
| :--- | :--- | :--- |
| **Parameters** | ~0.6 Billion | 600M parameters for a balance of speed and capability. |
| **Layers** | 28 | Number of Transformer blocks (depth). |
| **Embedding Dim** | 1024 | Width of the hidden states ($d_{model}$). |
| **Vocab Size** | 151,936 | Large vocabulary for efficient multilingual/code processing. |
| **Context Length** | 40,960 | Maximum sequence length (hard limit). |
| **Attention** | GQA | Grouped-Query Attention (8 KV groups for 16 Query heads). |
| **Positional Enc.** | RoPE | Rotary Positional Embeddings with a base of 1,000,000. |

## 2. Advanced Architectural Features

### Grouped-Query Attention (GQA)
Qwen3 uses **GQA**, where 16 Query heads share 8 Key-Value pairs (a 2:1 ratio).
- **Benefit:** Drastically reduces the memory footprint of the **KV Cache**, enabling the large 40k context window to fit in consumer GPU memory.

### Rotary Positional Embeddings (RoPE)
Positions are encoded by rotating the embedding vectors in complex space. 
- The high **RoPE Base (1,000,000)** allows the model to maintain precise positional relationships even at the very end of a 40,000-token sequence.

### Stability & Normalization
- **RMSNorm:** Faster normalization by calculating only the variance (skipping the mean).
- **QK Normalization:** Queries and Keys are normalized before the dot-product to prevent attention score explosions.

## 3. Input and Output Shapes

The architecture processes 2D tensors of tokens and outputs 3D tensors of probability scores (logits).

- **Input Shape:** `(Batch_Size, Sequence_Length)`
- **Output Shape:** `(Batch_Size, Sequence_Length, 151936)`

### Generation Step (with KV Cache)
1. **Prefill (Iter 1):** Input is `(1, N)`, Output is `(1, N, 151936)`.
2. **Decoding (Iter 2+):** Input is **`(1, 1)`**, Output is **`(1, 1, 151936)`**.

## 4. Limits and Constraints

### Sequence Length Limit (40,960)
The model cannot exceed 40,960 tokens because:
1. **RoPE Buffers:** The rotation tables are pre-calculated for exactly this length.
2. **Memory:** The KV Cache grows linearly with the sequence length. Exceeding this limit results in `IndexError` or GPU Out-of-Memory (OOM).

---
*Technical reference for Qwen3-0.6B.*
