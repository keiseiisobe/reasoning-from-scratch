# Text Generation and Inference Terminology

This document summarizes the distinction between "inference" in the context of Large Language Models (LLMs) and "inference" in traditional statistics.

## 1. The Terminology Collision
Confusion arises because "Inference" refers to opposite stages of the process depending on the field.

| Phase | In Machine Learning (LLMs) | In Statistics |
| :--- | :--- | :--- |
| **Stage 1: Learning** | **Training** (Learning weights) | **Inference** (Estimating parameters) |
| **Stage 2: Using** | **Inference** (Predicting tokens) | **Prediction** (Applying the function) |

### Key Distinction
- **Training is Statistical Inference:** When we train a model (estimating weights $	heta$ from data), we are performing what a statistician calls "inference."
- **Generation is Neural Network Inference:** When we run `generate_text_basic`, we are performing what a developer calls "inference"—the forward application of a fixed function to produce a prediction. Nothing is being learned during this stage.

## 2. Visual Comparison

A visualization comparing these two concepts can be generated using `visualize_inference.py`.

- **Neural Network Inference (Fixed):** Maps a single input to a single output using a static, pre-learned function.
- **Statistical Inference (Estimation):** Uses noisy data to estimate a range of possible functions and quantifies the uncertainty.

![Inference Comparison](../experiments/assets/inference_comparison.png)

---
*Based on "Build a Reasoning Model (From Scratch)" by Sebastian Raschka.*
