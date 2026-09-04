# Fused LLM Inference Kernels in CUDA
## Author: Tiayo Durel

![Language](https://img.shields.io/badge/language-CUDA%20%2F%20C%2B%2B-76B900)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-20%2F20%20kernels%20complete-success)

## Overview

This project implements high-performance CUDA kernels for large language model (LLM) inference, built up progressively from low-level GPU primitives to fully fused transformer building blocks.

It starts with warp- and block-level reductions and activation functions, then moves on to fused implementations of RMSNorm, Softmax, RoPE, and SwiGLU MLP layers — the same operations that power efficient inference in modern transformer models. The goal is to understand *why* these kernels are fast, not just *that* they work: how memory access patterns, warp shuffles, and kernel fusion combine to squeeze latency out of every layer of a transformer forward pass.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Kernel Checklist](#kernel-checklist)
- [Usage & Configuration](#usage--configuration)
- [Credits](#credits)

## Prerequisites

Make sure the following are installed before running the scaffold:

- Python 3.9+
- CUDA Toolkit 11.8+
- A CUDA-capable GPU with an up-to-date driver
- CMake 3.18+ (if building kernels outside the Python scaffold)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the scaffold to build and test the kernels end-to-end:

```bash
python scaffold.py
```

This compiles each kernel, runs it against a reference implementation, and reports pass/fail status for every step below.

## Kernel Checklist

Each item below is implemented and verified against a reference CPU/PyTorch implementation.

| # | Kernel | Status |
|---|--------|--------|
| 1 | `warp_reduce_sum` | ✅ |
| 2 | `warp_reduce_max` | ✅ |
| 3 | `block_reduce_sum` | ✅ |
| 4 | `block_reduce_max` | ✅ |
| 5 | `add_residual_kernel` | ✅ |
| 6 | `gelu_kernel` | ✅ |
| 7 | `silu_kernel` | ✅ |
| 8 | `swiglu_kernel` | ✅ |
| 9 | `rmsnorm_kernel` | ✅ |
| 10 | `layernorm_kernel` | ✅ |
| 11 | `fused_add_rmsnorm_kernel` | ✅ |
| 12 | `softmax_row_kernel` | ✅ |
| 13 | `causal_softmax_kernel` | ✅ |
| 14 | `embedding_lookup_kernel` | ✅ |
| 15 | `rope_kernel` | ✅ |
| 16 | `linear_kernel` | ✅ |
| 17 | `fused_linear_bias_gelu_kernel` | ✅ |
| 18 | `mlp_swiglu_forward` | ✅ |
| 19 | `rmsnorm_residual_block` | ✅ |
| 20 | `run_transformer_ffn` | ✅ |

## Usage & Configuration

Run a single kernel or test group instead of the full scaffold by passing its name or step number:

```bash
# Run a single kernel by name
python scaffold.py --kernel rmsnorm_kernel

# Run a range of steps (e.g., the fused FFN block)
python scaffold.py --steps 17-20

# Run with verbose timing output for benchmarking
python scaffold.py --kernel run_transformer_ffn --benchmark
```

**Example: benchmarking the fused FFN kernel**

```bash
$ python scaffold.py --kernel run_transformer_ffn --benchmark
[PASS] run_transformer_ffn
  input shape:  (batch=32, seq=2048, hidden=4096)
  latency:      1.42 ms (fused)  vs  3.87 ms (unfused reference)
  speedup:      2.72x
```

**Example: verifying RoPE against a reference implementation**

```bash
$ python scaffold.py --kernel rope_kernel
[PASS] rope_kernel
  max abs error vs reference: 3.1e-6
```

## Credits

Built on [Deep-ML](https://deep-ml.com).
