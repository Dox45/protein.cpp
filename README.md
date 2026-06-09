# ESMF: High-Performance C++ Inference for ESMC Protein Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/)
[![GGML](https://img.shields.io/badge/Backend-GGML-orange.svg)](https://github.com/ggerganov/ggml)
[![Bioinformatics](https://img.shields.io/badge/Domain-Bioinformatics-green.svg)](https://en.wikipedia.org/wiki/Bioinformatics)

**ESMF** is a lightweight, pure C++ inference engine for the **ESMC (Evolutionary Scale Modeling C)** series of protein language models (such as ESMC-300M). Built directly on the **GGML** library, ESMF provides high-throughput CPU-based inference with a minimal resource footprint, making it ideal for running local, large-scale sequence analyses.

---

## 🚀 Key Features

*   **Zero-Copy Weight Loading**: Fast mapping of model weights directly into GGML backend buffers.
*   **Memory Reuse Optimization (`ggml_gallocr`)**: Dynamically computes optimal activation tensor lifetimes and offsets, reducing peak forward-pass memory for a 2048-residue sequence from **52 GB** to less than **400 MB**.
*   **Strict PyTorch Parity**: Attention permutation, Rotary Embedding (`ggml_rope_ext`), SwiGLU activations, and LayerNorm implementations verified to be numerically identical to the PyTorch reference down to the logits.
*   **Highly Portable**: Zero heavy external dependencies (e.g., PyTorch/libtorch). Requires only a standard C++17 compiler and CMake.

---

## 🛠️ Build & Installation

### 1. Prerequisites
Ensure you have the following installed on your system:
*   CMake (version 3.10 or higher)
*   A C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+)
*   Python 3 (only for initial weights conversion)

### 2. Weight Conversion
Convert the PyTorch/HuggingFace ESMC weights to the ESMF format using the provided conversion utility:
```bash
python3 convert_esmc_to_esmf.py --input /path/to/pytorch_model.bin --output esmc_300m.esmf
```

### 3. Compilation
Run the unified build script to compile the GGML libraries and the ESMF executable:
```bash
bash build.sh
```
This produces the `./protein` binary.

---

## 📖 Usage

Run inference on a target amino acid sequence by passing the converted `.esmf` weights and the sequence:

```bash
./protein esmc_300m.esmf "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
```

### Options
*   `--threads <int>`: Set the number of CPU threads to use for execution (defaults to 4).
*   `--debug`: Prints Layer-0 intermediate tensor statistics (mean, standard deviation, first 5 elements) for auditing and numerical verification.

---

## 🧠 Model Architecture

The ESM-C architecture mapped in C++ consists of:
1.  **Token Embedding Layer**: Lookups mapped via `ggml_get_rows`.
2.  **Transformer Blocks** (30 layers for ESMC-300M):
    *   **Pre-LayerNorm**: Replicated using standard CPU LayerNorm operators.
    *   **Rotary Position Embeddings (RoPE)**: Dispatched using the `ggml_rope_ext` backend API.
    *   **Multi-Head Attention (MHA)**: Axes aligned to `[L, HD, NH]` for key-query contractions, maintaining compatibility for arbitrary sequence lengths.
    *   **SwiGLU Feed-Forward Network (FFN)**: Splits projections and applies the Swish activation gate: `silu(x_gate) * x_value` via `ggml_silu` and `ggml_mul`.
3.  **Regression Head**: Linear $\to$ GELU $\to$ LayerNorm $\to$ Linear projection to produce output logits over the vocabulary.

---

## 📊 Performance & Memory Footprint

Through the integration of GGML's graph memory allocator (`ggml_gallocr`), activation memory is reused across transformer layers.

| Sequence Length ($L$) | PyTorch / Static Allocation Memory | ESMF (Graph Allocator) Peak Memory |
| :---: | :---: | :---: |
| **53** | ~1.5 GB | **~1.27 GB** |
| **2048** | ~52.0 GB | **~1.67 GB** (incl. 1.27 GB weights) |

> [!NOTE]
> The absolute minimum memory requirement is dictated by the size of the model weights (1.27 GB for ESMC-300M in FP32).

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
