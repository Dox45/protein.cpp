# ESMF — C++ Inference for ESMC Protein Language Models

[![License: MIT](https://img.shields.io/badge/Code_License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/)
[![Backend: GGML](https://img.shields.io/badge/Backend-GGML-orange.svg)](https://github.com/ggerganov/ggml)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-lightgrey.svg)](#project-status)
[![Contributions welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](#contributing)

ESMF is a standalone C++ inference engine for the **ESMC (ESM Cambrian)** family of protein language models, built directly on the [GGML](https://github.com/ggerganov/ggml) tensor library. It runs ESMC-variants on CPU with no Python, PyTorch, or LibTorch at runtime — the compiled binary is portable and drops into existing C/C++ or Rust pipelines for structural biology and sequence analysis.

The goal is a small, auditable, hardware-efficient runner that produces outputs faithful to the reference PyTorch implementation, with optional integer quantization for memory-constrained machines.

> **Looking for collaborators.** ESMF is an active project and I'm opening it up for contributors — see [Contributing](#contributing). Numerical-correctness work, quantization, and support for larger ESMC variants are the areas where help would matter most right now.

---

## Contents

- [Project Status](#project-status)
- [Why ESMF](#why-esmf)
- [Performance](#performance)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Quantization](#quantization)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgements & Citation](#acknowledgements--citation)
- [License](#license)

---

## Project Status

**Beta.** The full forward pass is implemented and runs end to end; the loader, quantizer, and CLI are functional. Active work is focused on closing the remaining numerical gap against the reference PyTorch implementation across all layers.

**Validation.** Correctness is checked by comparing per-layer activations and final logits against `esm`/HuggingFace ESMC-300M/600M on a fixed set of reference sequences. See [`validation/`](#) for the harness and current tolerances. *Quantization accuracy claims below assume FP32 parity has been established; treat them as provisional until the validation suite passes cleanly.*

If you hit a discrepancy, the `--debug` flag (below) dumps Layer-0 activation statistics for side-by-side comparison — bug reports with that output attached are especially useful.

---

## Why ESMF

Running ESMC-300M through PyTorch on CPU carries large memory overhead, slow startup, and inefficient execution. ESMF targets the local-inference case directly:

- **No Python/PyTorch at runtime.** Once compiled, the binary is self-contained and portable.
- **Low memory footprint.** A compiled `ggml_gallocr` graph planner computes activations in-place in a small reusable buffer, keeping peak activation memory bounded even for long sequences.
- **Native quantization.** Built-in `Q8_0` (8-bit) and `Q4_0` (4-bit) integer quantization substantially shrink the on-disk and in-memory weights.
- **Instant load.** Weights are memory-mapped (`mmap`), so load time is effectively independent of file size.

---

## Performance

Measured on a single Linux workstation (x86_64, 4 CPU threads) with a 53-residue sequence. Numbers are from one machine and one sequence length; treat them as indicative and reproduce on your own hardware before relying on them. A reproduction script lives in [`bench/`](#).

| Metric | PyTorch (CPU) | ESMF FP32 | ESMF Q8_0 | ESMF Q4_0 |
| :--- | :---: | :---: | :---: | :---: |
| Model size on disk | 1.27 GB | 1.27 GB | 342 MB | **179 MB** |
| Peak RAM (weights) | ~2.54 GB | 1.27 GB | 342 MB | **179 MB** |
| Peak RAM (activations) | ~1.50 GB | ~400 MB | ~400 MB | **~400 MB** |
| Load time | ~2.80 s | ~0.08 s | ~0.03 s | **~0.02 s** |
| Forward pass | ~1.42 s | ~0.15 s | ~0.12 s | **~0.09 s** |
| Speedup vs PyTorch CPU | 1.0× | 9.4× | 11.8× | **15.7×** |

---

## Architecture

ESMC is mapped to GGML operations chosen to match the reference output:

1. **Token embedding** — input tokens mapped to continuous states via `ggml_get_rows`.
2. **Transformer stack** (per block):
   - **Pre-LayerNorm** before attention and FFN.
   - **Rotary Position Embeddings (RoPE)** applied to query and key states via `ggml_rope_ext`.
   - **Multi-head attention** with an optimized projection layout: the `V` matrix is transposed and made contiguous as `[L, HD, NH]` (sequence length, head dim, num heads) to align with the `[L, L, NH]` attention-score matrix, enabling fast parallel dot products.
   - **SwiGLU feed-forward**, with gate and value projections computed in one pass:

     $$\text{SwiGLU}(x) = \big(\text{SiLU}(x_{\text{gate}}) \odot x_{\text{value}}\big)\,W_{\text{down}}$$
3. **Masked LM head** — final representations projected through `Linear → GELU → LayerNorm → Linear` to produce per-residue vocabulary logits.

The loader and forward pass read block count, head count, and dimensions directly from the `.esmf` header, so they are hyperparameter-agnostic — supporting a larger ESMC variant is primarily a matter of extending the conversion script.

---

## Installation

### 1. Prerequisites

A C++17 compiler and CMake:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

### 2. Convert weights to the ESMF format

Done once, from a PyTorch/HuggingFace ESMC checkpoint:

```bash
pip3 install torch numpy transformers
python3 convert_esmc_to_esmf.py \
  --input /path/to/pytorch_model.bin \
  --output esmc_300m.esmf
```

### 3. Build

```bash
bash build.sh
```

This produces two executables:

- `./protein` — the inference runner
- `./protein-quantize` — the quantization utility

---

## Usage

Run inference on an amino-acid sequence. Use `X` to mask a residue for the model to predict:

```bash
./protein <model.esmf> <sequence> [n_threads] [--debug]
```

Example:

```bash
./protein esmc_300m_q4.esmf \
  "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK" 4
```

**Arguments**

| Argument | Description |
| :--- | :--- |
| `model.esmf` | Path to FP32, FP16, or quantized (`q4_0` / `q8_0`) weights |
| `sequence` | Target amino-acid sequence |
| `n_threads` | CPU threads to use (default: 4) |
| `--debug` | Print Layer-0 activation statistics (mean, std, first 5 elements) for numerical auditing |

---

## Quantization

Compress a converted model to reduce its memory footprint:

```bash
# 4-bit — lowest footprint
./protein-quantize esmc_300m.esmf esmc_300m_q4.esmf q4_0

# 8-bit — higher precision
./protein-quantize esmc_300m.esmf esmc_300m_q8.esmf q8_0
```

---

## Roadmap

- [ ] Full per-layer numerical parity with reference ESMC(FP32)
- [ ] Validation harness and reference fixtures checked into the repo
- [ ] Reproducible benchmark script with documented hardware
- [ ] Support for larger ESMC variants (6B), not yet tested
- [ ] Optional FP16 path and additional quantization schemes
- [ ] Prebuilt binaries / packaging

If you'd like to own one of these, open an issue and say so.

---

## Contributing

Contributions are very welcome — this is a good project to get into if you're interested in transformers, GGML, quantization, or computational biology.

**Good places to start**

- Reproduce the benchmarks on your hardware and report numbers
- Extend the validation harness with new reference sequences
- Help track down per-layer numerical discrepancies (the `--debug` output is your friend)
- Add the conversion mapping for a larger ESMC variant

**How to contribute**

1. Open an issue describing the bug, feature, or question before large changes, so we can align on approach.
2. Fork, branch, and submit a pull request. Keep PRs focused and include a short description of what you changed and how you tested it.
3. For numerical work, include before/after activation or logit comparisons against the reference.

If you're interested in collaborating more closely rather than one-off PRs, reach out via the repo's Discussions or issues — I'm happy to scope something.

> **Note for the C++ work:** patches that touch the forward pass should state which file and function they apply to, since local and reference copies can drift.

---

## Acknowledgements & Citation

ESMF builds on the work of others:

- **ESMC / ESM Cambrian** — model architecture and weights by [Biohub](https://www.biohub.org/). Refer to their repository and publications for the model itself.
- **GGML** — tensor library by Georgi Gerganov and contributors.

If ESMF is useful in your research, please cite this repository and the original ESM work. A template:

```bibtex
@software{esmf,
  author  = {Chima Emmanuel},
  title   = {ESMF: C++ Inference for ESMC Protein Language Models},
  year    = {2026},
  url     = {https://github.com/Dox45/protein.cpp}
}
```

Please also cite the underlying ESMC model per Biohub's citation guidance.

---

## License

The **ESMF source code** is released under the [MIT License](LICENSE).

**Model weights are licensed separately.** ESMC weights are distributed by EvolutionaryScale under their own terms — confirm and link that license before redistributing weights or `.esmf` conversions of them. GGML is MIT-licensed by its authors. ESMF's MIT license covers this project's code only, not third-party models or libraries.