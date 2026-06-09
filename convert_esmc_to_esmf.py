#!/usr/bin/env python3
"""
convert_esmc_to_esmf.py

Converts ESMC 300M HuggingFace/ESM SDK weights → ESMF v0 binary format.

ESMF v0 layout:
  [ MAGIC (4B) ]
  [ VERSION (4B) ]
  [ METADATA_SIZE (8B) ]
  [ METADATA JSON (variable) ]
  [ TENSOR_COUNT (4B) ]
  [ TENSOR DIRECTORY (n × entry) ]
  [ ALIGNMENT PAD to 64B ]
  [ TENSOR BLOB ]

Usage:
    python convert_esmc_to_esmf.py --model esmc_300m --out esmc_300m.esmf
    python convert_esmc_to_esmf.py --model esmc_600m --out esmc_600m.esmf
    python convert_esmc_to_esmf.py --safetensors path/to/model.safetensors --out esmc.esmf
"""

import argparse
import json
import struct
import sys
import os
import numpy as np
from pathlib import Path

# ── ESMF constants ────────────────────────────────────────────────────────────

ESMF_MAGIC   = b"ESMF"
ESMF_VERSION = 1

# dtype enum  (must stay stable forever)
DTYPE_F32  = 0
DTYPE_F16  = 1
DTYPE_BF16 = 2
DTYPE_Q4_0 = 3
DTYPE_Q8_0 = 4

NUMPY_TO_DTYPE = {
    np.float32: DTYPE_F32,
    np.float16: DTYPE_F16,
}

# BF16 stored as uint16 raw bytes — we tag it separately
BF16_DTYPE_TAG = DTYPE_BF16

TENSOR_NAME_MAXLEN = 64   # bytes, null-padded (fixed field in directory)
MAX_DIMS           = 4    # ESMF v0 supports up to 4D tensors

BLOB_ALIGNMENT = 64       # bytes

# ── HuggingFace → ESMF name mapping ──────────────────────────────────────────

def build_name_map(n_layers: int) -> dict:
    """
    Returns { hf_name: esmf_name } for all ESMC tensors.
    """
    m = {}

    # Embedding
    m["esmc.embed.weight"] = "token_embd.weight"

    # Per-block tensors
    for i in range(n_layers):
        p = f"esmc.transformer.blocks.{i}"
        b = f"blk.{i}"

        # Attention pre-norm (fused with QKV proj in HF)
        m[f"{p}.attn.layernorm_qkv.layer_norm_weight"] = f"{b}.attn_norm.weight"
        m[f"{p}.attn.layernorm_qkv.layer_norm_bias"]   = f"{b}.attn_norm.bias"

        # Fused QKV projection  [3*hidden, hidden]
        m[f"{p}.attn.layernorm_qkv.weight"] = f"{b}.attn_qkv.weight"

        # Output projection
        m[f"{p}.attn.out_proj.weight"] = f"{b}.attn_out.weight"

        # QK LayerNorm (applied post-split, pre-RoPE)
        m[f"{p}.attn.q_ln.weight"] = f"{b}.attn_q_ln.weight"
        m[f"{p}.attn.k_ln.weight"] = f"{b}.attn_k_ln.weight"

        # FFN pre-norm
        m[f"{p}.ffn.layer_norm_weight"] = f"{b}.ffn_norm.weight"
        m[f"{p}.ffn.layer_norm_bias"]   = f"{b}.ffn_norm.bias"

        # FFN weights
        # fc1 is gate+up fused: shape [2*ffn_inter, hidden]
        # split at dim 0: [:ffn_inter] = gate, [ffn_inter:] = up
        m[f"{p}.ffn.fc1_weight"] = f"{b}.ffn_gate_up.weight"
        m[f"{p}.ffn.fc2_weight"] = f"{b}.ffn_down.weight"

    # Final norm
    m["esmc.norm.weight"] = "output_norm.weight"
    m["esmc.transformer.norm.weight"] = "output_norm.weight"

    # LM head (3-layer MLP)
    m["lm_head.0.weight"] = "lm_head_proj.weight"
    m["lm_head.0.bias"]   = "lm_head_proj.bias"
    m["lm_head.2.weight"] = "lm_head_norm.weight"
    m["lm_head.2.bias"]   = "lm_head_norm.bias"
    m["lm_head.3.weight"] = "lm_head_out.weight"
    m["lm_head.3.bias"]   = "lm_head_out.bias"

    return m


# ── Tensor directory entry  ───────────────────────────────────────────────────
#
# On-disk layout (per entry):
#   name        : 64s   (null-padded UTF-8)
#   dtype       : I     (uint32)
#   ndim        : I     (uint32)
#   shape       : 4q    (4 × int64, unused dims = 0)
#   data_offset : Q     (uint64, offset into blob)
#   data_size   : Q     (uint64, bytes)
#
# Total: 64 + 4 + 4 + 32 + 8 + 8 = 120 bytes per entry

ENTRY_FMT  = "<64sII4qQQ"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)  # should be 120


def pack_entry(name: str, dtype: int, shape: tuple,
               data_offset: int, data_size: int) -> bytes:
    name_bytes = name.encode("utf-8")[:TENSOR_NAME_MAXLEN]
    name_bytes = name_bytes.ljust(TENSOR_NAME_MAXLEN, b"\x00")

    ndim = len(shape)
    assert ndim <= MAX_DIMS, f"Tensor {name} has {ndim} dims, max is {MAX_DIMS}"

    padded_shape = list(shape) + [0] * (MAX_DIMS - ndim)

    return struct.pack(ENTRY_FMT,
        name_bytes,
        dtype,
        ndim,
        *padded_shape,
        data_offset,
        data_size,
    )


# ── dtype helpers  ────────────────────────────────────────────────────────────

def tensor_to_numpy(tensor, target_dtype: str = "f32"):
    """Convert a torch tensor to numpy with the requested dtype."""
    import torch

    t = tensor.detach()

    if target_dtype == "f32":
        return t.float().cpu().numpy().astype(np.float32), DTYPE_F32
    elif target_dtype == "f16":
        return t.half().cpu().numpy().astype(np.float16), DTYPE_F16
    elif target_dtype == "bf16":
        # Store raw bf16 bytes; numpy has no bf16 so we use uint16 view
        arr = t.to(torch.bfloat16).cpu().numpy()   # numpy >= 2.0 supports bf16
        # Fallback: view as uint16 if numpy doesn't support bf16
        try:
            arr = arr.astype(np.float32)  # safest for v0
            return arr, DTYPE_F32
        except Exception:
            return arr.view(np.uint16), BF16_DTYPE_TAG
    else:
        raise ValueError(f"Unknown target dtype: {target_dtype}")


# ── Model loader  ─────────────────────────────────────────────────────────────

def load_from_esm_sdk(model_name: str):
    """Load weights via the official ESM Python SDK."""
    try:
        # from esm.models.esmc import ESMC
        from transformers import AutoModelForMaskedLM, AutoTokenizer

    except ImportError:
        print("ERROR: ESM SDK not found. Install with: pip install esm")
        sys.exit(1)

    print(f"Loading {model_name} via ESM SDK...")
    model = model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-300M")
    model.eval()
    return dict(model.named_parameters())


def load_from_safetensors(path: str):
    """Load weights from a .safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not found. Install with: pip install safetensors")
        sys.exit(1)

    import torch
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_from_pytorch(path: str):
    """Load weights from a .pt or .bin file."""
    import torch
    print(f"Loading from {path}...")
    state_dict = torch.load(path, map_location="cpu")
    # Handle wrapped state dicts
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


# ── Architecture inference  ───────────────────────────────────────────────────

def infer_arch(state_dict: dict) -> dict:
    """Infer architecture hyperparameters from tensor shapes."""

    embed = state_dict.get("esmc.embed.weight")
    if embed is None:
        raise ValueError("Could not find esmc.embed.weight — is this an ESMC checkpoint?")

    vocab_size, hidden_size = embed.shape

    # Count layers by finding max block index
    n_layers = 0
    for k in state_dict:
        if k.startswith("esmc.transformer.blocks."):
            idx = int(k.split(".")[3])
            n_layers = max(n_layers, idx + 1)

    # FFN dims from block 0
    fc1 = state_dict[f"esmc.transformer.blocks.0.ffn.fc1_weight"]
    fc2 = state_dict[f"esmc.transformer.blocks.0.ffn.fc2_weight"]
    ffn_intermediate = fc2.shape[1]   # fc2: [hidden, ffn_inter]
    # fc1 is gate+up fused so fc1.shape[0] = 2 * ffn_intermediate

    # n_heads from QKV weight
    qkv = state_dict[f"esmc.transformer.blocks.0.attn.layernorm_qkv.weight"]
    # qkv shape: [3*hidden, hidden] → n_heads inferred assuming head_dim=64
    head_dim = 64  # standard for ESMC
    n_heads = hidden_size // head_dim

    return {
        "model_type":       "esmc",
        "n_layers":         n_layers,
        "hidden_size":      hidden_size,
        "n_heads":          n_heads,
        "head_dim":         head_dim,
        "ffn_intermediate": ffn_intermediate,
        "vocab_size":       vocab_size,
        "ffn_activation":   "swiglu",
        "norm_type":        "layernorm",
        "qk_layernorm":     True,
        "max_seq_len":      2048,
        "esmf_version":     ESMF_VERSION,
    }


# ── Writer  ───────────────────────────────────────────────────────────────────

def write_esmf(out_path: str, state_dict: dict, arch: dict,
               target_dtype: str = "f32", verbose: bool = True):

    n_layers = arch["n_layers"]
    name_map = build_name_map(n_layers)

    # Validate: warn about unmapped tensors
    unmapped = [k for k in state_dict if k not in name_map]
    if unmapped:
        print(f"WARNING: {len(unmapped)} tensors not in name map (will be skipped):")
        for u in unmapped:
            print(f"  {u}")

    missing = [k for k in name_map if k not in state_dict]
    if missing:
        print(f"WARNING: {len(missing)} expected tensors missing from checkpoint:")
        for m in missing:
            print(f"  {m}")

    # Build tensor list in a stable order
    tensor_list = []  # [(esmf_name, numpy_array, dtype_enum)]

    for hf_name, esmf_name in name_map.items():
        if hf_name not in state_dict:
            continue
        arr, dtype_enum = tensor_to_numpy(state_dict[hf_name], target_dtype)
        tensor_list.append((esmf_name, arr, dtype_enum))

    if verbose:
        print(f"\nTensors to write: {len(tensor_list)}")
        print(f"Target dtype:     {target_dtype}")
        print(f"Output:           {out_path}")

    # ── Compute blob offsets  ─────────────────────────────────────────────────

    blob_entries = []
    offset = 0
    for esmf_name, arr, dtype_enum in tensor_list:
        size = arr.nbytes
        blob_entries.append((esmf_name, arr, dtype_enum, offset, size))
        offset += size
        # Pad to 64B alignment between tensors
        remainder = offset % BLOB_ALIGNMENT
        if remainder != 0:
            offset += BLOB_ALIGNMENT - remainder

    total_blob_size = offset

    # ── Serialize metadata JSON  ──────────────────────────────────────────────

    metadata_json = json.dumps(arch, indent=2).encode("utf-8")

    # ── Compute section offsets  ──────────────────────────────────────────────
    #
    # File layout:
    #   [0]  MAGIC        4B
    #   [4]  VERSION      4B
    #   [8]  META_SIZE    8B   (uint64, size of JSON blob)
    #   [16] META_JSON    variable
    #   [16+META_SIZE] TENSOR_COUNT   4B
    #   [16+META_SIZE+4] TENSOR_DIR   n * ENTRY_SIZE bytes
    #   [...] ALIGNMENT PAD to 64B
    #   [...] TENSOR BLOB

    header_size      = 4 + 4 + 8   # magic + version + meta_size field
    meta_section_end = header_size + len(metadata_json)
    tensor_count_pos = meta_section_end
    dir_start        = tensor_count_pos + 4
    dir_end          = dir_start + len(tensor_list) * ENTRY_SIZE

    # Align blob start to 64B
    remainder = dir_end % BLOB_ALIGNMENT
    pad_size  = (BLOB_ALIGNMENT - remainder) % BLOB_ALIGNMENT
    blob_start = dir_end + pad_size

    # ── Write file  ───────────────────────────────────────────────────────────

    with open(out_path, "wb") as f:

        # Magic + version
        f.write(ESMF_MAGIC)
        f.write(struct.pack("<I", ESMF_VERSION))

        # Metadata
        f.write(struct.pack("<Q", len(metadata_json)))
        f.write(metadata_json)

        # Tensor count
        f.write(struct.pack("<I", len(tensor_list)))

        # Tensor directory
        for esmf_name, arr, dtype_enum, data_offset, data_size in blob_entries:
            entry = pack_entry(
                name        = esmf_name,
                dtype       = dtype_enum,
                shape       = arr.shape,
                data_offset = data_offset,
                data_size   = data_size,
            )
            f.write(entry)

        # Alignment padding before blob
        f.write(b"\x00" * pad_size)

        assert f.tell() == blob_start, \
            f"Blob start mismatch: expected {blob_start}, got {f.tell()}"

        # Tensor blob
        written = 0
        for i, (esmf_name, arr, dtype_enum, data_offset, data_size) in enumerate(blob_entries):
            assert f.tell() == blob_start + data_offset, \
                f"Offset mismatch for {esmf_name}"

            f.write(arr.tobytes())
            written += data_size

            # Alignment pad between tensors
            remainder = (blob_start + data_offset + data_size) % BLOB_ALIGNMENT
            if remainder != 0:
                f.write(b"\x00" * (BLOB_ALIGNMENT - remainder))

            if verbose and (i % 20 == 0 or i == len(blob_entries) - 1):
                pct = (i + 1) / len(blob_entries) * 100
                print(f"  [{pct:5.1f}%] wrote {esmf_name} {arr.shape} {arr.dtype}")

    final_size = os.path.getsize(out_path)
    print(f"\nDone. File size: {final_size / 1e6:.1f} MB")
    print(f"Tensors written: {len(tensor_list)}")


# ── Verification  ─────────────────────────────────────────────────────────────

def verify_esmf(path: str):
    """Quick sanity check: parse header and directory, print summary."""
    print(f"\nVerifying {path}...")

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == ESMF_MAGIC, f"Bad magic: {magic}"

        version = struct.unpack("<I", f.read(4))[0]
        print(f"  Version:  {version}")

        meta_size = struct.unpack("<Q", f.read(8))[0]
        meta_json = json.loads(f.read(meta_size).decode("utf-8"))
        print(f"  Metadata: {json.dumps(meta_json, indent=4)}")

        n_tensors = struct.unpack("<I", f.read(4))[0]
        print(f"  Tensors:  {n_tensors}")

        entries = []
        for _ in range(n_tensors):
            raw = f.read(ENTRY_SIZE)
            name_b, dtype, ndim, *shape_and_offsets = struct.unpack(ENTRY_FMT, raw)
            shape = shape_and_offsets[:4]
            data_offset, data_size = shape_and_offsets[4], shape_and_offsets[5]
            name = name_b.rstrip(b"\x00").decode("utf-8")
            entries.append((name, dtype, ndim, shape[:ndim], data_offset, data_size))

        print(f"\n  First 5 tensors:")
        for name, dtype, ndim, shape, off, size in entries[:5]:
            dtype_str = {0:"f32",1:"f16",2:"bf16"}.get(dtype, "?")
            print(f"    {name:40s} {str(shape):20s} {dtype_str}  offset={off}  bytes={size}")

        print(f"\n  Last 5 tensors:")
        for name, dtype, ndim, shape, off, size in entries[-5:]:
            dtype_str = {0:"f32",1:"f16",2:"bf16"}.get(dtype, "?")
            print(f"    {name:40s} {str(shape):20s} {dtype_str}  offset={off}  bytes={size}")

    print("\nVerification passed.")


# ── CLI  ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert ESMC weights to ESMF v0 format"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model",
        help="ESM SDK model name (e.g. esmc_300m, esmc_600m)")
    source.add_argument("--safetensors",
        help="Path to .safetensors file")
    source.add_argument("--pytorch",
        help="Path to .pt or .bin file")

    parser.add_argument("--out", required=True,
        help="Output .esmf file path")
    parser.add_argument("--dtype", default="f32",
        choices=["f32", "f16"],
        help="Output tensor dtype (default: f32)")
    parser.add_argument("--verify", action="store_true",
        help="Verify output file after writing")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress per-tensor progress output")

    args = parser.parse_args()

    # Load weights
    if args.model:
        state_dict = load_from_esm_sdk(args.model)
    elif args.safetensors:
        state_dict = load_from_safetensors(args.safetensors)
    else:
        state_dict = load_from_pytorch(args.pytorch)

    # Infer architecture
    arch = infer_arch(state_dict)
    print(f"\nInferred architecture:")
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # Write ESMF
    write_esmf(
        out_path     = args.out,
        state_dict   = state_dict,
        arch         = arch,
        target_dtype = args.dtype,
        verbose      = not args.quiet,
    )

    # Optional verify pass
    if args.verify:
        verify_esmf(args.out)


if __name__ == "__main__":
    main()