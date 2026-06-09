import struct
import shutil
import os

src_path = "esmc_300m.esmf"
dst_path = "esmc_300m_hf.esmf"

print(f"Copying {src_path} to {dst_path}...")
shutil.copyfile(src_path, dst_path)

# Build the reverse name map
# GGUF-style name -> HF/ESMF-style name
m = {}
m["token_embd.weight"] = "esmc.embed.weight"
m["output_norm.weight"] = "esmc.transformer.norm.weight"
m["lm_head_proj.weight"] = "lm_head.0.weight"
m["lm_head_proj.bias"] = "lm_head.0.bias"
m["lm_head_norm.weight"] = "lm_head.2.weight"
m["lm_head_norm.bias"] = "lm_head.2.bias"
m["lm_head_out.weight"] = "lm_head.3.weight"
m["lm_head_out.bias"] = "lm_head.3.bias"

for i in range(30): # 30 layers
    p = f"esmc.transformer.blocks.{i}"
    b = f"blk.{i}"
    m[f"{b}.attn_norm.weight"] = f"{p}.attn.layernorm_qkv.layer_norm_weight"
    m[f"{b}.attn_norm.bias"]   = f"{p}.attn.layernorm_qkv.layer_norm_bias"
    m[f"{b}.attn_qkv.weight"]  = f"{p}.attn.layernorm_qkv.weight"
    m[f"{b}.attn_out.weight"]  = f"{p}.attn.out_proj.weight"
    m[f"{b}.attn_q_ln.weight"] = f"{p}.attn.q_ln.weight"
    m[f"{b}.attn_k_ln.weight"] = f"{p}.attn.k_ln.weight"
    m[f"{b}.ffn_norm.weight"]  = f"{p}.ffn.layer_norm_weight"
    m[f"{b}.ffn_norm.bias"]    = f"{p}.ffn.layer_norm_bias"
    m[f"{b}.ffn_gate_up.weight"] = f"{p}.ffn.fc1_weight"
    m[f"{b}.ffn_down.weight"]  = f"{p}.ffn.fc2_weight"

ENTRY_FMT = "<64sII4qQQ"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

print(f"Renaming tensors in {dst_path}...")
with open(dst_path, "r+b") as f:
    # Read magic
    magic = f.read(4)
    assert magic == b"ESMF"
    
    # Read version
    version = struct.unpack("<I", f.read(4))[0]
    
    # Read meta size
    meta_size = struct.unpack("<Q", f.read(8))[0]
    f.read(meta_size) # skip metadata JSON
    
    # Read tensor count
    tensor_count = struct.unpack("<I", f.read(4))[0]
    print(f"Found {tensor_count} tensors.")
    
    dir_start = f.tell()
    
    renamed_count = 0
    for i in range(tensor_count):
        entry_offset = dir_start + i * ENTRY_SIZE
        f.seek(entry_offset)
        raw_entry = f.read(ENTRY_SIZE)
        
        name_bytes, dtype, ndim, *rest = struct.unpack(ENTRY_FMT, raw_entry)
        curr_name = name_bytes.rstrip(b"\x00").decode("utf-8")
        
        if curr_name in m:
            new_name = m[curr_name]
            new_name_bytes = new_name.encode("utf-8").ljust(64, b"\x00")
            
            # Pack new entry (just modify the name)
            new_entry = struct.pack(ENTRY_FMT, new_name_bytes, dtype, ndim, *rest)
            
            f.seek(entry_offset)
            f.write(new_entry)
            renamed_count += 1
            
print(f"Successfully renamed {renamed_count} tensors.")
