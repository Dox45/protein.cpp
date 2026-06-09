import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np

# Load model and tokenizer
model_name = "biohub/ESMC-300M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
inputs = tokenizer(seq, return_tensors="pt")
input_ids = inputs["input_ids"] # shape [1, L]

def get_stats(name, t):
    if t is None:
        return f"{name}: None"
    # Convert to float32 numpy
    arr = t.detach().cpu().float().numpy()
    mean = arr.mean()
    std = arr.std()
    first_few = arr.flatten()[:5]
    first_few_str = ", ".join([f"{x:.6f}" for x in first_few])
    return f"{name:25s} | shape: {str(list(arr.shape)):15s} | mean: {mean:11.6f} | std: {std:11.6f} | first 5: [{first_few_str}]"

print("--- PyTorch Intermediate Activations (Layer 0) ---")

# Hook to extract layer 0 intermediate values
# We can do this by running parts of the forward pass manually or using hooks.
# Let's do it by extracting the layer 0 module and running it step-by-step!

block = model.esmc.transformer.blocks[0]
d_model = model.config.d_model
n_heads = block.attn.n_heads
d_head = d_model // n_heads

with torch.no_grad():
    # 0. Embeddings
    x = model.esmc.embed(input_ids)
    print(get_stats("embeddings", x))
    
    # 1. Pre-LN
    attn_in = block.attn.layernorm_qkv[0](x)
    print(get_stats("attn_in (Pre-LN)", attn_in))
    
    # 2. QKV projection
    qkv = block.attn.layernorm_qkv[1](attn_in)
    print(get_stats("qkv projection", qkv))
    
    # Split
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    print(get_stats("Q split", q))
    print(get_stats("K split", k))
    print(get_stats("V split", v))
    
    # 3. QK-Norm
    q_norm = block.attn.q_ln(q)
    k_norm = block.attn.k_ln(k)
    print(get_stats("Q after QK-Norm", q_norm))
    print(get_stats("K after QK-Norm", k_norm))
    
    # 4. RoPE
    # Reshape for rotary
    q_rope = q_norm.unflatten(-1, (n_heads, d_head))
    k_rope = k_norm.unflatten(-1, (n_heads, d_head))
    q_rope, k_rope = block.attn.rotary(q_rope, k_rope)
    # Flatten back
    q_rope = q_rope.flatten(-2, -1)
    k_rope = k_rope.flatten(-2, -1)
    print(get_stats("Q after RoPE", q_rope))
    print(get_stats("K after RoPE", k_rope))
    
    # 5. Attention scores
    # Reshape for matmul: [B, H, L, D_head]
    q_BHLD = q_rope.transpose(1, 2)
    k_BHLD = k_rope.transpose(1, 2)
    v_BHLD = v.unflatten(-1, (n_heads, d_head)).transpose(1, 2)
    
    scale = d_head**-0.5
    scores = torch.einsum("bhld,bhsd->bhls", q_BHLD, k_BHLD) * scale
    print(get_stats("scores before softmax", scores))
    
    attn_weights = torch.softmax(scores, dim=-1)
    print(get_stats("scores after softmax", attn_weights))
    
    # 6. Value multiplication
    context_BHLD = torch.einsum("bhls,bhsd->bhld", attn_weights, v_BHLD)
    context_BLD = context_BHLD.transpose(1, 2).flatten(-2, -1)
    print(get_stats("attn_out before out_proj", context_BLD))
    
    # Out proj
    attn_out = block.attn.out_proj(context_BLD)
    print(get_stats("attn_out after out_proj", attn_out))
    
    # Residual
    x_attn = x + attn_out / block.scaling_factor
    print(get_stats("x after attention residual", x_attn))
    
    # FFN
    ffn_in = block.ffn[0](x_attn)
    print(get_stats("ffn_in (FFN Pre-LN)", ffn_in))
    
    ffn_fc1 = block.ffn[1](ffn_in)
    print(get_stats("ffn_fc1 (Linear 1)", ffn_fc1))
    
    ffn_act = block.ffn[2](ffn_fc1)
    print(get_stats("ffn_act (SwiGLU)", ffn_act))
    
    ffn_out = block.ffn[3](ffn_act)
    print(get_stats("ffn_out (Linear 2)", ffn_out))
    
    x_block = x_attn + ffn_out / block.scaling_factor
    print(get_stats("x after block 0", x_block))

print("\n--- Final Outputs ---")
with torch.no_grad():
    outputs = model(input_ids)
    print(get_stats("final sequence_logits", outputs.logits[0]))
