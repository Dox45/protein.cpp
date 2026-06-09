import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np

# Load model and tokenizer
model_name = "biohub/ESMC-300M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# Sequence
seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
# Tokenize
inputs = tokenizer(seq, return_tensors="pt")
input_ids = inputs["input_ids"]

# Forward
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0] # Shape [L, V]

# Print top-5 for each position
# Note: input_ids[0] has CLS at 0, EOS at L-1
print("HF predictions:")
for i in range(1, len(input_ids[0]) - 1):
    tok_id = input_ids[0][i].item()
    aa = tokenizer.decode([tok_id])
    pos_logits = logits[i]
    probs = torch.softmax(pos_logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 5)
    print(f"pos {i} ({aa}):")
    for prob, idx in zip(topk_probs, topk_indices):
        pred_aa = tokenizer.decode([idx.item()])
        # Strip spaces
        pred_aa = pred_aa.strip()
        if not pred_aa:
            pred_aa = f"<{idx.item()}>"
        print(f"    {pred_aa:<8} {prob.item():.4f}")
