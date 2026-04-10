"""Quick inspection of GPT-2's hidden states — shows what we can extract."""
from transformers import GPT2Model, GPT2Tokenizer
import torch
import numpy as np

print("Downloading GPT-2 small (124M params)...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
gpt.eval()

print()
print("=== Architecture ===")
cfg = gpt.config
print(f"  Layers (transformer blocks) : {cfg.n_layer}")
print(f"  Attention heads per layer   : {cfg.n_head}")
print(f"  Hidden / residual dimension : {cfg.n_embd}")
print(f"  Vocabulary size             : {cfg.vocab_size}")
print(f"  Max sequence length         : {cfg.n_positions}")
print(f"  Total parameters            : {sum(p.numel() for p in gpt.parameters()):,}")

# --- Two contrasting prompts ------------------------------------------------
prompts = {
    "Coherent English":
        "The complexity of a dynamical system can be measured by examining "
        "the entropy of its state transitions over time. Systems that sit at "
        "the edge of chaos exhibit rich structure that is neither fully ordered "
        "nor completely random. This property",

    "Random tokens (gibberish)":
        "xqzt mflp wrvk bzzj qqq hhh zzz aaa bbb ccc ddd eee fff ggg hhh "
        "iii jjj kkk lll mmm nnn ooo ppp qqq rrr sss ttt uuu vvv www xxx "
        "yyy zzz aaa bbb ccc ddd eee fff ggg hhh iii jjj",
}

for label, text in prompts.items():
    tokens = tok(text, return_tensors="pt", truncation=True, max_length=64)
    seq_len = tokens["input_ids"].shape[1]

    with torch.no_grad():
        out = gpt(**tokens)

    hidden_states = out.hidden_states   # len = 13 (embed + 12 layers)

    print()
    print(f"=== {label} ===")
    print(f"  Sequence length : {seq_len} tokens")
    print(f"  Tensors returned: {len(hidden_states)}  "
          f"(1 embedding + {len(hidden_states)-1} transformer layers)")
    print(f"  Each tensor shape: {hidden_states[0].shape}  "
          f"(batch, seq_len, d_model={cfg.n_embd})")
    print()
    print(f"  {'Layer':>6}  {'mean':>8}  {'std':>8}  {'min':>8}  "
          f"{'max':>8}  {'frac>0':>8}  {'L2_norm':>9}")
    for i, hs in enumerate(hidden_states):
        h = hs[0].float().numpy()           # (seq, 768)
        lbl = "embed" if i == 0 else f"L{i:02d}"
        l2  = float(np.linalg.norm(h, axis=-1).mean())
        print(f"  {lbl:>6}  {h.mean():8.4f}  {h.std():8.4f}  "
              f"{h.min():8.4f}  {h.max():8.4f}  {(h>0).mean():8.4f}  {l2:9.2f}")

print()
print("=== What we can feed to compute_full_C ===")
print("  Option A (tokens-as-time, one layer):")
print("    volumes: list of (1, 1, 768) tensors — one per token position")
print("    T = seq_len (e.g. 64)  W = 768  — causal temporal order!")
print()
print("  Option B (layers-as-time, one token position):")
print("    volumes: list of (1, 1, 768) tensors — one per layer")
print("    T = 13 layers  W = 768  — information propagation depth")
print()
print("  Option C (tokens-as-time, all layers stacked):")
print("    volumes: (n_layers, 1, 768) per token -> T=seq_len, W=13*768=9984")
print("    Richest view: space = all layers, time = token sequence")
print()
print("Best for C metric: Option A with a long text input (T=256+ tokens).")
print("Tokens are genuinely causal: token t+1 follows t in the residual stream.")
