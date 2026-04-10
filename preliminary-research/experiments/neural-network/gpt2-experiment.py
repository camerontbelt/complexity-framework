"""
gpt2-experiment.py
==================
Two complementary experiments measuring complexity in GPT-2's residual stream.

  Exp E  Trained vs. Untrained GPT-2
         Same coherent text, different weight states.
         1 pre-trained GPT-2 vs N randomly initialised GPT-2s.
         Direct analogue of the CA experiment: does training drive the
         model toward a more dynamically complex regime?

  Exp F  Coherent vs. Random Input on the Trained GPT-2
         Same pre-trained model, two input types.
         5 coherent English passages vs 5 random uniform-token sequences.
         Tests whether the model's hidden-state dynamics reflect the
         structure of what it is processing.

Why GPT-2 is the right substrate
---------------------------------
Unlike a classifier, a language model cannot simplify its internal
representations without losing generative capability. Generating coherent
text requires hidden states that are neither frozen (repetitive output) nor
chaotic (random output) — precisely the edge-of-chaos regime the C metric
was designed to detect.

Temporal axis
-------------
GPT-2 is causally masked: h_t = f(tokens 0..t). Token position is
therefore a genuine time axis — h_{t+1} causally depends on h_t — giving
T = SEQ_LEN authentic temporal steps.

Volume format: one (1, 1, 768) tensor per token position.
T = SEQ_LEN = 200 tokens,  W = 768 hidden dimensions.
"""

import os
import csv as _csv
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as sp
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config

# =============================================================================
# Bootstrap — load compute_full_C from mnist-experiment.py
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_HERE, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# =============================================================================
# Configuration
# =============================================================================
P_THRESH = 0.05
D_THRESH = 0.50

SEQ_LEN       = 200          # tokens per sequence fed to the model
PRIMARY_LAYER = 6            # layer used for primary statistics (0=embed, 1-12=blocks)
PROFILE_LAYERS = [0, 3, 6, 9, 12]   # layers shown in the depth profile plot

# =============================================================================
# Coherent text passages  (written inline — no network dependency at run time)
# Five distinct passages covering themes relevant to this research.
# Each tokenises to ~160-200 GPT-2 tokens, padded to SEQ_LEN.
# =============================================================================
COHERENT_TEXTS = [
    # 1. Complexity science
    ("Cellular automata demonstrate how local rules produce global complexity. "
     "Wolfram's class-four automata, including Rule 110, sit at the edge between "
     "order and chaos where the richest computational structures emerge. These "
     "systems can support universal computation, store information indefinitely, "
     "and exhibit sensitive dependence on initial conditions without losing "
     "long-range coherence across the grid. The entropy of such systems is "
     "intermediate — neither the zero entropy of frozen states nor the maximum "
     "entropy of chaotic ones. This intermediate regime, sometimes called the "
     "edge of chaos, is where the most interesting dynamics occur. Information "
     "can propagate over long distances, structures can form and interact, and "
     "the system retains memory of its initial conditions while still evolving "
     "in non-trivial ways. The connection between edge-of-chaos dynamics and "
     "computational universality suggests a deep relationship between complexity "
     "and the capacity to process information."),

    # 2. Information theory
    ("Shannon's information theory provides a mathematical foundation for "
     "understanding the structure of communication and uncertainty. Entropy "
     "measures the average unpredictability of a source, while mutual information "
     "quantifies how much knowing one variable reduces uncertainty about another. "
     "Systems operating at intermediate entropy values tend to exhibit the richest "
     "structure — fully ordered systems carry no information, while maximally "
     "random ones carry no structure. The interplay between these two extremes "
     "defines a landscape of complexity that spans physics, biology, and "
     "computation. Compression algorithms exploit statistical regularities to "
     "represent data more compactly; the degree to which a sequence can be "
     "compressed reflects its algorithmic complexity, also known as Kolmogorov "
     "complexity. Systems at the edge of chaos are neither trivially compressible "
     "nor incompressible — they contain genuine structure at multiple scales."),

    # 3. Neural networks and representation learning
    ("Deep neural networks learn hierarchical representations of data through "
     "gradient descent on a loss function. Each successive layer transforms its "
     "input into progressively more abstract features, allowing the network to "
     "disentangle complex statistical dependencies. Language models extend this "
     "framework to sequences, learning to predict the next token from its context. "
     "The hidden state of a transformer accumulates information across token "
     "positions through masked self-attention, building a contextual representation "
     "that integrates local and long-range dependencies. In a well-trained model, "
     "this process creates internal dynamics that are sensitive to the structure "
     "of the input — semantic relationships, syntactic patterns, and pragmatic "
     "context all shape the evolution of the residual stream as it propagates "
     "through successive transformer blocks toward the final prediction."),

    # 4. Neuroscience and criticality
    ("The brain appears to operate near a critical point between ordered and "
     "disordered dynamics. Neuronal avalanches with power-law size distributions, "
     "branching ratios close to one, and long-range temporal correlations in "
     "neural activity all suggest that cortical networks sit at the edge of a "
     "phase transition. This critical regime maximises dynamic range, information "
     "transmission between regions, and the repertoire of distinguishable network "
     "states. Criticality may thus be the organising principle that allows the "
     "brain to be simultaneously stable and flexible. A brain that operates in "
     "the ordered regime would be too rigid to respond to novel stimuli; one "
     "operating in the chaotic regime would be too noisy to maintain coherent "
     "representations across time. The edge of chaos provides the optimal balance "
     "between these competing demands, enabling the complex cognitive functions "
     "that emerge from billions of interacting neurons."),

    # 5. Consciousness and integrated information
    ("Integrated information theory proposes that consciousness corresponds to "
     "the degree to which a physical system generates information above and beyond "
     "the information generated by its parts in isolation. A system with high "
     "integrated information, measured by the quantity phi, cannot be understood "
     "by decomposing it into independent subsystems — the causal interactions "
     "between parts give rise to a unified whole that is irreducible to any "
     "partition. This irreducibility is proposed as the mathematical signature "
     "of subjective experience. The theory predicts that systems with highly "
     "modular or feedforward architectures will have low phi, while systems with "
     "dense recurrent connectivity and differentiated responses will have high "
     "phi. Whether or not the theory is correct, it highlights a deep connection "
     "between the complexity of causal interactions within a system and the "
     "richness of the information it integrates and generates."),
]

# =============================================================================
# Shared helpers
# =============================================================================

def make_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def tokenize_text(text, tok, seq_len=SEQ_LEN):
    """Tokenise and truncate or right-pad to exactly seq_len tokens."""
    ids = tok(text.replace("\n", " "), return_tensors="pt")["input_ids"][0]
    if len(ids) >= seq_len:
        return ids[:seq_len].unsqueeze(0)
    pad = torch.full((seq_len - len(ids),), tok.eos_token_id, dtype=torch.long)
    return torch.cat([ids, pad]).unsqueeze(0)   # (1, seq_len)


def make_random_tokens(seq_len=SEQ_LEN, vocab_size=50257, seed=0):
    """Uniformly random token IDs — no linguistic structure whatsoever."""
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.randint(0, vocab_size, (1, seq_len)), dtype=torch.long)


def extract_all_hidden_states(model, input_ids):
    """
    Run ONE forward pass and return all layer hidden states as a list.
    Index 0 = embedding output, indices 1-12 = transformer block outputs.
    Each entry: numpy array of shape (T, 768).

    Single-pass caching avoids running the GPT-2 forward pass once per layer,
    which would multiply wall-clock time by the number of layers probed.
    """
    model.eval()
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    return [hs[0].float().numpy() for hs in out.hidden_states]   # list of (T, 768)


def hidden_to_volumes(h):
    """Convert a (T, 768) hidden-state array to a list of T (1,1,768) volumes."""
    return [h[t][np.newaxis, np.newaxis, :] for t in range(h.shape[0])]


def make_random_gpt2(seed=0):
    """GPT-2 small architecture with randomly initialised weights (no pretraining)."""
    torch.manual_seed(seed)
    model = GPT2Model(GPT2Config())
    model.eval()
    return model


def save_csv(filename, rows, fields):
    path = os.path.join(_HERE, filename)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"  CSV -> {path}")


def verdict_d(label, d, p):
    confirmed = d > D_THRESH and p < P_THRESH
    v = ("H1 CONFIRMED    [d>{:.1f}, p<{:.2f}]".format(D_THRESH, P_THRESH)
         if confirmed else
         "H0 NOT REJECTED [effect too small or not significant]")
    print(f"\n  RESULT: {v}")
    print(f"  {label}: Cohen d = {d:.3f},  p = {p:.4f}")
    return confirmed, v


# =============================================================================
# Experiment E — Trained vs. Untrained GPT-2
# =============================================================================
def exp_E(n_random=20):
    print("\n" + "=" * 70)
    print("EXP E  Trained vs. Untrained GPT-2")
    print()
    print("  Substrate : GPT-2 residual stream (768-dim per token)")
    print(f"  Time axis : {SEQ_LEN} token positions  (h_t = f(tokens 0..t), causal)")
    print(f"  Layer     : L{PRIMARY_LAYER}  (mid-network, primary stat)")
    print()
    print("  H1: The pre-trained GPT-2 produces more dynamically complex")
    print("      hidden-state dynamics (higher C_a at L6) than randomly")
    print("      initialised GPT-2s on the same coherent text.")
    print(f"      Threshold: Cohen d > {D_THRESH},  p < {P_THRESH}")
    print("  H0: No significant difference in C_a between trained and random GPT-2.")
    print("=" * 70)

    tok       = make_tokenizer()
    input_ids = tokenize_text(COHERENT_TEXTS[0], tok)
    print(f"\n  Input: passage 0  ({input_ids.shape[1]} tokens)")

    # --- Trained model — single forward pass, profile all layers ---
    print("\n  Loading pre-trained GPT-2...")
    trained = GPT2LMHeadModel.from_pretrained("gpt2")

    rows = []
    trained_layer_Ca = {}
    print("  Running single forward pass for trained model...")
    all_hs = extract_all_hidden_states(trained, input_ids)  # list of 13 (T, 768) arrays
    del trained   # free memory immediately after extraction

    print(f"  Computing C_a at layers {PROFILE_LAYERS}...")
    for l in PROFILE_LAYERS:
        vols = hidden_to_volumes(all_hs[l])
        r    = compute_full_C(vols)
        trained_layer_Ca[l] = r["C_a"]
        label = "embed" if l == 0 else f"L{l:02d}"
        print(f"    {label}: C_a={r['C_a']:.4f}  mi1={r['mi1']:.4f}  "
              f"tc={r['tc_mean']:.4f}  gzip={r['gzip_ratio']:.4f}  "
              f"op_dn={r['op_down']:.4f}")
        rows.append({"model": "trained", "seed": "pretrained", "layer": l, **r})

    # --- Random models — single forward pass each, primary layer only ---
    rand_Ca = []
    print(f"\n  Evaluating {n_random} random GPT-2s at L{PRIMARY_LAYER}...")
    for seed in range(n_random):
        rm      = make_random_gpt2(seed)
        all_hs_r = extract_all_hidden_states(rm, input_ids)
        del rm
        vols = hidden_to_volumes(all_hs_r[PRIMARY_LAYER])
        r    = compute_full_C(vols)
        rand_Ca.append(r["C_a"])
        rows.append({"model": "random", "seed": seed, "layer": PRIMARY_LAYER, **r})
        if seed % 5 == 0:
            print(f"    {seed:>2d}/{n_random}  C_a={r['C_a']:.4f}  mi1={r['mi1']:.4f}  "
                  f"tc={r['tc_mean']:.4f}")

    # --- Statistics ---
    t_Ca = trained_layer_Ca[PRIMARY_LAYER]
    ra   = np.array(rand_Ca)
    d    = (t_Ca - ra.mean()) / max(ra.std(), 1e-9)
    _, p = sp.ttest_1samp(rand_Ca, t_Ca)
    pct  = float(sp.percentileofscore(rand_Ca, t_Ca))

    print(f"\n  Trained C_a at L{PRIMARY_LAYER} : {t_Ca:.4f}")
    print(f"  Random mean C_a          : {ra.mean():.4f}  std={ra.std():.4f}")
    print(f"  Percentile               : {pct:.1f}th")
    confirmed, v = verdict_d(f"C_a at L{PRIMARY_LAYER} (trained vs. random GPT-2)", d, p)

    # --- CSV ---
    fields = ["model", "seed", "layer", "C", "C_a",
              "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
              "mean_H", "std_H", "op_up", "op_down",
              "mi1", "decay", "tc_mean", "gzip_ratio"]
    save_csv("expE_gpt2_trained_vs_random.csv", rows, fields)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rand_Ca, bins=20, alpha=0.7, color="steelblue",
                 label=f"Random GPT-2  (n={n_random})")
    axes[0].axvline(t_Ca, color="red", linestyle="--", linewidth=2,
                    label=f"Trained GPT-2  ({t_Ca:.4f})")
    axes[0].set_xlabel("C_a"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Agnostic C_a at Layer {PRIMARY_LAYER}")
    axes[0].legend(fontsize=9)
    axes[0].text(0.04, 0.95,
                 f"Rand mean  = {ra.mean():.4f}\nCohen d   = {d:.3f}\n"
                 f"p-value   = {p:.4f}\nPercentile = {pct:.1f}th",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85))

    # Layer profile of trained model
    prof_Ca = [trained_layer_Ca[l] for l in PROFILE_LAYERS]
    xlabels = ["embed" if l == 0 else f"L{l}" for l in PROFILE_LAYERS]
    axes[1].plot(PROFILE_LAYERS, prof_Ca, "r-o", label="Trained GPT-2", zorder=5)
    axes[1].axhline(ra.mean(), color="steelblue", linestyle="--", alpha=0.8,
                    label=f"Random mean at L{PRIMARY_LAYER}")
    axes[1].fill_between([min(PROFILE_LAYERS), max(PROFILE_LAYERS)],
                         ra.mean() - ra.std(), ra.mean() + ra.std(),
                         alpha=0.15, color="steelblue", label="Random ±1sd")
    axes[1].set_xticks(PROFILE_LAYERS); axes[1].set_xticklabels(xlabels)
    axes[1].set_xlabel("Layer (0=embed  →  12=final)")
    axes[1].set_ylabel("C_a")
    axes[1].set_title("C_a depth profile — trained model")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Exp E: GPT-2 Trained vs. Untrained\n{v}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expE_gpt2_trained_vs_random.png"), dpi=150)
    plt.show()
    return confirmed, trained_layer_Ca, rand_Ca


# =============================================================================
# Experiment F — Coherent vs. Random Input on Trained GPT-2
# =============================================================================
def exp_F():
    n = len(COHERENT_TEXTS)
    print("\n" + "=" * 70)
    print("EXP F  Coherent vs. Random Input on Trained GPT-2")
    print()
    print("  Substrate : pre-trained GPT-2, same weights throughout")
    print(f"  Inputs    : {n} coherent English passages")
    print(f"              {n} random uniform-token sequences  (vocab size 50 257)")
    print(f"  Layer     : L{PRIMARY_LAYER}   |   T={SEQ_LEN} tokens per sequence")
    print()
    print("  H1: The trained model produces more dynamically complex residual-")
    print("      stream dynamics (higher C_a) when processing coherent text than")
    print("      when processing random token sequences.")
    print(f"      Threshold: Cohen d > {D_THRESH},  p < {P_THRESH}")
    print("  H0: No significant difference in C_a between input types.")
    print("=" * 70)

    tok   = make_tokenizer()
    print("\n  Loading pre-trained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    rows        = []
    coherent_Ca = []
    random_Ca   = []

    print(f"\n  Processing {n} coherent passages at L{PRIMARY_LAYER}...")
    for i, text in enumerate(COHERENT_TEXTS):
        ids     = tokenize_text(text, tok)
        all_hs  = extract_all_hidden_states(model, ids)
        vols    = hidden_to_volumes(all_hs[PRIMARY_LAYER])
        r       = compute_full_C(vols)
        coherent_Ca.append(r["C_a"])
        rows.append({"input_type": "coherent", "passage_idx": i, **r})
        print(f"    Passage {i}: C_a={r['C_a']:.4f}  mi1={r['mi1']:.4f}  "
              f"tc={r['tc_mean']:.4f}  gzip={r['gzip_ratio']:.4f}  "
              f"op_dn={r['op_down']:.4f}")

    print(f"\n  Processing {n} random token sequences at L{PRIMARY_LAYER}...")
    for i in range(n):
        ids     = make_random_tokens(seq_len=SEQ_LEN, seed=i)
        all_hs  = extract_all_hidden_states(model, ids)
        vols    = hidden_to_volumes(all_hs[PRIMARY_LAYER])
        r       = compute_full_C(vols)
        random_Ca.append(r["C_a"])
        rows.append({"input_type": "random", "passage_idx": i, **r})
        print(f"    Random  {i}: C_a={r['C_a']:.4f}  mi1={r['mi1']:.4f}  "
              f"tc={r['tc_mean']:.4f}  gzip={r['gzip_ratio']:.4f}  "
              f"op_dn={r['op_down']:.4f}")

    del model

    # --- Statistics ---
    coh = np.array(coherent_Ca)
    rnd = np.array(random_Ca)
    d   = (coh.mean() - rnd.mean()) / max(rnd.std(), 1e-9)
    _, p = sp.ttest_ind(coherent_Ca, random_Ca)

    # Per-metric breakdown
    sub_keys = ["mi1", "decay", "tc_mean", "op_up", "op_down",
                "gzip_ratio", "mean_H", "C_a"]
    coh_rows = [r for r in rows if r["input_type"] == "coherent"]
    rnd_rows = [r for r in rows if r["input_type"] == "random"]

    print(f"\n  Coherent mean C_a : {coh.mean():.4f}  std={coh.std():.4f}")
    print(f"  Random   mean C_a : {rnd.mean():.4f}  std={rnd.std():.4f}")
    print("\n  -- Sub-metric breakdown (coherent vs. random) --")
    fmt = "  {:>14s}  coherent={:>8.4f}  random={:>8.4f}  d={:>6.2f}"
    for k in sub_keys:
        cv = np.array([r[k] for r in coh_rows])
        rv = np.array([r[k] for r in rnd_rows])
        dk = (cv.mean() - rv.mean()) / max(rv.std(), 1e-9)
        print(fmt.format(k, cv.mean(), rv.mean(), dk))

    confirmed, v = verdict_d(
        f"C_a at L{PRIMARY_LAYER} (coherent vs. random input)", d, p)

    # --- CSV ---
    fields = ["input_type", "passage_idx", "C", "C_a",
              "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
              "mean_H", "std_H", "op_up", "op_down",
              "mi1", "decay", "tc_mean", "gzip_ratio"]
    save_csv("expF_gpt2_coherent_vs_random_input.csv", rows, fields)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n)
    axes[0].plot(x, coherent_Ca, "b-o",
                 label=f"Coherent  (mean={coh.mean():.4f})")
    axes[0].plot(x, random_Ca,   "r-s",
                 label=f"Random    (mean={rnd.mean():.4f})")
    axes[0].set_xlabel("Passage / sequence index")
    axes[0].set_ylabel("C_a")
    axes[0].set_title(f"C_a at Layer {PRIMARY_LAYER}: coherent vs. random input")
    axes[0].legend(fontsize=9)
    axes[0].text(0.04, 0.95,
                 f"Cohen d = {d:.3f}\np = {p:.4f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85))

    # Sub-metric bar chart
    plot_keys   = ["mi1", "tc_mean", "gzip_ratio", "op_down", "C_a"]
    plot_labels = ["Temporal MI\n(lag 1)", "Temporal\ncompression",
                   "Gzip ratio", "Opacity\n(down)", "C_agnostic"]
    coh_vals = [np.mean([r[k] for r in coh_rows]) for k in plot_keys]
    rnd_vals = [np.mean([r[k] for r in rnd_rows]) for k in plot_keys]
    xb = np.arange(len(plot_keys)); w = 0.35
    axes[1].bar(xb - w/2, coh_vals, w, label="Coherent",
                color="royalblue", alpha=0.85)
    axes[1].bar(xb + w/2, rnd_vals, w, label="Random tokens",
                color="tomato",   alpha=0.85)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(plot_labels, fontsize=8)
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Sub-metric comparison")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Exp F: Coherent vs. Random Input on Trained GPT-2\n{v}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expF_gpt2_input_type.png"), dpi=150)
    plt.show()
    return confirmed, coh_rows, rnd_rows


# =============================================================================
# Main runner
# =============================================================================
if __name__ == "__main__":
    print("GPT-2 Complexity Experiments")
    print(f"Statistical thresholds: Cohen d > {D_THRESH},  p < {P_THRESH}")
    print(f"Sequence length: {SEQ_LEN} tokens   Primary layer: L{PRIMARY_LAYER}")

    E_result = exp_E()
    F_result = exp_F()

    E_confirmed = E_result[0]
    F_confirmed = F_result[0]

    print("\n\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print(f"Thresholds: Cohen d > {D_THRESH},  p < {P_THRESH}")
    print("=" * 70)
    for label, confirmed in [
        ("E  GPT-2 Trained vs. Untrained   (causal residual stream, L6)", E_confirmed),
        ("F  Coherent vs. Random Input     (same trained GPT-2, L6)     ", F_confirmed),
    ]:
        status = "H1 CONFIRMED" if confirmed else "H0 NOT REJECTED"
        print(f"  Exp {label}  [{status}]")
    print("=" * 70)
