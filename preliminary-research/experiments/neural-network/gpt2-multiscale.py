"""
gpt2-multiscale.py
==================
Multi-scale temporal complexity analysis of GPT-2 hidden states.

Mirrors the spatial pooling approach in gray-scott-multiscale.py but applied
to the temporal axis of the GPT-2 residual stream.

Spatial pooling (Gray-Scott) → Temporal striding (GPT-2)
  Pool ×p: average p×p cells to one super-cell      → coarser spatial entity
  Stride s: sample every s-th token from T tokens   → coarser temporal entity

At stride s=1  we measure word-level temporal dynamics    (200 steps)
At stride s=2  we measure bi-gram scale                   (100 steps)
At stride s=4  we measure phrase / clause scale            (50 steps)
At stride s=8  we measure sentence-level dynamics          (25 steps)
At stride s=16 we measure paragraph-level dynamics         (12 steps)

The adaptive 25-% threshold from the Gray-Scott experiment is NOT applied here
— GPT-2 hidden states are already signed real-valued and the standard
(h > 0) binarisation gives naturally variable density.  This means wH_a
varies slightly across strides and conditions, which is itself informative.

Four conditions (2 × 2 design):
  trained_coherent   pre-trained GPT-2  +  coherent English text
  trained_random     pre-trained GPT-2  +  uniform random tokens
  rand_coherent      randomly-init GPT-2 + coherent English text
  rand_random        randomly-init GPT-2 + uniform random tokens

Hypotheses
----------
H1: The trained GPT-2 processing coherent text shows the broadest multi-scale
    profile — elevated C_a at more temporal strides — compared with the other
    three conditions. This would indicate that training on language creates
    genuine multi-scale temporal structure in the residual stream.
H0: All four conditions produce similar C_a-vs-stride profiles; no condition
    is consistently broader.
"""

import os
import csv as _csv
import importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config

# ===========================================================================
# Bootstrap — import compute_full_C from mnist-experiment.py
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_HERE, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ===========================================================================
# Configuration
# ===========================================================================
SEQ_LEN          = 200
PRIMARY_LAYER    = 6          # mid-network transformer block
TEMPORAL_STRIDES = [1, 2, 4, 8, 16]
N_RANDOM_MODELS  = 3          # random-weight GPT-2 seeds
MIN_T_STEPS      = 8          # skip stride if resulting T < this value

# Four experimental conditions
CONDITIONS = [
    ("trained_coherent",  "Trained GPT-2  + Coherent text",   "royalblue",    "-",  2.5),
    ("trained_random",    "Trained GPT-2  + Random tokens",   "cornflowerblue","--", 2.0),
    ("rand_coherent",     "Random GPT-2   + Coherent text",   "dimgray",       "-",  2.0),
    ("rand_random",       "Random GPT-2   + Random tokens",   "lightgray",    "--", 1.5),
]

# Coherent passages (from gpt2-experiment.py — same set)
COHERENT_TEXTS = [
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
     "representations across time."),

    ("Integrated information theory proposes that consciousness corresponds to "
     "the degree to which a physical system generates information above and beyond "
     "the information generated by its parts in isolation. A system with high "
     "integrated information, measured by the quantity phi, cannot be understood "
     "by decomposing it into independent subsystems — the causal interactions "
     "between parts give rise to a unified whole that is irreducible to any "
     "partition. This irreducibility is proposed as the mathematical signature "
     "of subjective experience. The theory predicts that systems with highly "
     "modular or feedforward architectures will have low phi, while systems with "
     "dense recurrent connectivity and differentiated responses will have high phi."),
]

# ===========================================================================
# Helpers
# ===========================================================================

def make_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def tokenize_text(text, tok, seq_len=SEQ_LEN):
    ids = tok(text.replace("\n", " "), return_tensors="pt")["input_ids"][0]
    if len(ids) >= seq_len:
        return ids[:seq_len].unsqueeze(0)
    pad = torch.full((seq_len - len(ids),), tok.eos_token_id, dtype=torch.long)
    return torch.cat([ids, pad]).unsqueeze(0)


def make_random_tokens(seq_len=SEQ_LEN, vocab_size=50257, seed=0):
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.randint(0, vocab_size, (1, seq_len)), dtype=torch.long)


def extract_hidden(model, input_ids, layer=PRIMARY_LAYER):
    """Single forward pass; return (T, 768) hidden state at `layer`."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    return out.hidden_states[layer][0].float().numpy()   # (T, 768)


def make_random_gpt2(seed=0):
    torch.manual_seed(seed)
    return GPT2Model(GPT2Config())


def strided_volumes(h, stride):
    """Subsample (T, 768) hidden state at given temporal stride."""
    indices = np.arange(0, h.shape[0], stride)
    return [h[i][np.newaxis, np.newaxis, :] for i in indices]


def compute_profile(h, strides=TEMPORAL_STRIDES, min_T=MIN_T_STEPS):
    """
    Compute C_a and sub-metrics at each temporal stride.
    Returns dict {stride: result_dict}.
    """
    profile = {}
    for s in strides:
        vols = strided_volumes(h, s)
        if len(vols) < min_T:
            continue
        r = compute_profile_entry(vols)
        profile[s] = r
    return profile


def compute_profile_entry(vols):
    return compute_full_C(vols)


def profile_means(profiles_list, strides=TEMPORAL_STRIDES):
    """Aggregate a list of per-trial profiles into mean/std per stride."""
    out = {}
    for s in strides:
        vals = [p[s]["C_a"] for p in profiles_list if s in p]
        subs = {mk: [p[s][mk] for p in profiles_list if s in p]
                for mk in ["mi1", "tc_mean", "gzip_ratio", "op_down", "mean_H"]}
        if vals:
            out[s] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "n":    len(vals),
                **{mk: float(np.mean(subs[mk])) for mk in subs},
            }
    return out


def breadth(profile_agg, threshold_frac=0.80):
    """Number of strides within threshold_frac of the peak mean C_a."""
    vals = {s: profile_agg[s]["mean"] for s in profile_agg}
    if not vals:
        return 0, None
    peak = max(vals.values())
    cutoff = peak * threshold_frac
    broad  = sum(1 for v in vals.values() if v >= cutoff)
    peak_s = max(vals, key=vals.get)
    return broad, peak_s


# ===========================================================================
# Experiment
# ===========================================================================

def run_experiment():
    print("\n" + "="*72)
    print("GPT-2 Multi-Scale Temporal Complexity Experiment")
    print(f"  Layer L{PRIMARY_LAYER}  |  T={SEQ_LEN} tokens  |  strides {TEMPORAL_STRIDES}")
    print(f"  {len(COHERENT_TEXTS)} coherent passages  |  {len(COHERENT_TEXTS)} random token sequences")
    print(f"  {N_RANDOM_MODELS} random GPT-2 models")
    print()
    print("  H1: trained GPT-2 + coherent text shows the broadest multi-scale")
    print("      C_a profile (elevated at most temporal strides).")
    print("  H0: All four conditions produce indistinguishable profiles.")
    print("="*72)

    tok = make_tokenizer()

    # Build all input tensors up-front
    coherent_ids = [tokenize_text(t, tok) for t in COHERENT_TEXTS]
    random_ids   = [make_random_tokens(seed=i) for i in range(len(COHERENT_TEXTS))]

    # ---- Trained GPT-2 ----
    print("\n  Loading pre-trained GPT-2...")
    trained_model = GPT2LMHeadModel.from_pretrained("gpt2")

    tc_profiles = []   # trained + coherent
    tr_profiles = []   # trained + random tokens

    print(f"  Trained GPT-2 + coherent text (L{PRIMARY_LAYER})...")
    for i, ids in enumerate(coherent_ids):
        h = extract_hidden(trained_model, ids)
        p = compute_profile(h)
        tc_profiles.append(p)
        line = "    passage {}: ".format(i) + "  ".join(
            "s{} {:.3f}".format(s, p[s]["C_a"]) for s in TEMPORAL_STRIDES if s in p)
        print(line)

    print(f"  Trained GPT-2 + random tokens (L{PRIMARY_LAYER})...")
    for i, ids in enumerate(random_ids):
        h = extract_hidden(trained_model, ids)
        p = compute_profile(h)
        tr_profiles.append(p)
        line = "    random   {}: ".format(i) + "  ".join(
            "s{} {:.3f}".format(s, p[s]["C_a"]) for s in TEMPORAL_STRIDES if s in p)
        print(line)

    del trained_model

    # ---- Random GPT-2 ----
    rc_profiles = []   # random model + coherent
    rr_profiles = []   # random model + random tokens

    print(f"\n  {N_RANDOM_MODELS} random GPT-2 models, coherent + random tokens each...")
    for seed in range(N_RANDOM_MODELS):
        rmodel = make_random_gpt2(seed)
        for i, ids in enumerate(coherent_ids):
            h = extract_hidden(rmodel, ids)
            p = compute_profile(h)
            rc_profiles.append(p)
        for i, ids in enumerate(random_ids):
            h = extract_hidden(rmodel, ids)
            p = compute_profile(h)
            rr_profiles.append(p)
        peak_s_tc = max(tc_profiles[0], key=lambda s: tc_profiles[0][s]["C_a"]) if tc_profiles else "?"
        h_c = extract_hidden(rmodel, coherent_ids[0])
        p_c = compute_profile(h_c)
        line = "    seed {}: coh s1={:.3f}".format(seed, p_c.get(1, {}).get("C_a", 0))
        print(line)
        del rmodel

    # ---- Aggregate ----
    agg = {
        "trained_coherent": profile_means(tc_profiles),
        "trained_random":   profile_means(tr_profiles),
        "rand_coherent":    profile_means(rc_profiles),
        "rand_random":      profile_means(rr_profiles),
    }

    # ---- Breadth summary ----
    print("\n" + "-"*72)
    print("Multi-scale C_a summary:")
    print(f"  {'Condition':35s}  {'peak stride':11s}  breadth  " +
          "  ".join(f"s{s:>2d}" for s in TEMPORAL_STRIDES))
    print("-"*72)
    for cid, clabel, *_ in CONDITIONS:
        ag   = agg.get(cid, {})
        br, ps = breadth(ag)
        vals = "  ".join(f"{ag[s]['mean']:.3f}" if s in ag else "  --- "
                         for s in TEMPORAL_STRIDES)
        print(f"  {clabel:35s}  s={ps!s:<10}  {br}/{len(ag)}   {vals}")

    # ---- Statistics: trained_coherent vs. trained_random at each stride ----
    print("\n  Stride-by-stride: trained coherent vs. trained random tokens")
    for s in TEMPORAL_STRIDES:
        tc_vals = [p[s]["C_a"] for p in tc_profiles if s in p]
        tr_vals = [p[s]["C_a"] for p in tr_profiles if s in p]
        if len(tc_vals) < 2 or len(tr_vals) < 2:
            continue
        d  = (np.mean(tc_vals) - np.mean(tr_vals)) / max(np.std(tr_vals), 1e-9)
        _, p_val = sp.ttest_ind(tc_vals, tr_vals)
        print(f"    stride s={s:2d}:  coh={np.mean(tc_vals):.4f}  "
              f"rnd={np.mean(tr_vals):.4f}  d={d:.2f}  p={p_val:.4f}")

    # ---- CSV output ----
    csv_rows = []
    for cid, clabel, *_ in CONDITIONS:
        profiles = {"trained_coherent": tc_profiles, "trained_random":   tr_profiles,
                    "rand_coherent":    rc_profiles, "rand_random":       rr_profiles}[cid]
        for trial_i, prof in enumerate(profiles):
            for s, r in prof.items():
                csv_rows.append({
                    "condition": cid, "trial": trial_i, "stride": s,
                    "T_eff": SEQ_LEN // s,
                    **{mk: r[mk] for mk in ["C_a", "mi1", "decay", "tc_mean",
                                            "gzip_ratio", "op_up", "op_down",
                                            "mean_H", "wH_a", "wOPs_a",
                                            "wOPt_a", "wT_a", "wG_a"]},
                })

    csv_path = os.path.join(_HERE, "gpt2_multiscale.csv")
    fields   = ["condition", "trial", "stride", "T_eff",
                "C_a", "mi1", "decay", "tc_mean", "gzip_ratio", "op_up", "op_down",
                "mean_H", "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a"]
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\n  CSV  ->  {csv_path}")

    all_profiles = {
        "trained_coherent": tc_profiles,
        "trained_random":   tr_profiles,
        "rand_coherent":    rc_profiles,
        "rand_random":      rr_profiles,
    }
    return agg, all_profiles


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_results(agg, all_profiles, output_path):
    """
    Layout
    ------
    Row 0 (full width): Combined overview — 4 condition lines, C_a vs. stride.
    Row 1: 4 individual condition panels, each showing per-trial thin lines
           and mean as thick coloured line.
    """
    fig = plt.figure(figsize=(22, 14))
    gs  = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1],
                   hspace=0.45, wspace=0.32)

    ax_comb = fig.add_subplot(gs[0, :])

    cond_map = {cid: (clabel, cc, ls, lw)
                for cid, clabel, cc, ls, lw in CONDITIONS}

    # ---- Combined overview ----
    for cid, clabel, color, ls, lw in CONDITIONS:
        ag = agg.get(cid, {})
        x  = [s for s in TEMPORAL_STRIDES if s in ag]
        y  = [ag[s]["mean"] for s in x]
        e  = [ag[s]["std"]  for s in x]

        ax_comb.plot(x, y, marker="o", markersize=8, linewidth=lw,
                     color=color, linestyle=ls, label=clabel)
        ax_comb.fill_between(x,
                             [m - s_e for m, s_e in zip(y, e)],
                             [m + s_e for m, s_e in zip(y, e)],
                             alpha=0.12, color=color)

    ax_comb.set_xscale("log", base=2)
    ax_comb.set_xticks(TEMPORAL_STRIDES)
    ax_comb.set_xticklabels(
        [f"stride ×{s}\n(T={SEQ_LEN//s} steps)" for s in TEMPORAL_STRIDES],
        fontsize=9)
    ax_comb.set_xlabel("Temporal stride  (→ coarser scale,  fewer, longer-range steps)",
                       fontsize=10)
    ax_comb.set_ylabel("C_a  (agnostic complexity)", fontsize=10)
    ax_comb.set_title(
        "Multi-Scale Temporal Complexity Profile — GPT-2 (Layer 6)",
        fontsize=13, fontweight="bold")
    ax_comb.legend(fontsize=9, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.9)
    ax_comb.grid(True, alpha=0.25)
    ax_comb.set_ylim(bottom=0)

    ax_comb.text(
        0.01, 0.97,
        "H1: trained GPT-2 + coherent text shows highest C_a across most temporal scales "
        "(broadest profile).  If confirmed, the metric detects multi-scale linguistic structure.",
        transform=ax_comb.transAxes, fontsize=8.5, va="top",
        bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                  alpha=0.85, boxstyle="round,pad=0.4"))

    # ---- Individual condition panels ----
    for ci, (cid, clabel, color, ls, lw) in enumerate(CONDITIONS):
        ax  = fig.add_subplot(gs[1, ci])
        profiles = all_profiles.get(cid, [])
        ag  = agg.get(cid, {})

        # Per-trial thin lines
        for prof in profiles:
            x = [s for s in TEMPORAL_STRIDES if s in prof]
            y = [prof[s]["C_a"] for s in x]
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.35, linestyle=ls)

        # Mean ± std
        x = [s for s in TEMPORAL_STRIDES if s in ag]
        y = [ag[s]["mean"] for s in x]
        e = [ag[s]["std"]  for s in x]
        ax.plot(x, y, marker="o", markersize=8, linewidth=2.5,
                color=color, linestyle=ls, label="Mean C_a", zorder=4)
        ax.fill_between(x,
                        [m - s_e for m, s_e in zip(y, e)],
                        [m + s_e for m, s_e in zip(y, e)],
                        alpha=0.2, color=color)

        # Peak marker
        if y:
            pi = y.index(max(y))
            ax.plot(x[pi], y[pi], "*", markersize=13, color=color,
                    markeredgecolor="k", markeredgewidth=0.8, zorder=5)
            ax.axvline(x[pi], color=color, linestyle=":", alpha=0.4, linewidth=1.2)

        # Breadth annotation
        br, ps = breadth(ag)
        ax.text(0.97, 0.97,
                f"peak s={ps}\nbreadth {br}/{len(ag)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.85,
                          boxstyle="round,pad=0.3"))

        ax.set_xscale("log", base=2)
        ax.set_xticks(TEMPORAL_STRIDES)
        ax.set_xticklabels([f"x{s}" for s in TEMPORAL_STRIDES], fontsize=8)
        ax.set_xlabel("Temporal stride", fontsize=8)
        ax.set_ylabel("C_a", fontsize=8)
        ax.set_ylim(bottom=0)
        n_trials = len(profiles)
        ax.set_title(f"{clabel}\n({n_trials} trials)", fontsize=8.5,
                     color=color, fontweight="bold", pad=4)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"GPT-2 Multi-Scale Temporal Complexity Analysis\n"
        f"Layer L{PRIMARY_LAYER}  |  {SEQ_LEN} tokens  |  strides {TEMPORAL_STRIDES}  |  "
        f"{len(COHERENT_TEXTS)} passages  |  {N_RANDOM_MODELS} random models",
        fontsize=12, fontweight="bold", y=1.005)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot ->  {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    agg, all_profiles = run_experiment()
    png_path = os.path.join(_HERE, "gpt2_multiscale.png")
    plot_results(agg, all_profiles, png_path)
    print("\nDone.")
