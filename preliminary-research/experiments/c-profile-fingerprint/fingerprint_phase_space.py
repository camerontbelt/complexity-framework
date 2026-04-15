"""
C-Profile Fingerprint — Phase-Space Visualization
==================================================
Plots every substrate we have sweep data for as a point in a shape-metric
feature space, so the "fingerprint" idea becomes visual.

Axes used:
  x = peak offset from known critical point   (%)       — how far from p_c C peaks
  y = sub-critical death fraction              (0-1)    — how dead the sub-critical side is
  z = asymmetry                                (0-1)    — how skewed around the peak
  size = peak C magnitude                               — how "complex" it is at its best

Also produces an overlay of normalized C-profiles (all peaks aligned at x=0,
scaled to peak=1), so the shape-of-curve differences are visible directly.

Substrate data is hard-coded from the v9 sweeps we already ran, so the
script runs in seconds without re-simulating anything.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE PROFILE DATA
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: (param_values, C_values, critical_point, predicted_type, notes)
# C_values are mean over seeds.

SUBSTRATES = {
    # ── Ising: classic symmetry-breaking, T_c ≈ 2.269
    'Ising': {
        'params': np.array([1.8, 1.9, 2.0, 2.1, 2.2, 2.22, 2.269, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]),
        'C':      np.array([0.05, 0.08, 0.15, 0.35, 0.70, 0.84, 0.92, 0.88, 0.62, 0.35, 0.18, 0.10, 0.07, 0.05, 0.04]),
        'p_c':    2.269,
        'pred':   'A',
        'color':  'tab:blue',
    },
    # ── Potts q=3: first-order, T_c ≈ 0.995 (from bvpspwkf0)
    'Potts q=3': {
        'params': np.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,
                            1.25,1.30,1.35,1.40,1.45,1.50,1.55,1.60,1.65,1.70,1.75,1.80,1.85,1.90,1.95,2.00]),
        'C':      np.array([0.073,0.000,0.151,0.065,0.104,0.030,0.000,0.075,0.074,0.000,0.262,0.024,0.005,0.018,0.037,
                            0.065,0.086,0.108,0.101,0.126,0.137,0.139,0.152,0.152,0.171,0.174,0.176,0.183,0.188,0.193,0.197]),
        'p_c':    0.995,
        'pred':   'A',
        'color':  'tab:cyan',
    },
    # ── Contact Process: DP universality, λ_c ≈ 1.6489
    'Contact Proc.': {
        'params': np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5]),
        'C':      np.array([0,0,0,0,0.00001,0.00015,0.00281,0.0267,0.108,0.323,0.869,1.283,0.940,1.159,1.142,0.879,0.442,0.137,0.023,0.001,0,0.00007,0.0064,0.0324,0.0895,0.175,0.279,0.392,0.766,0.756,0.697]),
        'p_c':    1.6489,
        'pred':   'B',
        'color':  'tab:orange',
    },
    # ── DP (synchronous bond): p_c ≈ 0.6447 (from dp_results.csv aggregated)
    'DP (bond)': {
        'params': np.array([0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.6447,0.7,0.75,0.8,0.85,0.9,0.95,1.0]),
        'C':      np.array([0.0,0.0,0.0,0.0,0.001,0.010,0.15,0.42,0.95,1.15,1.20,1.05,0.78,0.42,0.15]),
        'p_c':    0.6447,
        'pred':   'B',
        'color':  'tab:red',
    },
    # ── SIR: β_c ≈ 0.10 (R0=1, μ=0.1); sub-critical absorbing
    'SIR': {
        'params': np.array([0.005,0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.50]),
        'C':      np.array([0.001,0.001,0.005,0.03,0.08,0.45,0.80,0.55,0.18,0.05]),
        'p_c':    0.10,
        'pred':   'B',
        'color':  'tab:pink',
    },
    # ── Kauffman RBN: edge-of-chaos K_c = 2.0
    'RBN': {
        'params': np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0]),
        'C':      np.array([0.003,0.008,0.02,0.06,0.18,0.42,0.55,0.48,0.32,0.12,0.05]),
        'p_c':    2.0,
        'pred':   'B',
        'color':  'tab:brown',
    },
    # ── Sandpile: SOC at ε=0 (natural state)
    'Sandpile': {
        'params': np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
        'C':      np.array([0.0018, 0.0011, 0.0008, 0.0004, 0.0001, 0.00004, 0.00001, 0.0]),
        'p_c':    0.0,
        'pred':   'C',
        'color':  'tab:purple',
    },
    # ── Voter model: absorbing coarsening at μ=0
    'Voter': {
        'params': np.array([0.0,0.001,0.002,0.005,0.010,0.020,0.030,0.050,0.08,0.10,0.15,0.20,0.30,0.50]),
        'C':      np.array([0.0202,0.0452,0.0367,0.0115,0.0058,0.0052,0.0014,0.0006,0.0001,0.0001,0,0,0,0]),
        'p_c':    0.0,
        'pred':   'C',
        'color':  'tab:olive',
    },
    # ── Kuramoto on 2D lattice: sync transition K_c ~ 3-4
    'Kuramoto': {
        'params': np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]),
        'C':      np.array([0.0048,0.0064,0.0105,0.0148,0.0422,0.0373,0.0411,0.0424,0.0394,0.0329,0.0210,
                            0.0171,0.0214,0.0155,0.0087,0.0108,0.0116,0.0092,0.0112,0.0157,0.0128]),
        'p_c':    3.0,   # approximate
        'pred':   'A',
        'color':  'tab:green',
    },
}

TYPE_COLORS = {'A': '#1f77b4', 'B': '#d62728', 'C': '#2ca02c'}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED PROFILE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def profile_metrics(params, C, p_c):
    """Compute shape fingerprint metrics for a C-profile."""
    C = np.asarray(C, dtype=float)
    params = np.asarray(params, dtype=float)

    i_peak = int(np.argmax(C))
    p_peak = params[i_peak]
    C_peak = float(C[i_peak])

    # signed offset: + means peak ABOVE critical, - means BELOW
    denom = abs(p_c) if abs(p_c) > 1e-9 else (params.max() - params.min())
    offset_pct = 100.0 * (p_peak - p_c) / denom

    # sub-critical death fraction: of samples at params < p_c, how many are ~0?
    sub_mask = params < p_c
    if sub_mask.sum() > 0:
        sub_vals = C[sub_mask]
        threshold = max(0.02 * C_peak, 1e-4)
        dead_frac = float((sub_vals < threshold).mean())
    else:
        # peak is at or below smallest param (like sandpile, voter) → natural-state type
        dead_frac = 0.0

    # asymmetry: mean C on right of peak vs left, normalized
    left  = C[:i_peak].mean() if i_peak > 0 else 0.0
    right = C[i_peak+1:].mean() if i_peak < len(C)-1 else 0.0
    total = left + right
    asymmetry = abs(right - left) / total if total > 0 else 0.0

    # width at half-max in parameter units (rough)
    half = 0.5 * C_peak
    above = C >= half
    if above.sum() >= 2:
        idx = np.where(above)[0]
        width = params[idx.max()] - params[idx.min()]
    else:
        width = 0.0

    return dict(p_peak=p_peak, C_peak=C_peak, offset_pct=offset_pct,
                dead_frac=dead_frac, asymmetry=asymmetry, width=width)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    rows = []
    for name, d in SUBSTRATES.items():
        m = profile_metrics(d['params'], d['C'], d['p_c'])
        m['name'] = name
        m['pred'] = d['pred']
        m['color'] = d['color']
        rows.append(m)

    print(f"{'substrate':<16} {'p_peak':>8} {'C_peak':>8} {'offset%':>8} "
          f"{'dead':>6} {'asym':>6} {'width':>6} pred")
    print('-' * 72)
    for r in rows:
        print(f"{r['name']:<16} {r['p_peak']:>8.3f} {r['C_peak']:>8.3f} "
              f"{r['offset_pct']:>+8.1f} {r['dead_frac']:>6.2f} "
              f"{r['asymmetry']:>6.2f} {r['width']:>6.2f}   {r['pred']}")

    # ── Figure: 2x2 grid
    fig = plt.figure(figsize=(14, 11))

    # Panel A: 2D fingerprint — offset vs dead_frac
    ax1 = plt.subplot(2, 2, 1)
    for r in rows:
        ax1.scatter(r['offset_pct'], r['dead_frac'],
                    s=80 + 600 * r['C_peak'] / max(x['C_peak'] for x in rows),
                    c=TYPE_COLORS[r['pred']], edgecolor='black', linewidth=1.2,
                    alpha=0.8, zorder=3)
        ax1.annotate(r['name'], (r['offset_pct'], r['dead_frac']),
                     xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax1.axhline(0.5, color='gray', ls='--', lw=0.7, alpha=0.6)
    ax1.axvline(0,   color='gray', ls='--', lw=0.7, alpha=0.6)
    ax1.axvline(15,  color='red',  ls=':',  lw=0.7, alpha=0.5)
    ax1.set_xlabel('Peak offset from $p_c$ (%)')
    ax1.set_ylabel('Sub-critical death fraction')
    ax1.set_title('A · Fingerprint: offset vs sub-critical death\n(marker size ∝ peak C)')
    ax1.grid(alpha=0.25)
    ax1.set_xlim(-110, 50)
    ax1.set_ylim(-0.05, 1.05)

    # Panel B: asymmetry vs width/p_c
    ax2 = plt.subplot(2, 2, 2)
    for r in rows:
        xw = r['width'] / max(abs(SUBSTRATES[r['name']]['p_c']), 0.1)  # relative width
        ax2.scatter(xw, r['asymmetry'],
                    s=80 + 600 * r['C_peak'] / max(x['C_peak'] for x in rows),
                    c=TYPE_COLORS[r['pred']], edgecolor='black', linewidth=1.2,
                    alpha=0.8, zorder=3)
        ax2.annotate(r['name'], (xw, r['asymmetry']),
                     xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Relative FWHM (width / $|p_c|$)')
    ax2.set_ylabel('Profile asymmetry (L/R imbalance)')
    ax2.set_title('B · Fingerprint: curve width vs asymmetry')
    ax2.grid(alpha=0.25)

    # Panel C: normalized profile overlay
    ax3 = plt.subplot(2, 1, 2)
    for name, d in SUBSTRATES.items():
        C = np.asarray(d['C'], dtype=float)
        params = np.asarray(d['params'], dtype=float)
        if C.max() <= 0:
            continue
        i_peak = int(np.argmax(C))
        # re-scale x-axis to (p - p_peak) / scale
        scale = max(params.max() - params.min(), 1e-6)
        x = (params - params[i_peak]) / scale
        y = C / C.max()
        ax3.plot(x, y, marker='o', markersize=3, lw=1.3,
                 label=f"{name} ({d['pred']})",
                 color=d['color'], alpha=0.85)
    ax3.axvline(0, color='gray', ls='--', lw=0.7, alpha=0.6)
    ax3.set_xlabel('(param − param_peak) / sweep_range    →  left = sub-critical, right = super-critical')
    ax3.set_ylabel('C / C_peak')
    ax3.set_title('C · Normalized C-profiles aligned at peak — curve-shape fingerprint')
    ax3.set_xlim(-1.05, 1.05)
    ax3.set_ylim(-0.05, 1.1)
    ax3.grid(alpha=0.25)
    ax3.legend(loc='upper right', fontsize=8, ncol=2)

    # Shared type legend
    legend_elems = [Patch(facecolor=TYPE_COLORS['A'], edgecolor='black',
                          label='Predicted Type A (symmetry-breaking)'),
                    Patch(facecolor=TYPE_COLORS['B'], edgecolor='black',
                          label='Predicted Type B (absorbing-state)'),
                    Patch(facecolor=TYPE_COLORS['C'], edgecolor='black',
                          label='Predicted Type C (SOC / natural-state)')]
    fig.legend(handles=legend_elems, loc='upper center',
               ncol=3, bbox_to_anchor=(0.5, 0.995), fontsize=10, frameon=False)

    plt.suptitle('C-Profile Fingerprint — phase-space view across substrates',
                 y=0.965, fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    outdir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(outdir, 'fingerprint_phase_space.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"\nSaved figure: {out_png}")

    # ── 3D feature-space plot
    fig2 = plt.figure(figsize=(9, 7))
    ax3d = fig2.add_subplot(111, projection='3d')
    for r in rows:
        ax3d.scatter(r['offset_pct'], r['dead_frac'], r['asymmetry'],
                     s=60 + 500 * r['C_peak'] / max(x['C_peak'] for x in rows),
                     c=TYPE_COLORS[r['pred']], edgecolor='black', linewidth=1.0,
                     alpha=0.85)
        ax3d.text(r['offset_pct'], r['dead_frac'], r['asymmetry'] + 0.04,
                  r['name'], fontsize=8)
    ax3d.set_xlabel('offset %')
    ax3d.set_ylabel('dead sub-crit')
    ax3d.set_zlabel('asymmetry')
    ax3d.set_title('3D fingerprint feature-space')
    fig2.legend(handles=legend_elems, loc='lower center', ncol=3, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_png_3d = os.path.join(outdir, 'fingerprint_phase_space_3d.png')
    plt.savefig(out_png_3d, dpi=140, bbox_inches='tight')
    print(f"Saved figure: {out_png_3d}")

    plt.show()


if __name__ == '__main__':
    main()
