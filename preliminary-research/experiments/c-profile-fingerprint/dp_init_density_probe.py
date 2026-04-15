"""
DP vs Contact Process discrepancy — initial-density probe.

Hypothesis: the DP profile peak at +30% above p_c (vs CP peak AT p_c) is an
artifact of DP's sparse initial condition (INIT_DENSITY=0.02). With a dense
initial condition the sub-critical/marginal phase can keep activity alive
through the measurement window, so C should peak closer to p_c.

Run the same DP model (v9 _dp_run) with four init densities spanning
very-sparse to dense, and see where C peaks.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from complexity_framework_v9 import _dp_run, compute_C, DP_CFG

P_C     = DP_CFG['PC_EXACT']                  # 0.2873
P_VALS  = np.round(np.arange(0.20, 0.50, 0.025), 4)
N_SEEDS = 5
INIT_DENSITIES = [0.02, 0.10, 0.30, 0.50]     # 4 conditions

def sweep(init_density):
    cfg = DP_CFG.copy()
    cfg['INIT_DENSITY'] = init_density
    rows = []
    for p in P_VALS:
        Cs, dens = [], []
        for s in range(N_SEEDS):
            hist = _dp_run(p, cfg, seed=s)
            m = compute_C(hist, cfg['BURNIN'], cfg['WINDOW'])
            Cs.append(m['score'])
            dens.append(hist[cfg['BURNIN']:].mean())
        rows.append(dict(p=p, C=np.mean(Cs), density=np.mean(dens)))
    return rows

results = {}
t0 = time.time()
for rho in INIT_DENSITIES:
    print(f"sweeping init_density={rho}...")
    results[rho] = sweep(rho)
print(f"total {time.time()-t0:.1f}s")

# Print table
print(f"\n{'p':>6}  " + "  ".join(f"rho={rho}:C,den" for rho in INIT_DENSITIES))
for i, p in enumerate(P_VALS):
    row = f"{p:>6.3f}  "
    for rho in INIT_DENSITIES:
        r = results[rho][i]
        row += f"  {r['C']:>6.3f}/{r['density']:>4.2f}  "
    print(row)

# Find peak offsets
print(f"\n{'init_density':>12}  {'p_peak':>8}  {'C_peak':>8}  {'offset%':>8}")
for rho in INIT_DENSITIES:
    Cs = np.array([r['C'] for r in results[rho]])
    if Cs.max() == 0:
        print(f"{rho:>12.2f}  (all zero — dead)")
        continue
    i = int(Cs.argmax())
    p_peak = P_VALS[i]
    offset = 100.0 * (p_peak - P_C) / P_C
    print(f"{rho:>12.2f}  {p_peak:>8.3f}  {Cs.max():>8.3f}  {offset:>+8.1f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for rho, rows in results.items():
    p  = [r['p'] for r in rows]
    C  = [r['C'] for r in rows]
    den = [r['density'] for r in rows]
    ax1.plot(p, C,   marker='o', label=f'init ρ = {rho}')
    ax2.plot(p, den, marker='s', label=f'init ρ = {rho}')

for ax in (ax1, ax2):
    ax.axvline(P_C, color='red', ls='--', lw=1, label=f'$p_c$ = {P_C}')
    ax.legend(fontsize=9)
    ax.set_xlabel('p')
    ax.grid(alpha=0.3)
ax1.set_ylabel('C (mean over seeds)')
ax1.set_title('DP — C profile vs initial density')
ax2.set_ylabel('steady-state activity density')
ax2.set_title('DP — density vs initial density')
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'dp_init_density_probe.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out}")
