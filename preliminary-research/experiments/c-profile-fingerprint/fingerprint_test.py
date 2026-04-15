"""
C-Profile Fingerprint Test
===========================
Blinded test of the C-profile taxonomy hypothesis.

Predictions (recorded BEFORE running):
  Contact Process  → Type B (absorbing-state, DP universality class)
  Voter Model      → Type B (absorbing-state, different universality)
  Kuramoto         → Type A (symmetry-breaking / sync transition)
  Potts q=3        → Type A (symmetry-breaking, first-order)

Each substrate is swept across its control parameter, and the C profile
shape is measured to classify it.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from complexity_framework_v9 import compute_C


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE 1: CONTACT PROCESS (DP universality class)
# ═══════════════════════════════════════════════════════════════════════════════
# Standard contact process on 2D lattice.
# Each active site recovers with rate 1, infects each neighbour with rate λ.
# Absorbing-state transition at λ_c ≈ 1.6489 (2D, von Neumann).
# PREDICTION: Type B — absorbing sub-critical phase, peak offset above λ_c.

def contact_process_run(lam, G=128, steps=400, burnin=50, window=200, seed=42):
    """Synchronous contact process on 2D toroidal grid.

    Each step:
      - Active sites die with probability 1/(1+λ)
      - Each active site independently infects each neighbour with prob λ/(4*(1+λ))
    Equivalent to discrete-time CP at rate λ.
    """
    rng  = np.random.default_rng(seed)
    grid = (rng.random((G, G)) < 0.5).astype(np.int8)
    history = []

    p_die = 1.0 / (1.0 + lam)
    p_infect = lam / (4.0 * (1.0 + lam))  # per-neighbour infection prob

    for t in range(steps):
        history.append(grid.ravel().copy())

        # Death: each active site dies with prob p_die
        dies = grid & (rng.random((G, G)) < p_die)

        # Infection: each neighbour of active site infects independently
        padded = np.pad(grid, 1, mode='wrap')
        # Each active neighbour tries to infect with p_infect
        # P(get infected) = 1 - (1 - p_infect)^(number of active neighbours)
        nb_sum = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                  padded[1:-1, :-2] + padded[1:-1, 2:]).astype(np.float32)
        p_activate = 1.0 - (1.0 - p_infect) ** nb_sum
        births = (~grid.astype(bool)) & (rng.random((G, G)) < p_activate)

        new_grid = grid.copy()
        new_grid[dies.astype(bool)] = 0
        new_grid[births] = 1
        grid = new_grid.astype(np.int8)

        if grid.sum() == 0:
            for _ in range(steps - t - 1):
                history.append(np.zeros(G*G, dtype=np.int8))
            break

    return np.array(history, dtype=np.int8)


def contact_process_sweep(verbose=True):
    G, STEPS, BURNIN, WINDOW, N_SEEDS = 128, 400, 50, 200, 8
    LAM_C = 1.6489
    lam_vals = np.round(np.arange(0.5, 3.55, 0.1), 2)

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONTACT PROCESS — lambda sweep")
        print(f"  Grid: {G}x{G}  Seeds: {N_SEEDS}  lambda_c ~ {LAM_C}")
        print(f"  PREDICTION: Type B (absorbing-state, DP universality)")
        print(f"{'='*70}")
        print(f"  {'lam':>5}  {'C':>10}  {'density':>8}  {'gzip':>8}  {'wOPs':>6}  {'wOPt':>6}")

    rows = []
    for lam in lam_vals:
        scores, densities, gzips, wops_s, wops_t = [], [], [], [], []
        for s in range(N_SEEDS):
            hist = contact_process_run(lam, G, STEPS, BURNIN, WINDOW, seed=s*7+3)
            m = compute_C(hist, BURNIN, WINDOW)
            scores.append(m['score'])
            post = hist[BURNIN:BURNIN+WINDOW]
            densities.append(float(post.mean()))
            gzips.append(m['gzip'])
            wops_s.append(m['w_OP_s'])
            wops_t.append(m['w_OP_t'])

        avg_C = np.mean(scores)
        marker = ' << lam_c' if abs(lam - LAM_C) < 0.06 else ''
        rows.append(dict(lam=float(lam), C=float(avg_C),
                         density=float(np.mean(densities)),
                         gzip=float(np.mean(gzips)),
                         wOPs=float(np.mean(wops_s)),
                         wOPt=float(np.mean(wops_t))))
        if verbose:
            print(f"  l={lam:>4.1f}  C={avg_C:>10.5f}  den={np.mean(densities):>7.4f}  "
                  f"gz={np.mean(gzips):>7.4f}  wOPs={np.mean(wops_s):>5.3f}  "
                  f"wOPt={np.mean(wops_t):>5.3f}{marker}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE 2: VOTER MODEL
# ═══════════════════════════════════════════════════════════════════════════════
# Each cell copies the state of a random neighbour.
# With mutation rate mu, each cell flips randomly with probability mu.
# Absorbing states at mu=0 (consensus). Phase transition in coarsening
# dynamics — with mu>0, ordered domains vs disordered mixing.
# PREDICTION: Type B — absorbing-state (consensus) at low mu.

def voter_model_run(mu, G=128, steps=400, burnin=50, window=200, seed=42):
    """Voter model with mutation on 2D toroidal grid.

    Each step (synchronous):
      - Each cell copies a random von Neumann neighbour
      - With probability mu, cell mutates to random state instead
    """
    rng  = np.random.default_rng(seed)
    grid = (rng.random((G, G)) < 0.5).astype(np.int8)
    history = []

    for t in range(steps):
        history.append(grid.ravel().copy())

        # Each cell picks a random neighbour direction
        direction = rng.integers(0, 4, size=(G, G))

        # Get neighbour values
        nb = np.empty_like(grid)
        nb[direction == 0] = np.roll(grid, 1, axis=0)[direction == 0]   # up
        nb[direction == 1] = np.roll(grid, -1, axis=0)[direction == 1]  # down
        nb[direction == 2] = np.roll(grid, 1, axis=1)[direction == 2]   # left
        nb[direction == 3] = np.roll(grid, -1, axis=1)[direction == 3]  # right

        # Copy neighbour
        new_grid = nb.copy()

        # Mutation: flip to random state
        mutants = rng.random((G, G)) < mu
        new_grid[mutants] = rng.integers(0, 2, size=mutants.sum()).astype(np.int8)

        grid = new_grid.astype(np.int8)

    return np.array(history, dtype=np.int8)


def voter_model_sweep(verbose=True):
    G, STEPS, BURNIN, WINDOW, N_SEEDS = 128, 400, 50, 200, 8
    mu_vals = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
               0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    if verbose:
        print(f"\n{'='*70}")
        print(f"VOTER MODEL — mutation rate sweep")
        print(f"  Grid: {G}x{G}  Seeds: {N_SEEDS}")
        print(f"  PREDICTION: Type B (absorbing at mu=0, complex at intermediate mu)")
        print(f"{'='*70}")
        print(f"  {'mu':>6}  {'C':>10}  {'density':>8}  {'gzip':>8}  {'wOPs':>6}  {'wOPt':>6}")

    rows = []
    for mu in mu_vals:
        scores, densities, gzips, wops_s, wops_t = [], [], [], [], []
        for s in range(N_SEEDS):
            hist = voter_model_run(mu, G, STEPS, BURNIN, WINDOW, seed=s*7+3)
            m = compute_C(hist, BURNIN, WINDOW)
            scores.append(m['score'])
            post = hist[BURNIN:BURNIN+WINDOW]
            densities.append(float(post.mean()))
            gzips.append(m['gzip'])
            wops_s.append(m['w_OP_s'])
            wops_t.append(m['w_OP_t'])

        avg_C = np.mean(scores)
        rows.append(dict(mu=float(mu), C=float(avg_C),
                         density=float(np.mean(densities)),
                         gzip=float(np.mean(gzips)),
                         wOPs=float(np.mean(wops_s)),
                         wOPt=float(np.mean(wops_t))))
        if verbose:
            print(f"  mu={mu:>5.3f}  C={avg_C:>10.5f}  den={np.mean(densities):>7.4f}  "
                  f"gz={np.mean(gzips):>7.4f}  wOPs={np.mean(wops_s):>5.3f}  "
                  f"wOPt={np.mean(wops_t):>5.3f}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE 3: KURAMOTO OSCILLATORS (on 2D lattice)
# ═══════════════════════════════════════════════════════════════════════════════
# Coupled phase oscillators on a 2D grid.
# theta_i(t+1) = theta_i(t) + omega_i + (K/4) * sum_j sin(theta_j - theta_i)
# Synchronisation transition at K_c.
# Binary encoding: cell is 1 if phase > pi, else 0.
# PREDICTION: Type A (symmetry-breaking — desync/sync phases both dense).

def kuramoto_run(K, G=64, steps=400, burnin=100, window=200, dt=0.1, seed=42):
    """Kuramoto oscillators on 2D toroidal grid.

    Each oscillator has natural frequency omega_i ~ N(0, 1).
    Coupling K/4 to each von Neumann neighbour.
    Binary encoding: phase > pi → 1, else → 0.
    """
    rng = np.random.default_rng(seed)

    # Natural frequencies (fixed per realization)
    omega = rng.standard_normal((G, G)).astype(np.float32)

    # Random initial phases
    theta = rng.uniform(0, 2*np.pi, (G, G)).astype(np.float32)

    history = []

    for t in range(steps):
        # Binary encoding: phase in [pi, 2pi) → 1, else → 0
        binary = (theta % (2*np.pi) > np.pi).astype(np.int8)
        history.append(binary.ravel())

        # Coupling: sum of sin(theta_j - theta_i) over 4 neighbours
        coupling = (np.sin(np.roll(theta, 1, 0) - theta) +
                    np.sin(np.roll(theta, -1, 0) - theta) +
                    np.sin(np.roll(theta, 1, 1) - theta) +
                    np.sin(np.roll(theta, -1, 1) - theta))

        theta = theta + dt * (omega + (K / 4.0) * coupling)

    return np.array(history, dtype=np.int8)


def kuramoto_sweep(verbose=True):
    G, STEPS, BURNIN, WINDOW, N_SEEDS = 64, 400, 100, 200, 8
    K_vals = np.round(np.arange(0.0, 10.5, 0.5), 1)
    # For 2D lattice Kuramoto with standard normal frequencies,
    # K_c is not analytically known but is around K ~ 2-4.

    if verbose:
        print(f"\n{'='*70}")
        print(f"KURAMOTO OSCILLATORS — coupling sweep (2D lattice)")
        print(f"  Grid: {G}x{G}  Seeds: {N_SEEDS}  dt=0.1")
        print(f"  PREDICTION: Type A (sync transition, both phases dense)")
        print(f"{'='*70}")
        print(f"  {'K':>5}  {'C':>10}  {'gzip':>8}  {'tcomp':>6}  {'wOPs':>6}  "
              f"{'wOPt':>6}  {'sync_r':>7}")

    rows = []
    for K in K_vals:
        scores, gzips, tcs, wops_s, wops_t, sync_rs = [], [], [], [], [], []
        for s in range(N_SEEDS):
            hist = kuramoto_run(K, G, STEPS, BURNIN, WINDOW, seed=s*11+5)
            m = compute_C(hist, BURNIN, WINDOW)
            scores.append(m['score'])
            gzips.append(m['gzip'])
            tcs.append(m['tcomp'])
            wops_s.append(m['w_OP_s'])
            wops_t.append(m['w_OP_t'])

            # Order parameter: r = |<exp(i*theta)>| — but we only have binary
            # Proxy: fraction of cells in majority phase (rough sync measure)
            post = hist[BURNIN:BURNIN+WINDOW]
            frac_1 = post.mean()
            sync_rs.append(abs(2 * frac_1 - 1))  # 0 = desync, 1 = full sync

        avg_C = np.mean(scores)
        rows.append(dict(K=float(K), C=float(avg_C),
                         gzip=float(np.mean(gzips)),
                         tcomp=float(np.mean(tcs)),
                         wOPs=float(np.mean(wops_s)),
                         wOPt=float(np.mean(wops_t)),
                         sync_r=float(np.mean(sync_rs))))
        if verbose:
            print(f"  K={K:>4.1f}  C={avg_C:>10.5f}  gz={np.mean(gzips):>7.4f}  "
                  f"tc={np.mean(tcs):>5.3f}  wOPs={np.mean(wops_s):>5.3f}  "
                  f"wOPt={np.mean(wops_t):>5.3f}  r={np.mean(sync_rs):>6.3f}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE 4: POTTS MODEL (q=3)
# ═══════════════════════════════════════════════════════════════════════════════
# q-state Potts model: E = -J * sum_{<ij>} delta(s_i, s_j)
# First-order transition at T_c = 1 / ln(1 + sqrt(3)) ≈ 0.9950 for q=3.
# Binary encoding: state 0 → 0, states 1,2 → 1 (majority/minority).
# PREDICTION: Type A (symmetry-breaking, both phases dense).

def potts_run(T, q=3, G=64, sweeps=20000, burnin=10000, window=200,
              snap_every=50, seed=42):
    """q-state Potts model with Metropolis dynamics on 2D toroidal grid.

    Uses checkerboard update for speed.
    Binary encoding: state 0 → 0, states 1..q-1 → 1.
    """
    rng = np.random.default_rng(seed)

    # Random IC: each cell in {0, 1, ..., q-1}
    spins = rng.integers(0, q, size=(G, G), dtype=np.int8)

    history = []
    snap_count = 0
    total = burnin + window * snap_every

    ii, jj = np.mgrid[0:G, 0:G]
    black = ((ii + jj) % 2 == 0)
    white = ~black

    for sweep in range(total):
        for mask in (black, white):
            # For each cell in sublattice, propose a new random state
            proposed = rng.integers(0, q, size=(G, G), dtype=np.int8)

            # Count matching neighbours for current and proposed states
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb = np.roll(np.roll(spins, di, axis=0), dj, axis=1)
                if di == 0 and dj == 1:
                    match_cur = (spins == nb).astype(np.float32)
                    match_new = (proposed == nb).astype(np.float32)
                else:
                    match_cur += (spins == nb).astype(np.float32)
                    match_new += (proposed == nb).astype(np.float32)

            dE = -(match_new - match_cur)  # energy change (J=1)

            # Accept if dE <= 0 or random < exp(-dE/T)
            accept = mask & ((dE <= 0) |
                             (rng.random((G, G)) < np.exp(np.clip(-dE / max(T, 1e-10), -30, 0))))
            spins = np.where(accept, proposed, spins)

        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                # Binary: state 0 → 0, states 1+ → 1
                binary = (spins > 0).astype(np.int8)
                history.append(binary.ravel())
                snap_count += 1

    return np.array(history, dtype=np.int8)


def potts_sweep(verbose=True):
    G, N_SEEDS = 64, 8
    SWEEPS, BURNIN, WINDOW, SNAP_EVERY = 20000, 10000, 200, 50
    T_C = 1.0 / np.log(1 + np.sqrt(3))  # ≈ 0.9950 for q=3
    T_vals = np.round(np.arange(0.5, 2.05, 0.05), 3)

    if verbose:
        print(f"\n{'='*70}")
        print(f"POTTS MODEL q=3 — temperature sweep")
        print(f"  Grid: {G}x{G}  Seeds: {N_SEEDS}  T_c = {T_C:.4f}")
        print(f"  PREDICTION: Type A (symmetry-breaking, first-order)")
        print(f"{'='*70}")
        print(f"  {'T':>5}  {'C':>10}  {'gzip':>8}  {'tcomp':>6}  {'wOPs':>6}  "
              f"{'wOPt':>6}  {'mag':>6}")

    rows = []
    for T in T_vals:
        scores, gzips, tcs, wops_s, wops_t, mags = [], [], [], [], [], []
        for s in range(N_SEEDS):
            hist = potts_run(T, q=3, G=G, sweeps=SWEEPS, burnin=BURNIN,
                             window=WINDOW, snap_every=SNAP_EVERY, seed=s*13+7)
            m = compute_C(hist, burnin=0, window=WINDOW)
            scores.append(m['score'])
            gzips.append(m['gzip'])
            tcs.append(m['tcomp'])
            wops_s.append(m['w_OP_s'])
            wops_t.append(m['w_OP_t'])
            # Magnetisation proxy: deviation from 1/3 density
            dens = hist.mean()
            mags.append(abs(dens - 2.0/3.0) / (1.0/3.0))  # 0=disordered, 1=all one state

        avg_C = np.mean(scores)
        marker = ' << Tc' if abs(T - T_C) < 0.03 else ''
        rows.append(dict(T=float(T), C=float(avg_C),
                         gzip=float(np.mean(gzips)),
                         tcomp=float(np.mean(tcs)),
                         wOPs=float(np.mean(wops_s)),
                         wOPt=float(np.mean(wops_t)),
                         mag=float(np.mean(mags))))
        if verbose:
            print(f"  T={T:>4.2f}  C={avg_C:>10.5f}  gz={np.mean(gzips):>7.4f}  "
                  f"tc={np.mean(tcs):>5.3f}  wOPs={np.mean(wops_s):>5.3f}  "
                  f"wOPt={np.mean(wops_t):>5.3f}  M={np.mean(mags):>5.3f}{marker}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_profile(params, scores, p_c=None):
    """Classify a C profile as Type A, B, or C based on shape metrics."""
    params = np.array(params, dtype=float)
    scores = np.array(scores, dtype=float)

    unique_p = sorted(set(params))
    avg_scores = np.array([np.mean(scores[params == p]) for p in unique_p])
    unique_p = np.array(unique_p)

    if avg_scores.max() == 0:
        return 'INDETERMINATE', {}

    peak_idx = np.argmax(avg_scores)
    p_peak   = unique_p[peak_idx]
    C_peak   = avg_scores[peak_idx]

    # Offset
    if p_c is not None and p_c != 0:
        offset_pct = (p_peak - p_c) / abs(p_c) * 100
    else:
        offset_pct = 0.0

    # Sub-critical death fraction
    if p_c is not None:
        sub = unique_p < p_c
        if sub.any():
            dead_frac = float(np.mean(avg_scores[sub] < 0.01 * C_peak))
        else:
            dead_frac = 0.0
    else:
        dead_frac = 0.0

    # Asymmetry
    if peak_idx > 0 and peak_idx < len(unique_p) - 1:
        left_area  = float(np.trapezoid(avg_scores[:peak_idx+1], unique_p[:peak_idx+1]))
        right_area = float(np.trapezoid(avg_scores[peak_idx:], unique_p[peak_idx:]))
        total = left_area + right_area
        asymmetry = (right_area - left_area) / total if total > 0 else 0
    elif peak_idx == 0:
        asymmetry = 1.0  # all weight to the right → monotonic decline
    else:
        asymmetry = -1.0

    metrics = dict(p_peak=round(p_peak, 4), C_peak=round(float(C_peak), 5),
                   offset_pct=round(offset_pct, 1),
                   dead_subcrit=round(dead_frac, 2),
                   asymmetry=round(asymmetry, 3))

    # Classification rules
    if peak_idx == 0 and asymmetry > 0.7:
        profile_type = 'C'  # monotonic decline from natural state
    elif dead_frac > 0.5 and offset_pct > 15:
        profile_type = 'B'  # dead sub-critical, offset peak
    elif abs(offset_pct) < 15:
        profile_type = 'A'  # symmetric peak near p_c
    elif dead_frac > 0.5:
        profile_type = 'B'
    else:
        profile_type = 'A'  # default

    return profile_type, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('substrate', nargs='?', default='all',
                   choices=['contact', 'voter', 'kuramoto', 'potts', 'all'])
    args = p.parse_args()

    results = {}

    if args.substrate in ('contact', 'all'):
        t0 = time.time()
        rows = contact_process_sweep()
        elapsed = time.time() - t0
        ptype, metrics = classify_profile(
            [r['lam'] for r in rows], [r['C'] for r in rows], p_c=1.6489)
        results['contact'] = (ptype, metrics)
        print(f"\n  >> CONTACT PROCESS: Classified as TYPE {ptype}")
        print(f"     Predicted: B | Metrics: {metrics}")
        print(f"     Time: {elapsed:.1f}s")

    if args.substrate in ('voter', 'all'):
        t0 = time.time()
        rows = voter_model_sweep()
        elapsed = time.time() - t0
        # Voter model: no sharp p_c, but transition is around mu ~ 0.01-0.05
        ptype, metrics = classify_profile(
            [r['mu'] for r in rows], [r['C'] for r in rows], p_c=0.01)
        results['voter'] = (ptype, metrics)
        print(f"\n  >> VOTER MODEL: Classified as TYPE {ptype}")
        print(f"     Predicted: B | Metrics: {metrics}")
        print(f"     Time: {elapsed:.1f}s")

    if args.substrate in ('kuramoto', 'all'):
        t0 = time.time()
        rows = kuramoto_sweep()
        elapsed = time.time() - t0
        # K_c for 2D lattice Kuramoto not analytically known, estimate ~2-4
        ptype, metrics = classify_profile(
            [r['K'] for r in rows], [r['C'] for r in rows], p_c=3.0)
        results['kuramoto'] = (ptype, metrics)
        print(f"\n  >> KURAMOTO: Classified as TYPE {ptype}")
        print(f"     Predicted: A | Metrics: {metrics}")
        print(f"     Time: {elapsed:.1f}s")

    if args.substrate in ('potts', 'all'):
        t0 = time.time()
        rows = potts_sweep()
        elapsed = time.time() - t0
        T_C = 1.0 / np.log(1 + np.sqrt(3))
        ptype, metrics = classify_profile(
            [r['T'] for r in rows], [r['C'] for r in rows], p_c=T_C)
        results['potts'] = (ptype, metrics)
        print(f"\n  >> POTTS q=3: Classified as TYPE {ptype}")
        print(f"     Predicted: A | Metrics: {metrics}")
        print(f"     Time: {elapsed:.1f}s")

    # Final scorecard
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"SCORECARD")
        print(f"{'='*70}")
        predictions = {'contact': 'B', 'voter': 'B', 'kuramoto': 'A', 'potts': 'A'}
        correct = 0
        for name, (ptype, metrics) in results.items():
            pred = predictions[name]
            match = 'MATCH' if ptype == pred else 'MISMATCH'
            if ptype == pred: correct += 1
            print(f"  {name:12s}  predicted={pred}  actual={ptype}  "
                  f"offset={metrics['offset_pct']:>+6.1f}%  "
                  f"dead={metrics['dead_subcrit']:.2f}  {match}")
        print(f"\n  Score: {correct}/{len(results)}")
