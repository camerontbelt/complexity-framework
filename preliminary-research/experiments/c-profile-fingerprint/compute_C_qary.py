"""
q-ary C: a compact generalisation of the binary compute_C pipeline.

Accepts a (T, W) int grid whose cells take values in {0, 1, ..., q-1} and
returns a score with the same structure as the binary composite, but with
every sub-metric generalised to q states.

Components (all normalised to [0,1]):

  w_H    : Shannon entropy of the global state distribution / log(q)
           Peaks at 1 when all q states are equally populated.

  w_OP_s : Spatial mutual information I(X ; X_neighbour) / H(X)
           q x q joint count matrix over horizontal-neighbour pairs.

  w_OP_t : Temporal mutual information I(X_t ; X_{t+1}) / H(X_t)
           q x q transition count matrix.

  w_T    : Fraction of cells that change state between consecutive frames
           Peaks at 0.5 (neither frozen nor noisy).

  w_G    : Fraction of unlike-neighbour edges (spatial boundary density)
           Peaks at (q-1)/q; we rescale so max = 1.

Composite:   C = w_H * (w_OP_s + w_OP_t) * w_T * w_G / 2
             (mirrors the binary pipeline's geometric averaging)
"""
import numpy as np


def _hist(grid, q):
    counts = np.bincount(grid.ravel(), minlength=q).astype(np.float64)
    p = counts / counts.sum()
    return p


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _joint_counts(a, b, q):
    # a, b: flat int arrays of same length, values 0..q-1
    idx = a * q + b
    return np.bincount(idx, minlength=q*q).reshape(q, q).astype(np.float64)


def _mi_normalised(M):
    # M: q x q count matrix -> MI / H(row), all in nats
    total = M.sum()
    if total == 0: return 0.0
    P = M / total
    pr = P.sum(axis=1, keepdims=True)
    pc = P.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        logterm = np.where(P > 0, np.log(P / (pr @ pc + 1e-30)), 0.0)
    MI = float((P * logterm).sum())
    Hr = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum())
    if Hr < 1e-9: return 0.0
    return float(np.clip(MI / Hr, 0.0, 1.0))


def compute_C_qary(grid, q, burnin, window):
    """grid: (T, W) int array, values 0..q-1.
    Returns dict with raw metrics, weights, and composite score."""
    sub = grid[burnin:burnin+window]
    T, W = sub.shape
    if T < 2 or W < 2:
        return dict(score=0.0, w_H=0.0, w_OP_s=0.0, w_OP_t=0.0, w_T=0.0, w_G=0.0)

    # w_H: entropy normalised by log(q)
    p = _hist(sub, q)
    H = _entropy(p)
    w_H = float(np.clip(H / np.log(q), 0.0, 1.0))

    # w_OP_s: spatial MI via horizontal neighbour pairs, pooled over frames
    a = sub[:, :-1].ravel()
    b = sub[:,  1:].ravel()
    M_s = _joint_counts(a, b, q)
    w_OP_s = _mi_normalised(M_s)

    # w_OP_t: temporal MI via frame-to-next-frame pairs
    a = sub[:-1].ravel()
    b = sub[1: ].ravel()
    M_t = _joint_counts(a, b, q)
    w_OP_t = _mi_normalised(M_t)

    # w_T: temporal change fraction, peaked at 0.5
    changes = (sub[1:] != sub[:-1]).mean()
    w_T = float(1.0 - 2.0 * abs(changes - 0.5))

    # w_G: fraction of unlike-neighbour edges, rescaled so maxunif = 1
    unlike = (sub[:, :-1] != sub[:,  1:]).mean()
    max_unlike = (q - 1) / q
    w_G = float(np.clip(unlike / max_unlike, 0.0, 1.0)) if max_unlike > 0 else 0.0
    # peaked weighting so a perfectly random-coloured field doesn't trivially max
    w_G = float(1.0 - 2.0 * abs(w_G - 0.5))  # peak at half-maximum disorder

    w_geom = 0.5 * (w_OP_s + w_OP_t)
    score = float(w_H * w_geom * w_T * w_G)
    return dict(score=score, w_H=w_H, w_OP_s=w_OP_s, w_OP_t=w_OP_t,
                w_T=w_T, w_G=w_G, H_nats=H)


def coarsen_history_qary(hist_3d, pool, q):
    """Spatial pooling for q-ary field: take the mode (most common state)
    within each pool x pool block. Shape (T, G, G) -> (T, G/pool, G/pool)."""
    T, G, _ = hist_3d.shape
    G2 = G // pool
    out = np.zeros((T, G2, G2), dtype=hist_3d.dtype)
    for i in range(G2):
        for j in range(G2):
            block = hist_3d[:, i*pool:(i+1)*pool, j*pool:(j+1)*pool]
            block_flat = block.reshape(T, -1)
            # Mode per-frame via bincount loop (small blocks)
            for t in range(T):
                counts = np.bincount(block_flat[t], minlength=q)
                out[t, i, j] = counts.argmax()
    return out


if __name__ == '__main__':
    # quick sanity: uniform random q-ary field should give mid-range C
    rng = np.random.default_rng(0)
    g = rng.integers(0, 5, size=(100, 64*64), dtype=np.int32)
    r = compute_C_qary(g, q=5, burnin=10, window=80)
    print("uniform random q=5 field:", {k: round(v, 3) if isinstance(v, float) else v
                                         for k, v in r.items()})
    # fully ordered: all same state
    g2 = np.zeros_like(g)
    r2 = compute_C_qary(g2, q=5, burnin=10, window=80)
    print("frozen q=5 field:        ", {k: round(v, 3) if isinstance(v, float) else v
                                         for k, v in r2.items()})
