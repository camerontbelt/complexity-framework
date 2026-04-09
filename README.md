# Complexity Framework

A framework for measuring complexity derived from eight candidate necessary
properties of genuinely complex systems. Validated empirically across four
simulation substrates.

**Paper:** _Toward a Measurable Theory of Complexity: Eight Candidate Laws,
Metric Derivation, and Empirical Validation Across Four Simulation Substrates_
— Cameron Belt (preprint, see `complexity_paper_v2.tex`)

---

## Key Results

| Substrate           | Rules/points | C4/C3 separation    | C4 rank result        |
| ------------------- | ------------ | ------------------- | --------------------- |
| 1D ECA (r=1)        | 256          | **45×**             | Ranks 1–4             |
| 1D totalistic (r=2) | 32           | 21×                 | Ranks 1–2             |
| 2D Life-like        | 17           | — (calibration gap) | All in top 10         |
| N-body scan         | 256          | —                   | Our universe 54th pct |

All four known Class 4 ECA rules (110, 124, 137, 193) rank 1–4 of 256 under
random IC with IC-independent weight functions. Scale invariance confirmed
W=100–300.

---

## Requirements

```bash
pip install numpy matplotlib
```

`scipy` is optional (only for N-body live visualisation).  
Python 3.9+ recommended. All experiments run on CPU; no GPU required.

---

## Repository Structure

```
complexity_framework.py   # Unified code for all four substrates
complexity_paper_v2.tex   # Full paper (LaTeX source)
data/
  eca_random_results.csv          # ECA 256 rules, random IC
  eca_single_results.csv          # ECA 256 rules, single-cell IC
  eca_scale_sweep.csv             # Scale invariance sweep W=100–300
  k3_results.csv                  # Radius-2 totalistic CA, 32 rules
  life_results.csv                # 2D Life-like, 17 rules
  nbody_scan.csv                  # N-body 16×16 parameter scan
  nbody_scan.json                 # Same scan with heatmap metadata
```

---

## Reproducing All Experiments

The single file `complexity_framework.py` runs everything.
Each command below reproduces one experiment from the paper.
Expected runtimes are on a modern laptop CPU.

### Experiment 1 — ECA, all 256 rules, random IC (~2 min)

This is the main result: 45× C4/C3 separation, all Class 4 rules at ranks 1–4.

```bash
python complexity_framework.py eca --csv data/eca_random_results.csv
```

**What you should see:**

```
C4/C3 separation : 45.0x   C4 ranks : [1, 2, 3, 4]
Rank   1  Rule 124  C4   1.215  ...
Rank   2  Rule 137  C4   1.210  ...
Rank   3  Rule 193  C4   1.157  ...
Rank   4  Rule 110  C4   1.129  ...
```

To save the ranking plot:

```bash
python complexity_framework.py eca --csv data/eca_random_results.csv \
  --save-plot eca_rankings.png
```

---

### Experiment 1b — ECA, single-cell IC (~2 min)

Shows IC-contamination of the opacity metric under single-cell IC.
C4 rules drop to ranks 133–139 because opacity in the cone-transient
window does not reflect asymptotic rule complexity.

```bash
python complexity_framework.py eca --ic single --csv data/eca_single_results.csv
```

---

### Experiment 1c — Scale invariance sweep (~8 min)

Confirms C4 rules stay at ranks 1–4 across grid widths 100–300.

```bash
python complexity_framework.py eca --scale-test \
  --widths 100 150 200 300 \
  --scale-csv data/eca_scale_sweep.csv
```

**Expected output:**

```
W=100: sep=33.5x  C4=[1,2,3,4]  R110=#4  tc=0.580
W=150: sep=44.9x  C4=[1,2,3,4]  R110=#4  tc=0.580
W=200: sep=34.2x  C4=[1,2,3,4]  R110=#1  tc=0.578
W=300: sep=28.3x  C4=[1,2,3,4]  R110=#2  tc=0.581
```

---

### Experiment 1d — Diagnose a specific rule (~10 sec)

Shows per-seed metric breakdown for Rule 110 (computationally universal):

```bash
python complexity_framework.py eca --diagnose 110
```

Compare Rule 30 (Class 3 chaotic) to see the entropy variance difference:

```bash
python complexity_framework.py eca --diagnose 30
```

Key finding: Rule 110 `std_H ≈ 0.013`, Rule 30 `std_H ≈ 0.006`.

---

### Experiment 2 — Radius-2 totalistic CA, 32 rules (~30 sec)

```bash
python complexity_framework.py k3 --csv data/k3_results.csv
```

C4-analog rules (10, 18) rank 1st and 2nd; 21× separation.
Note: rules 22 and 26 (sometimes listed as complex in totalistic literature)
annihilate under random IC and rank near the bottom.

---

### Experiment 3 — 2D Life-like CA, 17 rules (~2 min)

```bash
python complexity_framework.py life --csv data/life_results.csv
```

All four Class 4 rules appear in the top 10.
Conway and HighLife rank 7th and 6th respectively.
Gnarl (Class 3) ranks 1st due to the temporal compression calibration gap
discussed in Section 6 of the paper.

To see all available Life-like rules and their B/S notation:

```bash
python complexity_framework.py life --list-rules
```

---

### Experiment 4 — N-body parameter scan (~2–5 min depending on seeds)

```bash
python complexity_framework.py nbody --scan \
  --csv data/nbody_scan.csv
```

This runs a 16×16 grid of (α, αs) combinations with 3 seeds each.
Our universe (α=αs=1.0) scores at approximately the 54th percentile.
The scan also saves `nbody_scan.json` for the heatmap visualisation.

To visualise the heatmap from a saved scan:

```bash
python complexity_framework.py nbody --heatmap nbody_scan.json
```

To evaluate a single point (e.g. our universe):

```bash
python complexity_framework.py nbody --alpha 1.0 --alpha-s 1.0
```

---

### Run All Substrates at Once

```bash
python complexity_framework.py all --csv data/all_results.csv
```

This runs ECA (256 rules), k=3 CA (32 rules), Life-like (17 rules), and
N-body single-point in sequence. Approximately 5 minutes total.

---

## Understanding the Metrics

All experiments use **random 50% IC** as the universal standard.

| Metric    | What it measures            | Candidate law | IC status                |
| --------- | --------------------------- | ------------- | ------------------------ |
| `mean_H`  | Mean spatial entropy        | P7            | —                        |
| `std_H`   | Entropy variance (new)      | P7            | Critical under random IC |
| `opacity` | Local→global info hiding    | P1            | Random IC only           |
| `tcomp`   | Temporal persistence        | P4/P5         | IC-independent           |
| `gzip`    | Kolmogorov complexity proxy | P2            | IC-independent           |

### Weight Functions

```python
# Entropy — attractor weight (IC-independent)
w_H = tanh(50 * mean_H) * tanh(50 * (1 - mean_H))   # gate term
    * (1 + exp(-((std_H - 0.012) / 0.008)**2))        # variance bonus

# Opacity — Gaussian (random IC only)
w_OP = exp(-((opacity - 0.14) / 0.10)**2)

# Temporal compression — dual-Gaussian (IC-independent)
w_T = max(exp(-((tcomp - 0.58) / 0.08)**2),          # random IC attractor
          exp(-((tcomp - 0.73) / 0.08)**2))           # single-cell IC attractor

# Gzip — Gaussian (IC-independent)
w_G = exp(-((gzip - 0.10) / 0.05)**2)

# Composite (multiplicative — non-eliminability enforced)
C = w_H * w_OP * w_T * w_G
```

### Why Multiplicative?

The multiplicative structure means a system must score well on **all four**
dimensions simultaneously. This operationalises the theoretical claim that
genuine complexity resists reduction to any single observable.

- Class 1 (uniform): eliminated by `w_OP` and `w_G`
- Class 2 (periodic): eliminated by `w_T` (extreme tcomp)
- Class 3 (chaotic): eliminated by `w_OP` and variance component of `w_H`
- **Class 4 (complex): survives all four filters**

---

## CLI Reference

```
usage: complexity_framework.py [substrate] [options]

Substrates:
  eca     1D binary CA, r=1, 256 Wolfram rules (default)
  k3      1D binary CA, r=2 totalistic, 32 rules
  life    2D Life-like outer totalistic CA
  nbody   N-body particle simulation
  all     Run all substrates in sequence

Common options:
  --csv FILE          Save results to CSV
  --no-plot           Skip matplotlib output
  --save-plot FILE    Save plot to file
  --seeds N           Number of random seeds (default: 5)

ECA / k=3 options:
  --ic {random,single}    IC type (default: random)
  --density FLOAT         Random IC density (default: 0.5)
  --width N               Grid width (default ECA: 150, k3: 200)
  --diagnose RULE         Verbose per-seed breakdown for one rule
  --scale-test            Sweep grid widths
  --widths N [N ...]      Custom widths for scale test
  --scale-csv FILE        Save scale test results to CSV

2D Life options:
  --grid N            Grid size NxN (default: 64)
  --list-rules        Print all available Life-like rules

N-body options:
  --scan              Run full parameter scan
  --heatmap FILE      Visualise saved scan JSON
  --alpha FLOAT       EM coupling (default: 1.0)
  --alpha-s FLOAT     Strong force (default: 1.0)
```

---

## Open Questions

The following are documented in Section 8 of the paper:

1. **2D temporal compression calibration** — Conway/HighLife have `tcomp ≈ 0.87`,
   above both dual-Gaussian peaks. A third peak at ≈0.87 for 2D substrates
   is needed and requires validation against more 2D rules.

2. **Theoretical derivation of boundary values** — The entropy variance peak
   (`std_H = 0.012`), opacity peak (0.14), and gzip peak (0.10) were all
   discovered empirically. A derivation from phase-transition theory would
   transform them from calibration parameters into predictions.

3. **Ternary and higher-state CA** — The metrics generalise formally but
   the attractor values are unknown. A natural next experiment.

4. **Gray-Scott and continuous systems** — The entity-level pipeline
   (segment concentration field → track objects → compute entity metrics)
   is the correct approach for object-based continuous complexity.

5. **N-body force law fidelity** — The Lennard-Jones toy model cannot
   reproduce full Standard Model fine-tuning constraints. The peak near
   `(α≈0.13, αs≈2.0)` warrants expert evaluation.

---

## Citation

```
Belt, C. (2025). Toward a Measurable Theory of Complexity: Eight Candidate
Laws, Metric Derivation, and Empirical Validation Across Four Simulation
Substrates. Preprint.
```

---

## Acknowledgements

Developed in extended conversation with Claude (Anthropic), which contributed
computational implementation, metric design, and experimental execution.
Intellectual framework, design decisions, and interpretations are those of
the human author.
