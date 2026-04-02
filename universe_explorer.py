"""
Universe Complexity Explorer v2
================================
N-body particle simulation with tunable physical constants.
Validates complexity metrics against physical sterility criteria,
compares to analytically predicted island of stability, and
exports full data to CSV for further analysis.

Derived from eight candidate laws of complexity. Key insight from
cellular automaton experiments: complexity lives at the edge of chaos —
intermediate spatial entropy, intermediate temporal compression, intermediate
gzip ratio. Sterility (collapse or dispersal) appears at entropy extremes.

USAGE:
    python universe_explorer.py              # interactive menu
    python universe_explorer.py calibrate    # Phase 0: axis sweeps
    python universe_explorer.py scan         # Phase 1: full parameter scan
    python universe_explorer.py plot         # Phase 2: heatmaps + correlation
    python universe_explorer.py simulate [alpha] [alpha_s]  # live view

REQUIREMENTS:
    pip install numpy scipy matplotlib tqdm

PARAMETERS (our universe = normalized to 1.0):
    alpha      = EM coupling strength        (fine structure constant analog)
    alpha_s    = strong force / repulsion     (equilibrium separation)
    mass_ratio = heavy/light particle ratio   (proton/electron mass analog)
    temperature= initial thermal energy       (early universe temperature analog)

PREDICTED ISLAND BOUNDARIES (derived analytically from force law):
    alpha      in [0.2,  4.0]   — binding energy 0.3–3× thermal
    alpha_s    in [0.4,  2.5]   — equilibrium sep 1.5–4× mean spacing
    mass_ratio in [4.0, 80.0]   — qualitatively distinct species dynamics
    temperature in [0.05, 0.8]  — liquid-like regime (kT/well depth)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from scipy import ndimage
import json, csv, os, sys, time, zlib

# ============================================================
# CONFIGURATION
# ============================================================

CFG = {
    # Simulation
    'N': 80,
    'BOX': 20.0,
    'DT': 0.025,
    'DAMPING': 0.998,
    'FORCE_CLAMP': 30.0,
    'N_HEAVY': 20,       # heavy particles (nucleus analog)
    'N_LIGHT': 60,       # light particles (electron analog)

    # Default constants (our universe)
    'ALPHA': 1.0,
    'ALPHA_S': 1.0,
    'MASS_RATIO': 10.0,
    'TEMPERATURE': 0.35,

    # Scan settings
    'SCAN_STEPS': 350,
    'SCAN_SKIP': 70,
    'SCAN_SAMPLE': 5,
    'SCAN_SEEDS': [42, 123, 7],
    'SCAN_FILE': 'universe_scan.csv',
    'SCAN_JSON': 'universe_scan.json',

    # Calibration (axis sweep)
    'CAL_STEPS': 250,
    'CAL_SKIP': 50,
    'CAL_N': 15,

    # Sterility thresholds (from CA edge-of-chaos insight)
    'ENTROPY_COLLAPSE_MAX': 0.12,   # below this = collapsed/frozen
    'ENTROPY_DISPERSAL_MAX': 0.18,  # dispersed universes also low entropy
    'LARGEST_CLUSTER_COLLAPSE': 0.65,  # >65% in one cluster = collapsed
    'MAX_CLUSTER_DISPERSAL': 2,        # max cluster size ≤2 = dispersed

    # Predicted island bounds (analytically derived)
    'ISLAND': {
        'alpha':       (0.20, 4.00),
        'alpha_s':     (0.40, 2.50),
        'mass_ratio':  (4.00, 80.0),
        'temperature': (0.05, 0.80),
    },
}

# ============================================================
# PHYSICS — two-species particles
# ============================================================

def init_particles(cfg, seed=42):
    """
    Two particle species:
      Heavy (indices 0..N_HEAVY-1): mass = mass_ratio, slow
      Light (indices N_HEAVY..N-1): mass = 1.0, fast
    """
    np.random.seed(seed)
    N = cfg['N']; box = cfg['BOX']
    mass_ratio = cfg.get('MASS_RATIO', 10.0)
    temp = cfg.get('TEMPERATURE', 0.35)
    n_heavy = cfg['N_HEAVY']; n_light = cfg['N_LIGHT']

    pos = np.random.rand(N, 2) * box
    masses = np.ones(N)
    masses[:n_heavy] = mass_ratio

    # Maxwell-Boltzmann: v ~ sqrt(kT/m)
    vel = np.zeros((N, 2))
    for i in range(N):
        vel[i] = np.random.randn(2) * np.sqrt(temp / masses[i])

    return pos, vel, masses

def compute_forces(pos, masses, alpha, alpha_s, cfg):
    box = cfg['BOX']; clamp = cfg['FORCE_CLAMP']
    dx = pos[:,None,0] - pos[None,:,0]
    dy = pos[:,None,1] - pos[None,:,1]
    dx -= box*np.round(dx/box); dy -= box*np.round(dy/box)
    r2 = np.maximum(dx**2+dy**2, 0.04)
    cut = (alpha_s*3.5)**2
    mask = r2<cut; np.fill_diagonal(mask,False)
    # Scale alpha by geometric mean of masses (heavier = stronger binding)
    m_scale = np.sqrt(masses[:,None]*masses[None,:])
    sr2 = np.where(mask,(alpha_s**2)/r2,0)
    sr6 = sr2**3; sr12 = sr6**2
    fm = np.where(mask, 24*alpha*m_scale*(2*sr12-sr6)/r2, 0)
    np.fill_diagonal(fm,0)
    fx = np.clip((fm*dx).sum(1),-clamp,clamp)
    fy = np.clip((fm*dy).sum(1),-clamp,clamp)
    return fx, fy

def step_sim(pos, vel, masses, alpha, alpha_s, cfg):
    dt = cfg['DT']; damp = cfg['DAMPING']; box = cfg['BOX']
    fx,fy = compute_forces(pos,masses,alpha,alpha_s,cfg)
    # F = ma → a = F/m
    vel[:,0] += 0.5*(fx/masses)*dt
    vel[:,1] += 0.5*(fy/masses)*dt
    pos = (pos+vel*dt)%box
    fx2,fy2 = compute_forces(pos,masses,alpha,alpha_s,cfg)
    vel[:,0] += 0.5*(fx2/masses)*dt
    vel[:,1] += 0.5*(fy2/masses)*dt
    vel *= damp
    return pos, vel

def run_sim(alpha, alpha_s, mass_ratio, temperature, cfg, seed=42, steps=None, skip=None, sample=None):
    c = cfg.copy()
    c['ALPHA']=alpha; c['ALPHA_S']=alpha_s
    c['MASS_RATIO']=mass_ratio; c['TEMPERATURE']=temperature
    if steps  is not None: c['SCAN_STEPS']  = steps
    if skip   is not None: c['SCAN_SKIP']   = skip
    if sample is not None: c['SCAN_SAMPLE'] = sample
    pos, vel, masses = init_particles(c, seed=seed)
    frames = []
    for s in range(c['SCAN_STEPS']):
        pos, vel = step_sim(pos,vel,masses,alpha,alpha_s,c)
        if s >= c['SCAN_SKIP'] and s % c['SCAN_SAMPLE'] == 0:
            frames.append((pos.copy(), vel.copy()))
    return frames, masses

# ============================================================
# ENTITY DETECTION
# ============================================================

def find_clusters(pos, alpha_s, box, factor=2.0):
    bd = max(alpha_s*factor, 0.5)
    N = len(pos)
    dx = pos[:,None,0]-pos[None,:,0]; dy = pos[:,None,1]-pos[None,:,1]
    dx -= box*np.round(dx/box); dy -= box*np.round(dy/box)
    r = np.sqrt(dx**2+dy**2)
    parent = list(range(N))
    def find(x):
        root=x
        while parent[root]!=root: root=parent[root]
        while parent[x]!=root: nxt=parent[x];parent[x]=root;x=nxt
        return root
    rows,cols=np.where((r<bd)&(r>0.001))
    for a,b in zip(rows,cols):
        pa,pb=find(a),find(b)
        if pa!=pb: parent[pa]=pb
    g=defaultdict(list)
    for i in range(N): g[find(i)].append(i)
    return [v for v in g.values() if len(v)>=2]

# ============================================================
# PHYSICAL STERILITY CRITERION
# (independent of complexity metrics — pure physics)
# ============================================================

def classify_universe(frames, alpha_s, cfg):
    """
    Classify a universe as Collapsed, Dispersed, or Active.
    Uses ONLY particle positions and cluster structure — no complexity metrics.
    This is our ground truth for the island of stability.

    From CA insight: sterile universes have entropy near zero.
    Active (complex) universes have intermediate entropy.
    """
    if not frames:
        return 'Dispersed', 0.0, 0.0

    N = cfg['N']; box = cfg['BOX']
    # Use last third of frames for settled state
    settled = frames[len(frames)//3:]

    # Measure spatial entropy of density field over time
    sp_entropies = []
    for pos,_ in settled:
        h,_,_ = np.histogram2d(pos[:,0],pos[:,1],bins=10,
                                range=[[0,box],[0,box]])
        h = h/h.sum()
        sp_entropies.append(float(-np.sum(h[h>0]*np.log2(h[h>0]))/np.log2(100)))
    mean_entropy = float(np.mean(sp_entropies))

    # Cluster structure in last frame
    pos_last = frames[-1][0]
    clusters = find_clusters(pos_last, alpha_s, box)
    if clusters:
        largest_frac = max(len(c) for c in clusters) / N
    else:
        largest_frac = 0.0
    n_clusters = len(clusters)

    # Classification rules
    if largest_frac >= cfg['LARGEST_CLUSTER_COLLAPSE']:
        state = 'Collapsed'
    elif n_clusters == 0 or max((len(c) for c in clusters),default=0) <= cfg['MAX_CLUSTER_DISPERSAL']:
        state = 'Dispersed'
    elif mean_entropy < cfg['ENTROPY_COLLAPSE_MAX']:
        state = 'Collapsed'
    else:
        state = 'Active'

    return state, mean_entropy, largest_frac

# ============================================================
# COMPLEXITY METRICS (CA-derived edge-of-chaos framework)
# ============================================================

def compute_complexity(frames, alpha_s, cfg):
    """
    All metrics derived from CA experiment findings.
    Sterility = entropy extremes. Complexity = intermediate values.
    """
    if len(frames) < 4:
        return {k:0.0 for k in ['complexity','entity_c','field_c',
                'sp_entropy','t_comp','gzip_ratio','eoc_weight',
                'count','stability','interact','size_div','div_rate']}
    box = cfg['BOX']; N = cfg['N']
    # ---- Spatial entropy (P7 thermodynamic drive) ----
    sp_vals = []
    for pos,_ in frames:
        h,_,_ = np.histogram2d(pos[:,0],pos[:,1],bins=10,
                                range=[[0,box],[0,box]])
        h=h/h.sum()
        sp_vals.append(float(-np.sum(h[h>0]*np.log2(h[h>0]))/np.log2(100)))
    sp_entropy = float(np.mean(sp_vals))

    # ---- Edge-of-chaos Gaussian weight (from CA calibration) ----
    # In CA: complexity peaked at entropy ~0.85 with sigma ~0.08
    # For N-body: intermediate entropy is the signal, peak TBD by calibration
    # Use broader sigma since this is a different substrate
    def gauss(x,mu,sig): return float(np.exp(-0.5*((x-mu)/sig)**2))
    eoc_weight = gauss(sp_entropy, 0.55, 0.20)  # broader, calibration will refine

    # ---- Temporal compression (P4/P5 persistence) ----
    arr = np.array([pos for pos,_ in frames])
    # Quantize position to grid
    grid_pos = np.clip((arr/box*8).astype(int),0,7)
    flat = (grid_pos[:,:,0]*8 + grid_pos[:,:,1])  # shape (T, N)
    T = flat.shape[0]
    idxs = np.random.choice(N, min(60,N), replace=False)
    tc_vals = []
    for i in idxs:
        seq = flat[:,i]
        runs = 1+int(np.sum(seq[1:]!=seq[:-1]))
        tc_vals.append(1.0 - runs/T)
    t_comp = float(np.mean(tc_vals))

    # ---- Gzip ratio (P2 global structure / Kolmogorov) ----
    q = np.clip((arr/box*16).astype(np.uint8),0,15)
    raw = q.tobytes()
    gz = len(zlib.compress(raw,6))/len(raw)
    gzip_ratio = gz

    # ---- Entity metrics ----
    frame_clusters = [find_clusters(pos,alpha_s,box) for pos,_ in frames]
    counts = np.array([len(c) for c in frame_clusters],dtype=float)
    mean_count = float(counts.mean())
    if mean_count < 0.3:
        stability=0.0; interact=0.0; size_div=0.0; div_rate=0.0
    else:
        stability = float(1.0/(1.0+counts.std()/(mean_count+1e-8)))
        # Interaction density
        irates=[]
        for cl in frame_clusters:
            if len(cl)<2: irates.append(0.0); continue
            cents=np.array([np.mean([frames[0][0][i] for i in c],axis=0) for c in cl])
            dx=cents[:,None,0]-cents[None,:,0]; dy=cents[:,None,1]-cents[None,:,1]
            dx-=box*np.round(dx/box); dy-=box*np.round(dy/box)
            r=np.sqrt(dx**2+dy**2)
            pairs=(r<box*0.35).sum()-len(cl)
            irates.append(float(pairs)/len(cl))
        interact=float(np.mean(irates))
        # Size diversity
        sizes=[len(c) for cs in frame_clusters for c in cs]
        if sizes:
            sa=np.array(sizes); bins=min(12,max(2,len(np.unique(sa))))
            if bins>1:
                h,_=np.histogram(sa,bins=bins); h=h/h.sum()
                size_div=float(-np.sum(h[h>0]*np.log2(h[h>0]))/np.log2(bins))
            else: size_div=0.0
        else: size_div=0.0
        div_rate=float(np.sum(np.abs(np.diff(counts))>=1)/max(T-1,1))

    # ---- Composite score ----
    entity_c = float(mean_count*stability*(1+interact)*(1+div_rate*3))
    field_c  = float(eoc_weight*(1+t_comp)*(1+size_div))
    complexity = float(np.sqrt(max(entity_c,0)*max(field_c,0)+1e-12))

    return {
        'complexity': complexity,
        'entity_c':   entity_c,
        'field_c':    field_c,
        'sp_entropy': sp_entropy,
        't_comp':     t_comp,
        'gzip_ratio': gzip_ratio,
        'eoc_weight': eoc_weight,
        'count':      mean_count,
        'stability':  stability,
        'interact':   interact,
        'size_div':   size_div,
        'div_rate':   div_rate,
    }

# ============================================================
# PREDICTED ISLAND (analytical)
# ============================================================

def in_predicted_island(alpha, alpha_s, mass_ratio, temperature, cfg):
    """Returns True if parameters fall within analytically predicted island."""
    isl = cfg['ISLAND']
    return (isl['alpha'][0]      <= alpha      <= isl['alpha'][1]      and
            isl['alpha_s'][0]    <= alpha_s    <= isl['alpha_s'][1]    and
            isl['mass_ratio'][0] <= mass_ratio <= isl['mass_ratio'][1] and
            isl['temperature'][0]<= temperature<= isl['temperature'][1])

def island_boundary_alpha(cfg):
    """Analytical α boundary: binding energy 0.3–3.0× thermal energy."""
    temp = cfg['TEMPERATURE']
    # Binding energy ∝ alpha, thermal ∝ temperature
    # Island: 0.3*temp <= alpha <= 3.0*temp (simplified)
    return (0.3*temp*3, 3.0*temp*3)  # scaled to our parameter range

def island_boundary_alphas(cfg):
    """Analytical αs boundary: equilibrium separation in liquid-like regime."""
    box=cfg['BOX']; N=cfg['N']
    mean_spacing = (box**2/N)**0.5
    return (mean_spacing*0.15, mean_spacing*0.50)  # 1.5–4× spacing scaled

# ============================================================
# PHASE 0: CALIBRATION SWEEPS
# ============================================================

def run_calibration(cfg):
    print("\n=== Phase 0: Calibration Sweeps ===")
    print("Sweeping each parameter axis, holding others at our universe values.")
    print("This shows where each boundary falls empirically.\n")

    N_CAL = cfg['CAL_N']
    results = {}

    params = {
        'alpha':       (np.logspace(-1, 0.7, N_CAL),  1.0,  1.0,  0.35),
        'alpha_s':     (np.logspace(-0.5,0.5, N_CAL), 1.0,  10.0, 0.35),
        'mass_ratio':  (np.logspace(0, 2, N_CAL),     1.0,  1.0,  0.35),
        'temperature': (np.linspace(0.02,1.0, N_CAL), 1.0,  1.0,  10.0),
    }

    fig, axes = plt.subplots(2,2,figsize=(12,8),facecolor='#0e0e16')
    fig.suptitle('Phase 0: Calibration Sweeps — Each Parameter Axis',
                 color='#ccccdd',fontsize=12)

    for ax,(name,(values,fix_as,fix_mr,fix_t)) in zip(axes.flat,params.items()):
        ax.set_facecolor('#0e0e16')
        for sp in ax.spines.values(): sp.set_color('#333344')
        ax.tick_params(colors='#888899',labelsize=8)

        complexities=[]; states=[]; entropies=[]
        for v in values:
            a   = v   if name=='alpha'       else 1.0
            aps = v   if name=='alpha_s'     else fix_as
            mr  = v   if name=='mass_ratio'  else fix_mr
            tmp = v   if name=='temperature' else fix_t

            seed_c=[]; seed_s=[]; seed_e=[]
            for seed in cfg['SCAN_SEEDS']:
                frames,_ = run_sim(a,aps,mr,tmp,cfg,seed=seed,
                                   steps=cfg['CAL_STEPS'],
                                   skip=cfg['CAL_SKIP'],sample=5)
                m = compute_complexity(frames,aps,cfg)
                state,ent,_ = classify_universe(frames,aps,cfg)
                seed_c.append(m['complexity'])
                seed_s.append(state)
                seed_e.append(ent)
            complexities.append(float(np.mean(seed_c)))
            entropies.append(float(np.mean(seed_e)))
            # majority vote on state
            from collections import Counter
            states.append(Counter(seed_s).most_common(1)[0][0])

        # Plot complexity line
        ax.plot(values,complexities,color='#1D9E75',lw=2,label='complexity')
        ax.set_ylabel('complexity',color='#888899',fontsize=8)

        # Color background by physical state
        state_colors = {'Active':'#1D9E7520','Collapsed':'#D85A3020','Dispersed':'#7F77DD20'}
        prev_state=states[0]; prev_x=values[0]
        for i,(v,s) in enumerate(zip(values[1:],states[1:]),1):
            if s!=prev_state or i==len(values)-1:
                ax.axvspan(prev_x,v,color=state_colors.get(prev_state,'#33333320'),
                           alpha=0.5,zorder=0)
                prev_state=s; prev_x=v

        # Mark our universe value
        our_val = {'alpha':1.0,'alpha_s':1.0,'mass_ratio':10.0,'temperature':0.35}[name]
        ax.axvline(our_val,color='#EF9F27',lw=1.5,linestyle='--',label='our universe')

        # Mark predicted island bounds
        isl = cfg['ISLAND'][name.replace('temperature','temperature')]
        if name in cfg['ISLAND']:
            ax.axvline(cfg['ISLAND'][name][0],color='#7F77DD',lw=1,
                       linestyle=':',label='predicted island')
            ax.axvline(cfg['ISLAND'][name][1],color='#7F77DD',lw=1,linestyle=':')

        if name in ('alpha','alpha_s'): ax.set_xscale('log')
        ax.set_xlabel(name,color='#888899',fontsize=9)
        ax.set_title(f'Sweep: {name}',color='#aaaacc',fontsize=9)
        ax.legend(fontsize=7,facecolor='#1a1a2a',labelcolor='#ccccdd',
                  edgecolor='#333344',loc='best')

        # Legend for state colors
        handles=[mpatches.Patch(color=c.replace('20',''),alpha=0.5,label=s)
                 for s,c in state_colors.items()]
        ax.legend(handles=handles+ax.get_lines(),fontsize=6,
                  facecolor='#1a1a2a',labelcolor='#ccccdd',
                  edgecolor='#333344',ncol=2,loc='upper right')

        results[name]={'values':values.tolist(),'complexity':complexities,
                       'states':states,'entropy':entropies}

    plt.tight_layout()
    plt.savefig('calibration_sweep.png',dpi=120,facecolor='#0e0e16')
    print("  Saved calibration_sweep.png")
    plt.show()
    return results

# ============================================================
# PHASE 1: FULL PARAMETER SCAN
# ============================================================

def run_scan(cfg, n_per_axis=12):
    """
    2D scan: alpha × alpha_s (holding mass_ratio and temperature at our values)
    Outputs full CSV with all metrics, physical state, and island membership.
    """
    print(f"\n=== Phase 1: Parameter Scan ({n_per_axis}×{n_per_axis}) ===")

    ALPHA  = np.logspace(-1, 0.7, n_per_axis)
    ALPHAS = np.logspace(-0.5, 0.5, n_per_axis)
    MR     = cfg['MASS_RATIO']
    TEMP   = cfg['TEMPERATURE']
    total  = n_per_axis**2

    print(f"  α:  [{ALPHA[0]:.3f}, {ALPHA[-1]:.3f}]")
    print(f"  αs: [{ALPHAS[0]:.3f}, {ALPHAS[-1]:.3f}]")
    print(f"  mass_ratio={MR}, temperature={TEMP} (fixed)")
    print(f"  {total} combinations × {len(cfg['SCAN_SEEDS'])} seeds")
    print(f"  Output: {cfg['SCAN_FILE']}\n")

    csv_rows = []
    cmap  = np.zeros((n_per_axis,n_per_axis))
    emap  = np.zeros_like(cmap)
    smap  = np.zeros_like(cmap)   # stability
    entr  = np.zeros_like(cmap)   # spatial entropy
    tcomp = np.zeros_like(cmap)
    gzmap = np.zeros_like(cmap)
    state_map = np.full((n_per_axis,n_per_axis),'Dispersed',dtype=object)
    island_map = np.zeros_like(cmap,dtype=bool)
    t0 = time.time()

    for i,alpha in enumerate(ALPHA):
        for j,alpha_s in enumerate(ALPHAS):
            in_isl = in_predicted_island(alpha,alpha_s,MR,TEMP,cfg)
            island_map[i,j] = in_isl

            # Run seeds
            seed_m=[]; seed_st=[]; seed_ent=[]; seed_lf=[]
            for seed in cfg['SCAN_SEEDS']:
                frames,_ = run_sim(alpha,alpha_s,MR,TEMP,cfg,seed=seed)
                m = compute_complexity(frames,alpha_s,cfg)
                state,ent,lf = classify_universe(frames,alpha_s,cfg)
                seed_m.append(m); seed_st.append(state)
                seed_ent.append(ent); seed_lf.append(lf)

            # Average metrics
            avg_m = {k:float(np.mean([m[k] for m in seed_m])) for k in seed_m[0]}
            # Majority vote state
            from collections import Counter
            maj_state = Counter(seed_st).most_common(1)[0][0]
            avg_ent = float(np.mean(seed_ent))
            avg_lf  = float(np.mean(seed_lf))

            cmap[i,j]  = avg_m['complexity']
            emap[i,j]  = avg_m['entity_c']
            smap[i,j]  = avg_m['stability']
            entr[i,j]  = avg_m['sp_entropy']
            tcomp[i,j] = avg_m['t_comp']
            gzmap[i,j] = avg_m['gzip_ratio']
            state_map[i,j] = maj_state

            # CSV row — one row per parameter combination
            row = {
                'alpha':            round(alpha,6),
                'alpha_s':          round(alpha_s,6),
                'mass_ratio':       MR,
                'temperature':      TEMP,
                'complexity':       round(avg_m['complexity'],6),
                'entity_c':         round(avg_m['entity_c'],6),
                'field_c':          round(avg_m['field_c'],6),
                'sp_entropy':       round(avg_m['sp_entropy'],6),
                't_comp':           round(avg_m['t_comp'],6),
                'gzip_ratio':       round(avg_m['gzip_ratio'],6),
                'eoc_weight':       round(avg_m['eoc_weight'],6),
                'stability':        round(avg_m['stability'],6),
                'interact':         round(avg_m['interact'],6),
                'size_div':         round(avg_m['size_div'],6),
                'div_rate':         round(avg_m['div_rate'],6),
                'mean_cluster_count':round(avg_m['count'],4),
                'physical_state':   maj_state,
                'in_predicted_island': in_isl,
                'mean_largest_cluster_frac': round(avg_lf,4),
                'mean_physical_entropy': round(avg_ent,6),
                'seed_state_variance': len(set(seed_st))>1,
            }
            csv_rows.append(row)

            # Progress
            done=(i*n_per_axis+j+1)
            pct=done/total*100
            bar='█'*int(pct//5)+'░'*(20-int(pct//5))
            print(f"\r  [{bar}] {pct:4.0f}%  α={alpha:.3f} αs={alpha_s:.3f} "
                  f"→ {maj_state:<10} C={avg_m['complexity']:.3f}  "
                  f"t={time.time()-t0:.0f}s   ",end='',flush=True)
    print()

    # Save CSV
    with open(cfg['SCAN_FILE'],'w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\n  CSV saved to {cfg['SCAN_FILE']}")

    # Compute correlation scores
    active_mask  = (state_map == 'Active')
    island_mask  = island_map
    top25_mask   = cmap >= np.percentile(cmap,75)

    # Precision: of our top-25% complexity cells, what fraction are physically Active?
    if top25_mask.sum() > 0:
        precision = float((top25_mask & active_mask).sum()/top25_mask.sum())
    else: precision=0.0

    # Recall: of physically Active cells, what fraction are in our top-25%?
    if active_mask.sum() > 0:
        recall = float((top25_mask & active_mask).sum()/active_mask.sum())
    else: recall=0.0

    # F1
    f1 = 2*precision*recall/(precision+recall+1e-8)

    # Island prediction accuracy
    if island_mask.sum()>0:
        island_precision = float((island_mask&active_mask).sum()/island_mask.sum())
        island_recall    = float((island_mask&active_mask).sum()/(active_mask.sum()+1e-8))
        island_f1 = 2*island_precision*island_recall/(island_precision+island_recall+1e-8)
    else:
        island_precision=island_recall=island_f1=0.0

    # Our universe
    oi=np.argmin(np.abs(ALPHA-1.0)); oj=np.argmin(np.abs(ALPHAS-1.0))
    our_score=float(cmap[oi,oj]); our_pct=float(np.mean(cmap<our_score)*100)
    our_state=state_map[oi,oj]

    print(f"\n  === Correlation Results ===")
    print(f"  Metric precision  (top-25% C → Active):  {precision:.3f}")
    print(f"  Metric recall     (Active → top-25% C):  {recall:.3f}")
    print(f"  Metric F1 score:                         {f1:.3f}")
    print(f"  Predicted island precision:              {island_precision:.3f}")
    print(f"  Predicted island recall:                 {island_recall:.3f}")
    print(f"  Predicted island F1:                     {island_f1:.3f}")
    print(f"  Our universe:  score={our_score:.3f}  {our_pct:.0f}th pct  state={our_state}")
    print(f"  Active universes: {active_mask.sum()}/{total} = {active_mask.mean()*100:.0f}%")

    # Save JSON summary
    data = {
        'alpha_range':   ALPHA.tolist(),
        'alphas_range':  ALPHAS.tolist(),
        'complexity_map':cmap.tolist(),
        'entity_map':    emap.tolist(),
        'stability_map': smap.tolist(),
        'entropy_map':   entr.tolist(),
        'tcomp_map':     tcomp.tolist(),
        'gzip_map':      gzmap.tolist(),
        'state_map':     state_map.tolist(),
        'island_map':    island_map.tolist(),
        'n':             n_per_axis,
        'mass_ratio':    MR,
        'temperature':   TEMP,
        'our_universe':  {'alpha':1.0,'alpha_s':1.0,'score':our_score,
                          'percentile':our_pct,'state':our_state,'i':int(oi),'j':int(oj)},
        'scores': {
            'metric_precision':precision,'metric_recall':recall,'metric_f1':f1,
            'island_precision':island_precision,'island_recall':island_recall,
            'island_f1':island_f1,
        },
        'active_fraction': float(active_mask.mean()),
        'predicted_island_bounds': cfg['ISLAND'],
    }
    with open(cfg['SCAN_JSON'],'w') as f: json.dump(data,f,indent=2)
    print(f"  JSON saved to {cfg['SCAN_JSON']}")
    return data

# ============================================================
# PHASE 2: HEATMAP VISUALIZATION
# ============================================================

def show_heatmap(data_file=None, cfg=None):
    if data_file is None: data_file = CFG['SCAN_JSON']
    if not os.path.exists(data_file):
        print(f"\n  No scan file found at '{data_file}'.")
        print(f"  Run: python universe_explorer.py scan\n"); return
    if cfg is None: cfg = CFG

    with open(data_file) as f: data = json.load(f)

    ALPHA  = np.array(data['alpha_range'])
    ALPHAS = np.array(data['alphas_range'])
    CMAP   = np.array(data['complexity_map'])
    SMAP   = np.array(data['state_map'])
    IMAP   = np.array(data['island_map'])
    ENTR   = np.array(data.get('entropy_map', CMAP))
    TCOMP  = np.array(data.get('tcomp_map', CMAP))
    GZIP   = np.array(data.get('gzip_map', CMAP))
    our    = data['our_universe']
    scores = data['scores']
    isl    = data['predicted_island_bounds']

    # Active mask
    active_mask = SMAP=='Active'

    fig = plt.figure(figsize=(16,10),facecolor='#0e0e16')
    fig.suptitle(
        f'Universe Complexity Landscape  —  mass_ratio={data["mass_ratio"]}  '
        f'temperature={data["temperature"]}\n'
        f'Metric F1={scores["metric_f1"]:.3f}  '
        f'Island F1={scores["island_f1"]:.3f}  '
        f'Our universe: {our["percentile"]:.0f}th pct  ({our["state"]})',
        color='#ccccdd',fontsize=10)

    gs = gridspec.GridSpec(2,3,figure=fig,
                           left=0.06,right=0.97,top=0.90,bottom=0.07,
                           hspace=0.45,wspace=0.35)

    panels = [
        (gs[0,0], CMAP,  'Complexity index (composite)',   '#1D9E75'),
        (gs[0,1], ENTR,  'Spatial entropy (P7 — edge of chaos)', '#378ADD'),
        (gs[0,2], TCOMP, 'Temporal compression (P4/P5)',   '#7F77DD'),
        (gs[1,0], GZIP,  'Gzip ratio (P2 — global structure)', '#EF9F27'),
        (gs[1,1], None,  'Physical state map',             None),
        (gs[1,2], None,  'Correlation summary',            None),
    ]

    def style_ax(ax):
        ax.set_facecolor('#0e0e16')
        for sp in ax.spines.values(): sp.set_color('#333344')
        ax.tick_params(colors='#888899',labelsize=7)

    def add_markers(ax, our, isl, ALPHA, ALPHAS):
        # Adams island / predicted island box
        ax.add_patch(mpatches.Rectangle(
            (isl['alpha'][0], isl['alpha_s'][0]),
            isl['alpha'][1]-isl['alpha'][0],
            isl['alpha_s'][1]-isl['alpha_s'][0],
            fill=False,edgecolor='#7F77DD',lw=1.5,
            linestyle='--',zorder=6,label='predicted island'))
        # Our universe
        ax.plot(1.0,1.0,'o',color='#1D9E75',ms=10,zorder=7,
                markeredgecolor='white',markeredgewidth=1.2,
                label=f"our universe")
        # Peak complexity
        pk_idx = np.unravel_index(np.argmax(CMAP),CMAP.shape)
        ax.plot(ALPHA[pk_idx[0]],ALPHAS[pk_idx[1]],'*',
                color='#D85A30',ms=14,zorder=7,
                markeredgecolor='white',markeredgewidth=0.8,
                label='peak complexity')

    for spec,M,title,color in panels[:4]:
        ax = fig.add_subplot(spec); style_ax(ax)
        cmap_c = mcolors.LinearSegmentedColormap.from_list(
            'c',['#0e0e16','#1a2535',color,'#ffffff'],N=256)
        im = ax.imshow(M.T,origin='lower',aspect='auto',
                       extent=[ALPHA[0],ALPHA[-1],ALPHAS[0],ALPHAS[-1]],
                       cmap=cmap_c)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('α (EM coupling)',color='#888899',fontsize=8)
        ax.set_ylabel('αs (strong force)',color='#888899',fontsize=8)
        ax.set_title(title,color='#aaaacc',fontsize=8,pad=3)
        add_markers(ax, our, isl, ALPHA, ALPHAS)
        if title.startswith('Complexity'):
            ax.legend(fontsize=6,facecolor='#1a1a2a',labelcolor='#ccccdd',
                      edgecolor='#333344',loc='upper right')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.tick_params(
            colors='#888899',labelsize=6)

    # Physical state map
    ax5 = fig.add_subplot(gs[1,1]); style_ax(ax5)
    state_rgb = np.zeros((*SMAP.shape,3))
    state_rgb[SMAP=='Active']    = [0.11,0.62,0.46]   # green
    state_rgb[SMAP=='Collapsed'] = [0.85,0.35,0.19]   # red
    state_rgb[SMAP=='Dispersed'] = [0.50,0.47,0.87]   # purple
    ax5.imshow(state_rgb.transpose(1,0,2),origin='lower',aspect='auto',
               extent=[ALPHA[0],ALPHA[-1],ALPHAS[0],ALPHAS[-1]])
    ax5.set_xscale('log'); ax5.set_yscale('log')
    ax5.set_xlabel('α',color='#888899',fontsize=8)
    ax5.set_ylabel('αs',color='#888899',fontsize=8)
    ax5.set_title('Physical state (ground truth)',color='#aaaacc',fontsize=8,pad=3)
    add_markers(ax5,our,isl,ALPHA,ALPHAS)
    patches=[mpatches.Patch(color=[0.11,0.62,0.46],label='Active (complex)'),
             mpatches.Patch(color=[0.85,0.35,0.19],label='Collapsed'),
             mpatches.Patch(color=[0.50,0.47,0.87],label='Dispersed')]
    ax5.legend(handles=patches,fontsize=6,facecolor='#1a1a2a',
               labelcolor='#ccccdd',edgecolor='#333344',loc='upper right')

    # Correlation summary
    ax6 = fig.add_subplot(gs[1,2]); ax6.axis('off')
    ax6.set_facecolor('#0e0e16')
    active_frac = float(np.mean(SMAP=='Active'))*100
    summary = (
        f"Correlation scores\n"
        f"{'─'*32}\n\n"
        f"Metrics vs Physical state\n"
        f"  Precision  {scores['metric_precision']:.3f}\n"
        f"  Recall     {scores['metric_recall']:.3f}\n"
        f"  F1 score   {scores['metric_f1']:.3f}\n\n"
        f"Predicted island vs Physical\n"
        f"  Precision  {scores['island_precision']:.3f}\n"
        f"  Recall     {scores['island_recall']:.3f}\n"
        f"  F1 score   {scores['island_f1']:.3f}\n\n"
        f"Our universe\n"
        f"  Score      {our['score']:.3f}\n"
        f"  Percentile {our['percentile']:.0f}th\n"
        f"  State      {our['state']}\n\n"
        f"Parameter space\n"
        f"  Active     {active_frac:.0f}%\n"
        f"  Collapsed  {float(np.mean(SMAP=='Collapsed'))*100:.0f}%\n"
        f"  Dispersed  {float(np.mean(SMAP=='Dispersed'))*100:.0f}%\n\n"
        f"From CA experiments:\n"
        f"  Complexity peaks at\n"
        f"  intermediate entropy\n"
        f"  (edge-of-chaos principle)"
    )
    ax6.text(0.05,0.97,summary,transform=ax6.transAxes,color='#ccccdd',
             fontsize=8,va='top',fontfamily='monospace',
             bbox=dict(boxstyle='round',facecolor='#1a1a2a',
                       edgecolor='#333344',alpha=0.9))

    plt.savefig('universe_heatmap.png',dpi=130,facecolor='#0e0e16')
    print("  Saved universe_heatmap.png")
    plt.show()

# ============================================================
# LIVE SIMULATION
# ============================================================

def run_simulation(alpha=1.0, alpha_s=1.0, mass_ratio=10.0, temperature=0.35):
    print(f"\nLive Simulation  α={alpha}  αs={alpha_s}  "
          f"mass_ratio={mass_ratio}  temp={temperature}")
    print("Close window to exit.\n")
    cfg = CFG.copy()
    cfg.update({'ALPHA':alpha,'ALPHA_S':alpha_s,
                'MASS_RATIO':mass_ratio,'TEMPERATURE':temperature})
    pos,vel,masses = init_particles(cfg)
    frame_buf=[]; metric_hist={'complexity':[],'count':[],'entropy':[]}

    fig=plt.figure(figsize=(13,6),facecolor='#0e0e16')
    fig.suptitle(f'N-body Simulation  α={alpha:.3f}  αs={alpha_s:.3f}  '
                 f'mass_ratio={mass_ratio:.1f}  temp={temperature:.2f}',
                 color='#ccccdd',fontsize=11)
    gs=gridspec.GridSpec(2,3,figure=fig,left=0.05,right=0.97,
                         top=0.91,bottom=0.08,hspace=0.5,wspace=0.35)
    ax_main=fig.add_subplot(gs[:,0])
    ax_cx=fig.add_subplot(gs[0,1]); ax_cnt=fig.add_subplot(gs[1,1])
    ax_ent=fig.add_subplot(gs[0,2]); ax_info=fig.add_subplot(gs[1,2])

    for ax in [ax_main,ax_cx,ax_cnt,ax_ent,ax_info]:
        ax.set_facecolor('#0e0e16')
        for sp in ax.spines.values(): sp.set_color('#333344')
        ax.tick_params(colors='#888899',labelsize=8)

    ax_main.set_xlim(0,cfg['BOX']); ax_main.set_ylim(0,cfg['BOX'])
    ax_main.set_aspect('equal')
    ax_main.set_title('particle field  (■ heavy  · light)',color='#8888aa',fontsize=9)

    # Heavy and light particle scatters
    n_h=cfg['N_HEAVY']; n_l=cfg['N_LIGHT']
    scat_h=ax_main.scatter(pos[:n_h,0],pos[:n_h,1],s=40,
                            c='#EF9F27',alpha=0.9,marker='s',zorder=4)
    scat_l=ax_main.scatter(pos[n_h:,0],pos[n_h:,1],s=8,
                            c='#5577ff',alpha=0.7,zorder=3)

    ax_cx.set_title('complexity',color='#8888aa',fontsize=9)
    cx_line,=ax_cx.plot([],[],color='#1D9E75',lw=1.5)
    ax_cx.set_xlim(0,80)

    ax_cnt.set_title('cluster count',color='#8888aa',fontsize=9)
    cnt_line,=ax_cnt.plot([],[],color='#7F77DD',lw=1.5)
    ax_cnt.set_xlim(0,80)

    ax_ent.set_title('spatial entropy (edge-of-chaos)',color='#8888aa',fontsize=9)
    ent_line,=ax_ent.plot([],[],color='#378ADD',lw=1.5)
    # Draw edge-of-chaos target zone
    ax_ent.axhspan(0.4,0.7,color='#1D9E7520',zorder=0,label='complex zone')
    ax_ent.set_xlim(0,80); ax_ent.set_ylim(0,1)
    ax_ent.legend(fontsize=6,facecolor='#1a1a2a',labelcolor='#ccccdd',
                  edgecolor='#333344')

    ax_info.axis('off')
    info_txt=ax_info.text(0.05,0.97,'',transform=ax_info.transAxes,
                          color='#aaaacc',fontsize=8,va='top',
                          fontfamily='monospace')
    cluster_patches=[]
    step_n=[0]

    def update(frame):
        nonlocal pos,vel
        for _ in range(3):
            pos,vel=step_sim(pos,vel,masses,alpha,alpha_s,cfg)
            step_n[0]+=1

        clusters=find_clusters(pos,alpha_s,cfg['BOX'])
        colors_h=['#EF9F27']*n_h; colors_l=['#334466']*n_l
        pal=['#1D9E75','#D85A30','#7F77DD','#378ADD','#EF9F27','#ED93B1']
        for idx,cl in enumerate(clusters[:len(pal)]):
            for m in cl:
                if m<n_h: colors_h[m]=pal[idx%len(pal)]
                else: colors_l[m-n_h]=pal[idx%len(pal)]
        scat_h.set_offsets(pos[:n_h]); scat_h.set_color(colors_h)
        scat_l.set_offsets(pos[n_h:]); scat_l.set_color(colors_l)

        for p in cluster_patches: p.remove()
        cluster_patches.clear()
        for cl in clusters:
            c=pos[cl].mean(axis=0)
            r=np.sqrt(len(cl))*alpha_s*1.5
            p=plt.Circle(c,r,fill=False,color='#ffffff18',lw=0.5,zorder=2)
            ax_main.add_patch(p); cluster_patches.append(p)

        frame_buf.append((pos.copy(),vel.copy()))
        if len(frame_buf)>40: frame_buf.pop(0)

        if frame%4==0 and len(frame_buf)>=5:
            m=compute_complexity(frame_buf,alpha_s,cfg)
            for k,v in [('complexity',m['complexity']),
                        ('count',m['count']),
                        ('entropy',m['sp_entropy'])]:
                metric_hist[k].append(v)
            if len(metric_hist['complexity'])>80:
                for k in metric_hist: metric_hist[k].pop(0)

            x=list(range(len(metric_hist['complexity'])))
            cx_line.set_data(x,metric_hist['complexity'])
            cnt_line.set_data(x,metric_hist['count'])
            ent_line.set_data(x,metric_hist['entropy'])
            if metric_hist['complexity']:
                ax_cx.set_ylim(0,max(max(metric_hist['complexity'])*1.15,1))
                ax_cnt.set_ylim(0,max(max(metric_hist['count'])*1.15,1))

            # Classify state
            state,_,_ = classify_universe(frame_buf,alpha_s,cfg)
            state_colors={'Active':'#1D9E75','Collapsed':'#D85A30','Dispersed':'#7F77DD'}
            info_txt.set_text(
                f"step:      {step_n[0]:,}\n"
                f"clusters:  {len(clusters)}\n"
                f"state:     {state}\n"
                f"complex:   {m['complexity']:.3f}\n"
                f"entropy:   {m['sp_entropy']:.3f}\n"
                f"stability: {m['stability']:.3f}\n"
                f"interact:  {m['interact']:.3f}\n"
                f"t_comp:    {m['t_comp']:.3f}\n"
                f"gzip:      {m['gzip_ratio']:.3f}\n\n"
                f"α  = {alpha:.4f}\n"
                f"αs = {alpha_s:.4f}"
            )
            info_txt.set_color(state_colors.get(state,'#aaaacc'))

        return [scat_h,scat_l]+cluster_patches+[cx_line,cnt_line,ent_line,info_txt]

    ani=FuncAnimation(fig,update,interval=40,blit=False,cache_frame_data=False)
    plt.show()

# ============================================================
# INTERACTIVE MENU
# ============================================================

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  UNIVERSE COMPLEXITY EXPLORER v2                         ║
║  N-body simulation with tunable physical constants       ║
╚══════════════════════════════════════════════════════════╝

  Based on eight candidate laws of complexity.
  Key insight from CA experiments: complexity lives at the
  edge of chaos — intermediate entropy, not maximum or minimum.

  Constants (our universe = normalized to 1.0):
    α   = EM coupling       (fine structure constant analog)
    αs  = strong force      (equilibrium separation)
    m   = mass ratio        (proton/electron mass analog)
    T   = temperature       (early universe thermal energy)

  Predicted island of stability:
    α  ∈ [0.20, 4.00]
    αs ∈ [0.40, 2.50]
    m  ∈ [4.00, 80.0]
    T  ∈ [0.05, 0.80]
"""

def menu():
    print(BANNER)
    while True:
        print("  Options:")
        print("  1. Live simulation (visualise a single universe)")
        print("  2. Calibration sweeps (Phase 0 — axis profiles)")
        print("  3. Parameter scan (Phase 1 — full α×αs heatmap + CSV)")
        print("  4. Show heatmap (Phase 2 — visualise saved scan)")
        print("  5. Quit")
        print()
        ch=input("  Enter 1-5: ").strip()
        print()

        if ch=='1':
            a   = float(input(f"  α  [{CFG['ALPHA']}]: ").strip() or CFG['ALPHA'])
            aps = float(input(f"  αs [{CFG['ALPHA_S']}]: ").strip() or CFG['ALPHA_S'])
            mr  = float(input(f"  mass_ratio [{CFG['MASS_RATIO']}]: ").strip() or CFG['MASS_RATIO'])
            tmp = float(input(f"  temperature [{CFG['TEMPERATURE']}]: ").strip() or CFG['TEMPERATURE'])
            run_simulation(a,aps,mr,tmp)
        elif ch=='2':
            run_calibration(CFG)
        elif ch=='3':
            n=input(f"  Grid size per axis [12]: ").strip()
            run_scan(CFG, int(n) if n else 12)
        elif ch=='4':
            show_heatmap()
        elif ch=='5':
            print("  Goodbye.\n"); break
        else:
            print("  Please enter 1–5.")
        print()

if __name__=='__main__':
    args=sys.argv[1:]
    if not args: menu()
    elif args[0]=='simulate':
        run_simulation(
            float(args[1]) if len(args)>1 else 1.0,
            float(args[2]) if len(args)>2 else 1.0,
            float(args[3]) if len(args)>3 else 10.0,
            float(args[4]) if len(args)>4 else 0.35)
    elif args[0]=='calibrate': run_calibration(CFG)
    elif args[0]=='scan':
        n=int(args[1]) if len(args)>1 else 12
        run_scan(CFG,n)
    elif args[0] in ('plot','heatmap'):
        show_heatmap(args[1] if len(args)>1 else None)
    else:
        print(__doc__)
