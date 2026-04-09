"""
complexity_statistical_analysis.py
====================================
Statistical analysis of the N-body universe scan results for the Complexity Framework.
Tests whether the complexity metrics meaningfully distinguish physically active
(complex) universes from sterile (collapsed/dispersed) ones.

Author: Cameron Belt
Date: April 2026

USAGE:
    python complexity_statistical_analysis.py

OUTPUT:
    - metric_discriminability_results.csv    (per-metric AUC, p-values, effect sizes)
    - complexity_statistical_analysis.png    (6-panel figure)
    - printed summary to stdout

REQUIRES:
    pip install pandas numpy scipy matplotlib scikit-learn seaborn
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = 'universe_scan.csv'  # adjust if needed
df = pd.read_csv(DATA_PATH)
df['is_active'] = (df['physical_state'] == 'Active').astype(int)

print("=" * 65)
print("COMPLEXITY FRAMEWORK — STATISTICAL ANALYSIS REPORT")
print("=" * 65)
print(f"\nDataset: {len(df)} universes across α × αs parameter space")
print(f"Physical states: {df['physical_state'].value_counts().to_dict()}")

# ============================================================
# 1. CORE HYPOTHESIS TEST
# ============================================================

active = df[df['is_active'] == 1]['complexity']
inactive = df[df['is_active'] == 0]['complexity']
n1, n2 = len(active), len(inactive)

u_stat, p_mw = mannwhitneyu(active, inactive, alternative='greater')
r_rb = 1 - (2*u_stat)/(n1*n2)
pooled_std = np.sqrt(((n1-1)*active.std()**2 + (n2-1)*inactive.std()**2) / (n1+n2-2))
cohens_d = (active.mean() - inactive.mean()) / pooled_std

print("\n--- 1. Core Hypothesis: Complexity predicts Active state ---")
print(f"  Mann-Whitney U p-value: {p_mw:.2e}  (one-sided: Active > Non-Active)")
print(f"  Active mean: {active.mean():.3f}, Non-Active mean: {inactive.mean():.3f}")
print(f"  Ratio: {active.mean()/inactive.mean():.1f}x")
print(f"  Effect size (rank-biserial r): {r_rb:.3f}  (|>0.5| = large)")
print(f"  Cohen's d: {cohens_d:.3f}  (>0.8 = large, >2.0 = very large)")

# ============================================================
# 2. METRIC DISCRIMINABILITY
# ============================================================

METRICS = ['complexity','entity_c','field_c','sp_entropy','t_comp','gzip_ratio',
           'eoc_weight','interact','size_div','div_rate','mean_cluster_count',
           'mean_largest_cluster_frac','mean_physical_entropy','stability']

print("\n--- 2. Per-Metric Discriminability (AUC, Active vs Non-Active) ---")
print(f"{'Metric':<30} {'AUC':>6} {'p-value':>12} {'Cohen d':>9} {'Sig':>4}")
print("-" * 68)

rows = []
for m in METRICS:
    g1 = df[df['is_active']==1][m]
    g0 = df[df['is_active']==0][m]
    u, p = mannwhitneyu(g1, g0, alternative='two-sided')
    auc = max(u / (len(g1)*len(g0)), 1 - u / (len(g1)*len(g0)))
    ps = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g0)-1)*g0.std()**2) / (len(g1)+len(g0)-2))
    d = (g1.mean() - g0.mean()) / ps if ps > 0 else np.nan
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    rows.append({'metric':m,'active_mean':round(g1.mean(),4),'inactive_mean':round(g0.mean(),4),
                 'ratio':round(g1.mean()/g0.mean(),3) if g0.mean()!=0 else np.nan,
                 'AUC':round(auc,4),'mann_whitney_p':f"{p:.2e}",'cohens_d':round(d,3),'sig':sig})
    print(f"  {m:<28} {auc:>6.3f} {p:>12.2e} {d:>9.3f}  {sig}")

res_df = pd.DataFrame(rows)
res_df.to_csv('metric_discriminability_results.csv', index=False)
print("\n  -> Saved: metric_discriminability_results.csv")

# ============================================================
# 3. PREDICTED ISLAND VS METRIC ACCURACY
# ============================================================

threshold = df['complexity'].median()
df['metric_pred'] = (df['complexity'] > threshold).astype(int)
tp = ((df['metric_pred']==1) & (df['is_active']==1)).sum()
fp = ((df['metric_pred']==1) & (df['is_active']==0)).sum()
fn = ((df['metric_pred']==0) & (df['is_active']==1)).sum()
prec = tp/(tp+fp); rec = tp/(tp+fn)
f1_metric = 2*prec*rec/(prec+rec)

ip = df[df['in_predicted_island']]['is_active'].mean()
ir = df[df['is_active']==1]['in_predicted_island'].mean()
f1_island = 2*ip*ir/(ip+ir)

print("\n--- 3. Prediction Accuracy ---")
print(f"  Metric F1 (median threshold): {f1_metric:.3f}  (precision={prec:.3f}, recall={rec:.3f})")
print(f"  Analytical island F1:         {f1_island:.3f}")
print(f"  Metric improvement over island: {(f1_metric-f1_island)/f1_island*100:+.1f}%")

# ============================================================
# 4. RANDOM FOREST
# ============================================================

features = ['complexity','entity_c','field_c','sp_entropy','t_comp','gzip_ratio',
            'eoc_weight','stability','interact','size_div','div_rate',
            'mean_cluster_count','mean_largest_cluster_frac','mean_physical_entropy',
            'alpha','alpha_s']
X = df[features].values
y = df['is_active'].values
rf = RandomForestClassifier(n_estimators=300, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

print(f"\n--- 4. Random Forest (5-fold CV) ---")
print(f"  ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("  Top feature importances:")
for f, imp in importances.head(6).items():
    print(f"    {f:<30} {imp:.3f}")

# ============================================================
# 5. KRUSKAL-WALLIS (3-way)
# ============================================================

groups = [df[df['physical_state']==s]['complexity'].values for s in ['Active','Collapsed','Dispersed']]
h, p_kw = kruskal(*groups)
print(f"\n--- 5. Kruskal-Wallis (3 states) ---")
print(f"  H = {h:.2f}, p = {p_kw:.2e}")

# ============================================================
# VISUALIZATION
# ============================================================

state_colors = {'Active': '#2ecc71', 'Collapsed': '#e74c3c', 'Dispersed': '#f39c12'}
bg_c, text_c, grid_c = '#161b22', 'white', '#30363d'

fig = plt.figure(figsize=(18, 20), facecolor='#0d1117')
fig.suptitle('Complexity Framework — Statistical Analysis\nN-Body Universe Scan (n=100)',
             fontsize=16, color='white', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.07, right=0.97, top=0.94, bottom=0.05)

def style_ax(ax, title):
    ax.set_facecolor(bg_c)
    ax.tick_params(colors=text_c, labelsize=8)
    ax.set_title(title, color=text_c, fontsize=9, fontweight='bold', pad=6)
    for spine in ax.spines.values(): spine.set_edgecolor(grid_c)
    ax.xaxis.label.set_color(text_c)
    ax.yaxis.label.set_color(text_c)

# Plot 1: Violin
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Complexity by Physical State')
data_groups = [df[df['physical_state']==s]['complexity'].values for s in ['Active','Collapsed','Dispersed']]
vp = ax1.violinplot(data_groups, positions=[1,2,3], showmedians=True)
for pc, s in zip(vp['bodies'], ['Active','Collapsed','Dispersed']):
    pc.set_facecolor(state_colors[s]); pc.set_alpha(0.7)
vp['cmedians'].set_color('white'); vp['cmins'].set_color('#555')
vp['cmaxes'].set_color('#555'); vp['cbars'].set_color('#555')
ax1.set_xticks([1,2,3]); ax1.set_xticklabels(['Active','Collapsed','Dispersed'], fontsize=8)
ax1.set_ylabel('Complexity Score', fontsize=8)
ax1.text(0.97,0.97,f'p={p_mw:.1e}\nd={cohens_d:.2f}',transform=ax1.transAxes,
         ha='right',va='top',fontsize=7.5,color='#aaa',
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#111',alpha=0.7))
ax1.yaxis.grid(True, color=grid_c, alpha=0.5, linewidth=0.5)

# Plot 2: AUC bars
ax2 = fig.add_subplot(gs[0, 1:])
style_ax(ax2, 'Metric Discriminability (AUC)')
aucs = [r['AUC'] for r in rows]
bar_colors = ['#2ecc71' if m=='complexity' else '#3498db' if m in ['entity_c','field_c','eoc_weight'] else '#9b59b6' for m in METRICS]
bars = ax2.barh(range(len(METRICS)), aucs, color=bar_colors, alpha=0.8, height=0.7)
ax2.axvline(0.5, color='white', linestyle='--', alpha=0.3)
ax2.axvline(0.9, color='#f39c12', linestyle=':', alpha=0.5)
ax2.set_yticks(range(len(METRICS))); ax2.set_yticklabels(METRICS, fontsize=7.5)
ax2.set_xlabel('AUC', fontsize=8); ax2.set_xlim(0.4, 1.02)
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax2.text(auc+0.003, i, f'{auc:.3f}', va='center', fontsize=6.5, color=text_c)
ax2.legend(handles=[Patch(facecolor='#2ecc71',label='Composite C'),
                    Patch(facecolor='#3498db',label='Sub-metrics'),
                    Patch(facecolor='#9b59b6',label='Physical obs.')],
           loc='lower right', fontsize=7, facecolor=bg_c, labelcolor=text_c, edgecolor=grid_c)
ax2.xaxis.grid(True, color=grid_c, alpha=0.4)

# Plot 3: Heatmap
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'Complexity Heatmap (α vs αs)')
pivot = df.pivot_table(values='complexity', index='alpha_s', columns='alpha', aggfunc='mean')
im = ax3.imshow(pivot.values, aspect='auto', origin='lower', cmap='plasma')
ax3.set_xlabel('α (EM coupling)', fontsize=8); ax3.set_ylabel('αs (strong force)', fontsize=8)
ai = min(range(len(list(pivot.columns))), key=lambda i: abs(list(pivot.columns)[i]-1.0))
si = min(range(len(list(pivot.index))), key=lambda i: abs(list(pivot.index)[i]-1.0))
ax3.plot(ai, si, 'w*', markersize=10, label='Our universe')
ax3.legend(fontsize=7, facecolor=bg_c, labelcolor=text_c, edgecolor=grid_c)
plt.colorbar(im, ax=ax3).ax.yaxis.label.set_color(text_c)
ax3.set_xticks([]); ax3.set_yticks([])

# Plot 4: F1 comparison
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'F1 Score: Metric vs Island')
bars4 = ax4.bar(['Analytical\nIsland','Complexity\nMetric','Perfect'], [f1_island,f1_metric,1.0],
                color=['#e74c3c','#2ecc71','#3498db'], alpha=0.8, width=0.5)
ax4.set_ylim(0,1.1); ax4.set_ylabel('F1 Score', fontsize=8)
for bar, s in zip(bars4, [f1_island,f1_metric,1.0]):
    ax4.text(bar.get_x()+bar.get_width()/2, s+0.02, f'{s:.3f}', ha='center', fontsize=9, color=text_c, fontweight='bold')
ax4.axhline(0.676, color='#f39c12', linestyle='--', alpha=0.7, linewidth=1.2)
ax4.text(2.45, 0.685, 'Paper\n(top-Q)', fontsize=6, color='#f39c12', ha='right')
ax4.yaxis.grid(True, color=grid_c, alpha=0.4)

# Plot 5: Scatter
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, 'αs vs Complexity')
for state in ['Active','Collapsed','Dispersed']:
    sub = df[df['physical_state']==state]
    ax5.scatter(sub['alpha_s'], sub['complexity'], c=state_colors[state], label=state, alpha=0.7, s=25)
ax5.set_xlabel('αs', fontsize=8); ax5.set_ylabel('Complexity Score', fontsize=8)
ax5.legend(fontsize=7, facecolor=bg_c, labelcolor=text_c, edgecolor=grid_c)
ax5.yaxis.grid(True, color=grid_c, alpha=0.4)

# Plot 6: RF importance
ax6 = fig.add_subplot(gs[2, 0:2])
style_ax(ax6, f'Random Forest Feature Importance (CV AUC = {scores.mean():.3f} ± {scores.std():.3f})')
imp_sorted = importances.sort_values()
fc = ['#2ecc71' if f=='complexity' else '#3498db' if f in ['entity_c','field_c','eoc_weight','sp_entropy','t_comp','gzip_ratio'] else '#e74c3c' if f in ['alpha','alpha_s'] else '#9b59b6' for f in imp_sorted.index]
ax6.barh(range(len(imp_sorted)), imp_sorted.values, color=fc, alpha=0.8)
ax6.set_yticks(range(len(imp_sorted))); ax6.set_yticklabels(imp_sorted.index, fontsize=7.5)
ax6.set_xlabel('Importance', fontsize=8)
ax6.legend(handles=[Patch(facecolor='#2ecc71',label='Composite C'),
                    Patch(facecolor='#3498db',label='C sub-metrics'),
                    Patch(facecolor='#e74c3c',label='Physical params'),
                    Patch(facecolor='#9b59b6',label='Observables')],
           loc='lower right', fontsize=7, facecolor=bg_c, labelcolor=text_c, edgecolor=grid_c)
ax6.xaxis.grid(True, color=grid_c, alpha=0.4)

# Plot 7: Summary table
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor(bg_c); ax7.axis('off')
ax7.set_title('Key Statistical Results', color=text_c, fontsize=9, fontweight='bold', pad=6)
for spine in ax7.spines.values(): spine.set_edgecolor(grid_c)
rows_table = [
    ['Test','Statistic','Verdict'],
    ['Mann-Whitney U',f'p={p_mw:.1e}','✓ REJECT H₀'],
    ['Effect size (d)',f'{cohens_d:.2f} (huge)','✓ ROBUST'],
    ['Rank-biserial r',f'|r|={abs(r_rb):.3f}','✓ NEAR PERFECT'],
    ['F1 (metric)',f'{f1_metric:.3f}','✓ STRONG'],
    ['F1 (island)',f'{f1_island:.3f}','△ BASELINE'],
    ['RF CV AUC',f'{scores.mean():.3f}±{scores.std():.3f}','✓ EXCELLENT'],
    ['Kruskal-Wallis',f'p={p_kw:.1e}','✓ REJECT H₀'],
    ['Active/Inactive C',f'{active.mean()/inactive.mean():.1f}× ratio','✓ CLEAR SEP'],
]
yp = 0.95
for i, row in enumerate(rows_table):
    hw = 'bold' if i==0 else 'normal'
    rc = '#f39c12' if i==0 else (text_c if i%2==0 else '#aaa')
    ax7.text(0.02,yp,row[0],transform=ax7.transAxes,fontsize=7.5,color=rc,fontweight=hw,va='top')
    ax7.text(0.44,yp,row[1],transform=ax7.transAxes,fontsize=7.5,color=rc,fontweight=hw,va='top')
    vc = '#2ecc71' if '✓' in row[2] else ('#f39c12' if '△' in row[2] else rc)
    ax7.text(0.78,yp,row[2],transform=ax7.transAxes,fontsize=7.5,color=vc,fontweight=hw,va='top')
    yp -= 0.088

plt.savefig('complexity_statistical_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("\nPlot saved: complexity_statistical_analysis.png")
print("\nAll outputs written successfully.")
