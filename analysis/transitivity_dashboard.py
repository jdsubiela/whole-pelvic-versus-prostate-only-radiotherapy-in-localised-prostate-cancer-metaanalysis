#!/usr/bin/env python3
"""
Cross-trial comparability ("transitivity") dashboard — WPRT vs PORT in localized prostate cancer
================================================================================================
Five RCTs: GETUG-01, RTOG 9413, POP-RT, RTOG 0924, PEACE-2.

Pairwise MA (all trials randomize WPRT vs PORT directly) — "transitivity" here = clinical/
methodological COMPARABILITY of effect-modifier distributions across trials. Not-reported / not-
derivable cells are shown as 'n.r.' (grey) and EXCLUDED from normalization and the atypicality score
(no fabricated values).

3-panel composite: (a) heatmap of modifiers x trials; (b) radar (complete-data axes only);
(c) atypicality score (mean |deviation from cross-trial median| over reported cells).

Footnote keys: * estimate for visualization; † not located in primary source; ‡ observed median
(others prescribed); n.r. = not reported / not derivable from available sources.
Output: analysis/figures/transitivity_dashboard.{png,pdf}
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
DPI = 300
NA = np.nan

TRIALS = ['GETUG-01', 'RTOG 9413', 'POP-RT', 'RTOG 0924', 'PEACE-2']
# Coordinated manuscript palette (POP-RT coral and RTOG 0924 teal match Figure 4)
COLORS = ['#3D5A80', '#E8A33D', '#E8694A', '#0E7C7B', '#8E6BAA']

# label, numeric values (NA = not reported), display strings
DIMS = [
    ('% T3-T4',                 [26, 67, 78.3, 0, 90],     ['26', '67', '78', '0', '90']),
    ('% Gleason 7-10',          [51, 74, 90, 98, NA],      ['51', '74', '90', '98', 'n.r.*']),
    ('Median PSA (ng/mL)',      [12, 23, 28, NA, 22],      ['12', '23', '28', 'n.r.', '22']),
    ('Nodal risk >=35% (%)',    [9.8, NA, 55, NA, 57],     ['9.8', 'n.r.†', '55', 'n.r.', '57']),
    ('Pelvic EQD2 a/b=3 (Gy)',  [46, 48.4, 50, 43.2, 47],  ['46', '48.4', '50', '43.2', '~47*']),
    ('ADT duration (mo)',       [6, 4, 24, 13, 33],        ['4-8', '4', '24', '4-32', '33‡']),
    ('PSMA-PET staging (%)',    [0, 0, 80, 0, 0],          ['0', '0', '80', '0', 'few*']),
    ('Common-iliac coverage',   [0, 10, 100, 90, 50],      ['No', '~No', 'Yes', 'Yes', 'Partial']),
    ('IMRT (vs 2D/3D)',         [40, 0, 100, 90, 100],     ['2D/3D', '2D', 'IMRT', 'IMRT', 'IMRT']),
    ('Single-centre',           [0, 0, 100, 0, 0],         ['No', 'No', 'Yes', 'No', 'No']),
    ('Chemotherapy',            [0, 0, 0, 0, 100],         ['No', 'No', 'No', 'No', 'Cabaz.']),
    ('Prostate dose (Gy)',      [68, 70.2, 68, 79.2, 78],  ['66–70', '70.2', '68', '79.2', '78']),
    ('ADT mandatory',           [0, 100, 100, 100, 100],   ['Optional', 'Yes', 'Yes', 'Yes', 'Yes']),
]
labels = [d[0] for d in DIMS]
M = np.array([d[1] for d in DIMS], dtype=float)   # rows=dims, cols=trials (NaN allowed)
disp = [d[2] for d in DIMS]


def norm_rows(mat):
    out = np.full_like(mat, np.nan)
    for i in range(mat.shape[0]):
        row = mat[i]; fin = np.isfinite(row)
        lo, hi = np.min(row[fin]), np.max(row[fin])
        out[i, fin] = 0.5 if hi == lo else (row[fin] - lo) / (hi - lo)
    return out

Mn = norm_rows(M)

fig = plt.figure(figsize=(13.6, 6.7))
gs = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30,
              left=0.05, right=0.965, top=0.83, bottom=0.13)
ax_b = fig.add_subplot(gs[0, 0], polar=True)
ax_c = fig.add_subplot(gs[0, 1])

# ── (a) radar — complete-data axes (clinical + design characteristics) ──
cand = [(0, '% T3–4'), (11, 'Prostate dose'), (4, 'Pelvic EQD2'), (5, 'ADT mo'),
        (12, 'ADT mandatory'), (6, 'PSMA stag.'), (7, 'Common-iliac'), (8, 'IMRT'),
        (9, 'Single-centre'), (10, 'Chemo')]
radar = [(i, name) for i, name in cand if np.all(np.isfinite(M[i]))]
ridx = [i for i, _ in radar]; rcats = [n for _, n in radar]
R = Mn[ridx, :] * 100.0
ang = np.linspace(0, 2 * np.pi, len(rcats), endpoint=False).tolist(); ang += ang[:1]
for j, tr in enumerate(TRIALS):
    vals = R[:, j].tolist(); vals += vals[:1]
    lw = 2.6 if tr == 'POP-RT' else 1.7
    ax_b.plot(ang, vals, 'o-', color=COLORS[j], linewidth=lw, markersize=4.2)
    ax_b.fill(ang, vals, color=COLORS[j], alpha=0.06)
ax_b.set_xticks(ang[:-1]); ax_b.set_xticklabels(rcats, fontsize=9)
ax_b.tick_params(axis='x', pad=8); ax_b.set_ylim(0, 100)
ax_b.set_yticklabels([]); ax_b.set_rlabel_position(90)
ax_b.grid(color='#D9DDE2', lw=0.7)
ax_b.set_title('a', fontsize=15, fontweight='bold', loc='left', color='#1A1A1A', pad=24)

# ── (b) atypicality (over reported cells) ──
med = np.nanmedian(Mn, axis=1, keepdims=True)
atyp = np.nanmean(np.abs(Mn - med), axis=0)
order = np.argsort(atyp)
ax_c.barh([TRIALS[k] for k in order], [atyp[k] for k in order],
          color=[COLORS[k] for k in order], alpha=0.92, edgecolor='white', linewidth=0.8)
for rank, k in enumerate(order):
    ax_c.text(atyp[k] + 0.005, rank, f'{atyp[k]:.2f}', va='center', fontsize=10,
              fontweight='bold', color=COLORS[k])
ax_c.set_xlabel('Mean |deviation from cross-trial median| over reported cells\n'
                '(higher = breaks comparability more)', fontsize=9.5)
ax_c.set_xlim(0, max(atyp) * 1.18)
ax_c.set_title('b', fontsize=15, fontweight='bold', loc='left', color='#1A1A1A', pad=18)
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.xaxis.grid(True, alpha=0.25, linestyle='--'); ax_c.set_axisbelow(True)
for lab, k in zip(ax_c.get_yticklabels(), order):
    lab.set_color(COLORS[k])

# shared trial legend only (title and footnote moved to the manuscript legend)
fig.legend(handles=[Line2D([0], [0], color=COLORS[j], marker='o', lw=2.5, markersize=6, label=TRIALS[j])
                    for j in range(len(TRIALS))],
           loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=5, frameon=False, fontsize=11)

fig.savefig(os.path.join(OUT, 'transitivity_dashboard.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)

print('Saved: analysis/figures/transitivity_dashboard.png')
print('\nAtypicality (over reported cells):')
for k in np.argsort(-atyp):
    print(f'  {TRIALS[k]:<12} {atyp[k]:.3f}')
print(f'\nRadar axes used (complete data): {rcats}')
