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
DPI = 250
NA = np.nan

TRIALS = ['GETUG-01', 'RTOG 9413', 'POP-RT', 'RTOG 0924', 'PEACE-2']
COLORS = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']  # POP-RT = red

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

fig = plt.figure(figsize=(17, 13))
gs = GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28,
              left=0.13, right=0.97, top=0.86, bottom=0.09)
ax_a = fig.add_subplot(gs[0, :]); ax_b = fig.add_subplot(gs[1, 0], polar=True)
ax_c = fig.add_subplot(gs[1, 1])

# ── (a) heatmap ──
cmap = plt.cm.YlGnBu.copy(); cmap.set_bad('#EAEAEA')
im = ax_a.imshow(np.ma.masked_invalid(Mn), aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax_a.set_xticks(range(len(TRIALS))); ax_a.set_xticklabels(TRIALS, fontsize=12, fontweight='bold')
ax_a.set_yticks(range(len(labels))); ax_a.set_yticklabels(labels, fontsize=10.5)
ax_a.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
for j in range(len(TRIALS)):
    ax_a.get_xticklabels()[j].set_color(COLORS[j])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(Mn[i, j]):
            col = '#999999'
        else:
            col = 'white' if Mn[i, j] > 0.6 else '#222222'
        ax_a.text(j, i, disp[i][j], ha='center', va='center', fontsize=9.3,
                  fontweight='bold', color=col)
ax_a.set_title('a   Effect-modifier comparability across trials '
               '(per-row normalized over reported cells; darker = higher; grey = n.r.)',
               fontsize=12, fontweight='bold', loc='left', pad=12)
cbar = fig.colorbar(im, ax=ax_a, fraction=0.025, pad=0.02)
cbar.set_label('Within-dimension position (0=min, 1=max)', fontsize=9)

# ── (b) radar — complete-data axes only ──
cand = [(0, '% T3-4'), (1, '% GS 7-10'), (2, 'Median PSA'),
        (4, 'Pelvic EQD2'), (5, 'ADT mo'), (6, 'PSMA stag.')]
radar = [(i, name) for i, name in cand if np.all(np.isfinite(M[i]))]
ridx = [i for i, _ in radar]; rcats = [n for _, n in radar]
R = Mn[ridx, :] * 100.0
ang = np.linspace(0, 2 * np.pi, len(rcats), endpoint=False).tolist(); ang += ang[:1]
for j, tr in enumerate(TRIALS):
    vals = R[:, j].tolist(); vals += vals[:1]
    lw = 2.6 if tr == 'POP-RT' else 1.8
    ax_b.plot(ang, vals, 'o-', color=COLORS[j], linewidth=lw, markersize=5)
    ax_b.fill(ang, vals, color=COLORS[j], alpha=0.05)
ax_b.set_xticks(ang[:-1]); ax_b.set_xticklabels(rcats, fontsize=9.5)
ax_b.tick_params(axis='x', pad=14); ax_b.set_ylim(0, 100)
ax_b.set_yticklabels([]); ax_b.set_rlabel_position(90)
ax_b.set_title('b   Key modifiers, normalized per axis (complete-data axes only)',
               fontsize=11.5, fontweight='bold', loc='left', pad=26)

# ── (c) atypicality (over reported cells) ──
med = np.nanmedian(Mn, axis=1, keepdims=True)
atyp = np.nanmean(np.abs(Mn - med), axis=0)
order = np.argsort(atyp)
ax_c.barh([TRIALS[k] for k in order], [atyp[k] for k in order],
          color=[COLORS[k] for k in order], alpha=0.9, edgecolor='black', linewidth=0.6)
for rank, k in enumerate(order):
    ax_c.text(atyp[k] + 0.005, rank, f'{atyp[k]:.2f}', va='center', fontsize=10,
              fontweight='bold', color=COLORS[k])
ax_c.set_xlabel('Mean |deviation from cross-trial median| over reported cells\n'
                '(higher = breaks comparability more)', fontsize=10)
ax_c.set_xlim(0, max(atyp) * 1.18)
ax_c.set_title('c   Trial atypicality score', fontsize=11.5, fontweight='bold', loc='left', pad=22)
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.xaxis.grid(True, alpha=0.25, linestyle='--'); ax_c.set_axisbelow(True)

# title + shared legend + footnote
fig.suptitle('Cross-trial comparability (transitivity) dashboard — WPRT vs PORT in localized prostate cancer',
             fontsize=15, fontweight='bold', y=0.985)
fig.legend(handles=[Line2D([0], [0], color=COLORS[j], marker='o', lw=2.5, markersize=6, label=TRIALS[j])
                    for j in range(len(TRIALS))],
           loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=5, frameon=False, fontsize=11)
fig.text(0.13, 0.012,
         'POP-RT is the consistent outlier (single-centre, PSMA staging, highest pelvic EQD2, common-iliac '
         'coverage, 24-mo ADT). Effect-modifier axes are collinear → trial-level meta-regression is '
         'exploratory only.\n* estimate for visualization;  † not located in primary source (SEOR deck cited '
         '24.5%);  ‡ observed median (others prescribed);  n.r. = not reported / not derivable.\n'
         'PEACE-2 Gleason reported only as 8-10 (75–79%); RTOG 0924 median PSA not derivable from slides; '
         'PSMA staging counts PSMA-PET only (PEACE-2 ~18% imaging was mostly choline).',
         fontsize=8, style='italic', color='#555555', va='bottom')

fig.savefig(os.path.join(OUT, 'transitivity_dashboard.png'), dpi=DPI, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'transitivity_dashboard.pdf'), bbox_inches='tight')
plt.close(fig)

print('Saved: analysis/figures/transitivity_dashboard.{png,pdf}')
print('\nAtypicality (over reported cells):')
for k in np.argsort(-atyp):
    print(f'  {TRIALS[k]:<12} {atyp[k]:.3f}')
print(f'\nRadar axes used (complete data): {rcats}')
