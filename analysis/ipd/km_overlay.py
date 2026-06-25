#!/usr/bin/env python3
"""Visual QC of the KM reconstruction: smoothed reconstructed KM vs digitised input.

For every reconstructed endpoint it overlays:
  - the smoothed Kaplan-Meier of the reconstructed pseudo-IPD (PCHIP + Savitzky-
    Golay; monotonic-decreasing; truncated where < 10 % at risk), per arm; and
  - the digitised (time, survival) points that were the input (dots),
so the reader can see the reconstruction reproduces the published curve shape,
not only the hazard ratio. One panel per trial-endpoint.

Smoothing recipe matches `km_pooled.py` (ported from EV-P vs MVAC vs GC-Durva,
manuscript/Fig_slide.py).

Output: figures_ipd/km_reconstruction_overlay.{png,pdf}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DIG = os.path.join(HERE, 'digitized')
REC = os.path.join(HERE, 'reconstructed')
OUT = os.path.join(HERE, '..', 'figures_ipd')
os.makedirs(OUT, exist_ok=True)

from trials_ipd import TRIALS_IPD

# ── Palette (TaHG-inspired) + Nature-style rcParams ────────────────────────
C_WPRT = '#4C72B0'   # muted blue — pelvic (WPRT)
C_PORT = '#C44E52'   # muted maroon — prostate-only (PORT)
GRID_COLOR = '#E8E8E8'
TEXT_DARK = '#1A1A1A'
TEXT_MUTED = '#5A5A5A'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.7,
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def _truncate_reliable(t, s, df, frac=0.10):
    """Truncate KM at the time where < `frac` of the cohort remains at risk."""
    n_total = len(df)
    if n_total == 0 or t.size == 0:
        return t, s
    t_max_ok = 0.0
    for tc in np.linspace(0, t[-1], 200):
        if (df['time'] >= tc).sum() >= n_total * frac:
            t_max_ok = tc
    if t_max_ok > 0 and t_max_ok < t[-1]:
        m = t <= t_max_ok
        t, s = t[m], s[m]
    return t, s


def _smooth(t, s, n_points=600):
    """PCHIP + Savitzky-Golay with monotonic-decreasing constraint."""
    if t.size < 3:
        return t, s
    t_max = t[-1]
    n_anchors = max(20, int(t_max / 2))
    t_anc = np.linspace(0, t_max, n_anchors)
    s_anc = np.zeros(n_anchors)
    for i, ta in enumerate(t_anc):
        idx = max(0, min(np.searchsorted(t, ta, side='right') - 1, len(s) - 1))
        s_anc[i] = s[idx]
    t_f = np.linspace(0, t_max, n_points)
    s_f = PchipInterpolator(t_anc, s_anc, extrapolate=False)(t_f)
    win = min(51, len(s_f) // 4)
    if win % 2 == 0:
        win += 1
    if win >= 5:
        s_f = savgol_filter(s_f, win, 3)
    s_f = np.clip(s_f, 0, 1)
    s_f[0] = 1.0
    for i in range(1, len(s_f)):
        if s_f[i] > s_f[i - 1]:
            s_f[i] = s_f[i - 1]
    return t_f, s_f


# ── Build panels list ──────────────────────────────────────────────────────
panels = []
for trial, tcfg in TRIALS_IPD.items():
    for ep, ecfg in tcfg['endpoints'].items():
        panels.append((trial, ep, ecfg, tcfg['source']))

ncol = 4
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow), squeeze=False)

for idx, (trial, ep, ecfg, source) in enumerate(panels):
    ax = axes[idx // ncol][idx % ncol]
    ipd = pd.read_csv(os.path.join(REC, f'{trial.replace(" ", "")}_{ep}_ipd.csv'))
    panel_max = 0.0

    for arm, color, name in [(1, C_WPRT, 'WPRT'), (0, C_PORT, 'PORT')]:
        sub = ipd[ipd.arm == arm]
        kmf = KaplanMeierFitter().fit(sub.time, sub.event)
        sf = kmf.survival_function_
        t = sf.index.values.astype(float)
        s = sf.values[:, 0].astype(float)
        t, s = _truncate_reliable(t, s, sub, frac=0.10)
        t_s, s_s = _smooth(t, s)
        ax.plot(t_s, s_s, color=color, lw=1.7, zorder=2,
                label=f'{name} (reconstructed)',
                solid_capstyle='round', solid_joinstyle='round')
        panel_max = max(panel_max, t_s[-1] if t_s.size else 0.0)

        dig = pd.read_csv(os.path.join(DIG, ecfg['arms'][name]['coords']))
        ax.scatter(dig.time, dig.surv, s=15, facecolor='white',
                   edgecolor=color, lw=0.9, zorder=4,
                   label=f'{name} (digitised)')

    phr = ecfg['published_hr']
    ax.text(0.04, 0.06, f'pub HR {phr[0]:.2f}',
            transform=ax.transAxes, fontsize=7.6, color=TEXT_DARK,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.5))
    ax.set_title(f'{trial} — {ep}  [{source}]',
                 fontsize=9.5, fontweight='bold', color=TEXT_DARK, pad=4)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, panel_max * 1.02 if panel_max else 1)
    ax.set_xlabel('Months from randomisation', fontsize=8.5, color=TEXT_DARK)
    ax.set_ylabel('Survival probability', fontsize=8.5, color=TEXT_DARK)
    ax.tick_params(labelsize=7.8, color='#666666', length=2.5)
    ax.grid(True, axis='both', color=GRID_COLOR, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if idx == 0:
        ax.legend(fontsize=6.6, loc='upper right', frameon=False,
                  handletextpad=0.4, labelspacing=0.3)

# Hide unused panels
for j in range(len(panels), nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')

fig.suptitle(
    'Reconstruction QC — smoothed reconstructed Kaplan-Meier (lines) versus '
    'digitised input (open dots). WPRT blue · PORT maroon.',
    fontsize=11, fontweight='bold', y=0.998, color=TEXT_DARK)
fig.tight_layout(rect=(0, 0, 1, 0.982))

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'km_reconstruction_overlay.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved km_reconstruction_overlay.{{png,pdf}} ({len(panels)} panels) in {OUT}')
