#!/usr/bin/env python3
"""Pooled reconstructed Kaplan-Meier (WPRT vs PORT) — smoothed, extended layout.

Vertical 3-panel figure (one panel per endpoint). Each panel shows the smoothed
Kaplan-Meier survival curves for WPRT (blue) and PORT (maroon), with 95% pointwise
confidence band, the pooled random-effects hazard ratio annotated, and a
numbers-at-risk table below.

Smoothing recipe (ported from EV-P vs MVAC vs GC-Durva project,
manuscript/Fig_slide.py): step KM is anchored on a regular grid, interpolated
with PchipInterpolator on a fine grid (800 points), then passed through a
Savitzky-Golay filter (order 3) and constrained to be monotonically decreasing
and bounded in [0, 1]. Curves are truncated at the time where at least 10% of
the original pseudo-IPD cohort remains at risk, to avoid the unreliable tail
of the reconstructed estimate.

Aesthetics: Nature-style (Arial sans-serif, no top/right spines, light-grey
grid, muted blue/maroon palette).

Output: figures_ipd/km_pooled_reconstructed.{png,pdf}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, 'reconstructed')
OUT = os.path.join(HERE, '..', 'figures_ipd')

from pool_ipd import GROUPS, cox_loghr, re_pool

# ── Palette (TaHG-inspired) ────────────────────────────────────────────────
C_WPRT = '#4C72B0'   # muted blue  — pelvic (WPRT)
C_PORT = '#FF7F0E'   # tab:orange — prostate-only (PORT)
GRID_COLOR = '#E8E8E8'
FILL_ALPHA = 0.12
TEXT_DARK = '#1A1A1A'
TEXT_MUTED = '#5A5A5A'

# ── Nature-style rcParams ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def _km_arrays(times, events):
    """Fit KM with lifelines and return (t, s, lo, hi) arrays."""
    kmf = KaplanMeierFitter().fit(times, events)
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    t = sf.index.values.astype(float)
    s = sf.values[:, 0].astype(float)
    lo = ci.values[:, 0].astype(float)
    hi = ci.values[:, 1].astype(float)
    return t, s, lo, hi


def _truncate_reliable(t, s, lo, hi, df, frac=0.10):
    """Truncate KM at the time where < `frac` of the cohort remains at risk."""
    n_total = len(df)
    if n_total == 0 or t.size == 0:
        return t, s, lo, hi
    t_max_ok = 0.0
    for tc in np.linspace(0, t[-1], 200):
        if (df['time'] >= tc).sum() >= n_total * frac:
            t_max_ok = tc
    if t_max_ok > 0 and t_max_ok < t[-1]:
        m = t <= t_max_ok
        t, s, lo, hi = t[m], s[m], lo[m], hi[m]
    return t, s, lo, hi


def _smooth(t, s, n_points=800):
    """PCHIP + Savitzky-Golay smoothing with monotonic-decreasing constraint."""
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


def _at_risk(df, tick_times):
    return [int((df['time'] >= t).sum()) for t in tick_times]


def _xticks(t_max):
    """Whole-number axis ticks at sensible intervals."""
    step = 12 if t_max <= 96 else 24 if t_max <= 180 else 36
    n_steps = int(np.ceil(t_max / step))
    return [step * i for i in range(n_steps + 1) if step * i <= t_max]


def pooled_hr(trials):
    ys, ss = [], []
    for _, fn, _ in trials:
        y, s = cox_loghr(pd.read_csv(os.path.join(REC, fn)))
        ys.append(y); ss.append(s)
    p = re_pool(ys, ss)
    return np.exp(p['est']), np.exp(p['lo']), np.exp(p['hi']), p['I2'], p['k']


# ── Build figure ────────────────────────────────────────────────────────────
# Squarish main panels: width ≈ height of plotting area, with a thin
# at-risk strip below each one.
n_panels = len(GROUPS)
fig = plt.figure(figsize=(7.2, 3.6 * n_panels), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=n_panels * 2,
    ncols=1,
    height_ratios=[5, 1.6] * n_panels,
    hspace=0.65,
    left=0.14, right=0.985, top=0.99, bottom=0.05,
)

for i, (ep, trials) in enumerate(GROUPS.items()):
    ax = fig.add_subplot(gs[i * 2, 0])
    ax_risk = fig.add_subplot(gs[i * 2 + 1, 0], sharex=ax)

    # Stacked pseudo-IPD across all reconstructable trials for this endpoint
    frames = []
    for label, fn, _src in trials:
        d = pd.read_csv(os.path.join(REC, fn))
        d['trial'] = label
        frames.append(d)
    comb = pd.concat(frames, ignore_index=True)
    n_pts = len(comb)
    n_trials = comb['trial'].nunique()

    # Plot per arm
    panel_max = 0.0
    risk_rows = {}
    for arm, color, name in [
        (1, C_WPRT, 'Pelvic RT (WPRT)'),
        (0, C_PORT, 'Prostate-only RT (PORT)'),
    ]:
        sub = comb[comb.arm == arm]
        t, s, lo, hi = _km_arrays(sub.time, sub.event)
        t, s, lo, hi = _truncate_reliable(t, s, lo, hi, sub, frac=0.10)
        t_s, s_s = _smooth(t, s)
        _, lo_s = _smooth(t, lo)
        _, hi_s = _smooth(t, hi)
        ax.fill_between(t_s, lo_s, hi_s, color=color, alpha=FILL_ALPHA, lw=0)
        ax.plot(t_s, s_s, color=color, lw=2.0, label=f'{name}  (n={len(sub):,})',
                solid_capstyle='round', solid_joinstyle='round')
        panel_max = max(panel_max, t_s[-1] if t_s.size else 0.0)
        risk_rows[name] = sub

    # Pooled HR annotation — lower-left of the panel (curves descend from
    # upper-left, leaving lower-left empty)
    hr, lo_hr, hi_hr, I2, k = pooled_hr(trials)
    ax.text(0.025, 0.06,
            f'Pooled HR  {hr:.2f}  ({lo_hr:.2f}–{hi_hr:.2f})\n'
            f'I² = {I2:.0f}%   ·   {k} trials,  {n_pts:,} patients',
            transform=ax.transAxes, fontsize=8.5, color=TEXT_DARK,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', lw=0.6))

    # Cosmetic axes
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, panel_max)
    ax.grid(True, axis='both', color=GRID_COLOR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ep, fontsize=10.5, fontweight='bold', color=TEXT_DARK)
    ax.text(-0.105, 1.05, chr(97 + i), transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=TEXT_DARK, va='top', ha='left')
    # Legend at lower-right corner (HR box occupies lower-left)
    ax.legend(loc='lower right', bbox_to_anchor=(0.985, 0.04),
              frameon=False, fontsize=8.5, handlelength=1.6,
              ncol=1, labelspacing=0.4)
    ax.tick_params(axis='both', labelsize=9, color='#666666', length=3)

    # At-risk strip
    ticks = _xticks(panel_max)
    ax_risk.set_xlim(0, panel_max)
    ax_risk.set_yticks([])
    for spine in ('top', 'right', 'left', 'bottom'):
        ax_risk.spines[spine].set_visible(False)
    ax_risk.tick_params(axis='x', length=0, labelbottom=False)

    ax_risk.text(-0.03, 1.05, 'No. at risk',
                 transform=ax_risk.transAxes, ha='right', va='bottom',
                 fontsize=8.5, color=TEXT_MUTED)
    for j, (name, color) in enumerate([('Pelvic RT (WPRT)', C_WPRT),
                                       ('Prostate-only RT (PORT)', C_PORT)]):
        sub = risk_rows[name]
        n_at_risk = _at_risk(sub, ticks)
        y_pos = 0.95 - j * 0.42
        ax_risk.text(-0.03, y_pos, name.split('(')[1].rstrip(')'),
                     transform=ax_risk.transAxes, ha='right', va='center',
                     fontsize=8.5, color=color, fontweight='bold')
        for tk, nk in zip(ticks, n_at_risk):
            ax_risk.text(tk, y_pos, f'{nk:,}',
                         ha='center', va='center', fontsize=8.5,
                         color=TEXT_DARK,
                         transform=ax_risk.get_xaxis_transform())

    # Month axis on the KM panel (above the at-risk table)
    ax.set_xticks(ticks)
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.tick_params(axis='x', labelsize=9, colors=TEXT_DARK, length=3)
    ax.set_xlabel('Months from randomisation', fontsize=10, color=TEXT_DARK, labelpad=4)

fig.savefig(os.path.join(OUT, 'km_pooled_reconstructed.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved km_pooled_reconstructed.png in {OUT}')
