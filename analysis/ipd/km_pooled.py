#!/usr/bin/env python3
"""Pooled reconstructed Kaplan-Meier (WPRT vs PORT), one panel per endpoint.

Stacks the validated pseudo-IPD across all reconstructable trials (source format
-- full text vs congress slide -- is NOT used to segregate) and plots a combined
KM per arm. The pooled hazard ratio annotated is the formal two-stage random-
effects estimate (DL + Hartung-Knapp); the curve itself is a naive-pooled
visualisation, so for OS it is dominated by RTOG 0924 (N=2473) -- noted on-figure.

Output: figures_ipd/km_pooled_reconstructed.{png,pdf}
"""
import os

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, 'reconstructed')
OUT = os.path.join(HERE, '..', 'figures_ipd')

from pool_ipd import GROUPS, cox_loghr, re_pool

C_WPRT = '#1F4E79'; C_PORT = '#C0392B'


def pooled_hr(trials):
    ys, ss = [], []
    for _, fn, _ in trials:
        y, s = cox_loghr(pd.read_csv(os.path.join(REC, fn)))
        ys.append(y); ss.append(s)
    p = re_pool(ys, ss)
    return np.exp(p['est']), np.exp(p['lo']), np.exp(p['hi']), p['I2'], p['k']


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.16, wspace=0.22)

for ax, letter, (ep, trials) in zip(axes, ['a', 'b', 'c'], GROUPS.items()):
    frames = []
    for label, fn, source in trials:
        d = pd.read_csv(os.path.join(REC, fn)); d['trial'] = label
        frames.append(d)
    comb = pd.concat(frames, ignore_index=True)
    n_trials = comb['trial'].nunique()
    n_pts = len(comb)
    for arm, color, name in [(1, C_WPRT, 'Pelvic RT (WPRT)'), (0, C_PORT, 'Prostate-only RT (PORT)')]:
        sub = comb[comb.arm == arm]
        kmf = KaplanMeierFitter().fit(sub.time, sub.event, label=name)
        # Smooth visual: linearly-interpolated KM points (preserves all jump points,
        # avoids the staircase look of step plots). 95% CI band shown shaded.
        sf = kmf.survival_function_.reset_index(); sf.columns = ['t', 's']
        ci = kmf.confidence_interval_survival_function_.reset_index()
        ci.columns = ['t', 'lo', 'hi']
        ax.fill_between(ci['t'].values, ci['lo'].values, ci['hi'].values,
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(sf['t'].values, sf['s'].values, color=color, lw=1.8, label=name,
                solid_capstyle='round', solid_joinstyle='round')
    hr, lo, hi, I2, k = pooled_hr(trials)
    ax.text(0.03, 0.10, f'Pooled HR {hr:.2f} ({lo:.2f}–{hi:.2f})\nI²={I2:.0f}%  ·  {k} trials, {n_pts:,} pts',
            transform=ax.transAxes, fontsize=8.5, color='#111',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb', lw=0.6))
    ax.set_title(f'{letter}  {ep}', loc='left', fontsize=11, fontweight='bold')
    ax.set_xlabel('Months from randomization', fontsize=9)
    ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, comb.time.quantile(0.99))
    ax.legend(fontsize=7.5, loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Pooled reconstructed Kaplan-Meier — whole-pelvic vs prostate-only radiotherapy',
             x=0.5, y=0.95, fontsize=13, fontweight='bold')
fig.text(0.07, 0.015,
         'Naive-pooled pseudo-IPD across all reconstructable trials (POP-RT, GETUG-01, PEACE-2; '
         'RTOG 0924 adds OS). Annotated HR = two-stage random-effects (DL+Hartung-Knapp). '
         'OS curve dominated by RTOG 0924 (N=2473). RTOG 9413 not reconstructable. HR<1 favours WPRT.',
         ha='left', fontsize=6.8, color='#666')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'km_pooled_reconstructed.{ext}'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved km_pooled_reconstructed.{{png,pdf}} in {OUT}')
