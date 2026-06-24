#!/usr/bin/env python3
"""Visual QC of the KM reconstruction: reconstructed Kaplan-Meier vs digitized input.

For every reconstructed endpoint it overlays:
  - the Kaplan-Meier of the reconstructed pseudo-IPD (step line, per arm), and
  - the digitized (time, survival) points that were the input (dots),
so you can see the reconstruction reproduces the published curve shape, not just
the hazard ratio. One panel per trial-endpoint.

Output: figures_ipd/km_reconstruction_overlay.{png,pdf}
"""
import os

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DIG = os.path.join(HERE, 'digitized')
REC = os.path.join(HERE, 'reconstructed')
OUT = os.path.join(HERE, '..', 'figures_ipd')
os.makedirs(OUT, exist_ok=True)

from trials_ipd import TRIALS_IPD

C_WPRT = '#1F4E79'; C_PORT = '#C0392B'

panels = []
for trial, tcfg in TRIALS_IPD.items():
    for ep, ecfg in tcfg['endpoints'].items():
        panels.append((trial, ep, ecfg, tcfg['source']))

ncol = 4
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)

for idx, (trial, ep, ecfg, source) in enumerate(panels):
    ax = axes[idx // ncol][idx % ncol]
    ipd = pd.read_csv(os.path.join(REC, f'{trial.replace(" ", "")}_{ep}_ipd.csv'))
    for arm, color, name in [(1, C_WPRT, 'WPRT'), (0, C_PORT, 'PORT')]:
        sub = ipd[ipd.arm == arm]
        kmf = KaplanMeierFitter().fit(sub.time, sub.event)
        sf = kmf.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where='post', color=color, lw=1.6, zorder=2,
                label=f'{name} (recon)')
        dig = pd.read_csv(os.path.join(DIG, ecfg['arms'][name]['coords']))
        ax.scatter(dig.time, dig.surv, s=18, color=color, edgecolor='white', lw=0.5,
                   zorder=4, label=f'{name} (digitized)')
    phr = ecfg['published_hr']
    ax.text(0.04, 0.06, f'pub HR {phr[0]:.2f}', transform=ax.transAxes, fontsize=7.5, color='#444')
    ax.set_title(f'{trial} — {ep}  [{source}]', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.02); ax.set_xlim(0, ipd.time.max() * 1.02)
    ax.set_xlabel('Months', fontsize=8); ax.set_ylabel('Survival', fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.spines[['top', 'right']].set_visible(False)
    if idx == 0:
        ax.legend(fontsize=6.2, loc='upper right', frameon=False)

for j in range(len(panels), nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')

fig.suptitle('KM reconstruction QC — reconstructed pseudo-IPD (lines) vs digitized input (dots). '
             'WPRT blue, PORT red.', fontsize=11, fontweight='bold', y=0.998)
fig.tight_layout(rect=(0, 0, 1, 0.985))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'km_reconstruction_overlay.{ext}'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved km_reconstruction_overlay.{{png,pdf}} ({len(panels)} panels) in {OUT}')
