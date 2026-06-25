#!/usr/bin/env python3
"""Pre-specified sensitivity analyses for the WPRT-vs-PORT pairwise meta-analysis.

Reuses the single source of truth for the extracted per-trial hazard ratios
(`pairwise_ma.py`) so no estimate is re-entered or invented here. For each pooled
endpoint (OS, biochemical/progression, distant metastasis/MFS) it runs:

  1. Primary           — all available trials, DerSimonian-Laird τ² + modified HK CI
  2. Published-only    — drop the congress-slide-only trials (PEACE-2, RTOG 0924)
  3. Exclude POP-RT    — drop the heterogeneity driver / positive outlier
  4. Fixed-effect      — inverse-variance, τ²=0
  5. τ² = Paule-Mandel — alternative heterogeneity estimator + HK CI
  6. τ² = REML         — alternative heterogeneity estimator + HK CI
  7. Classical Wald CI — DL τ² but no Hartung-Knapp (shows how much HK widens)

plus leave-one-out (drop each trial in turn).

Outputs (in analysis/figures/):
  sensitivity_summary.csv
  forest_sensitivity_robustness.{png,pdf}
  forest_sensitivity_leaveoneout.{png,pdf}

Note on validity (k=4-5): no funnel/Egger/meta-regression — invalid at this k.
These are robustness checks, not confirmatory subgroup tests. The published-only
analysis is the key one: PEACE-2 and RTOG 0924 are not yet peer-reviewed (data from
congress slides), so the pooled effect must be shown to hold without them.
"""
import csv
import os

import numpy as np
from scipy.stats import t as student_t, norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import blended_transform_factory

from pairwise_ma import TRIALS, ENDPOINTS, _yv, OUT

DPI = 300
SLIDE_ONLY = {'PEACE-2', 'RTOG 0924'}   # not yet peer-reviewed (congress slides)
C_POOL = '#1F4E79'; C_ALT = '#7F8C8D'; C_KEY = '#C0392B'; C_NULL = '#9AA0A6'
XLIM = (0.05, 90); EFFECT_MAX = 6.0; SEP_X = 7.5; TXT_X = 10.0
XTICKS = [0.1, 0.25, 0.5, 1, 2, 4]


# ── τ² estimators ──────────────────────────────────────────────────────────
def _tau2_DL(y, v):
    k = len(y); wfe = 1.0 / v; ybar = (wfe * y).sum() / wfe.sum()
    Q = (wfe * (y - ybar) ** 2).sum()
    C = wfe.sum() - (wfe ** 2).sum() / wfe.sum()
    return max((Q - (k - 1)) / C, 0.0) if C > 0 else 0.0


def _tau2_PM(y, v):
    k = len(y)
    def g(t2):
        w = 1.0 / (v + t2); ybar = (w * y).sum() / w.sum()
        return (w * (y - ybar) ** 2).sum() - (k - 1)
    if g(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 10.0
    while g(hi) > 0 and hi < 1e6:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _tau2_REML(y, v):
    # Maximise the profiled restricted log-likelihood over tau2>=0 (robust at small k;
    # the naive fixed-point iteration can get clamped at the 0 boundary).
    def negll(t2):
        w = 1.0 / (v + t2); sw = w.sum(); ybar = (w * y).sum() / sw
        return 0.5 * (np.log(v + t2).sum() + np.log(sw) + (w * (y - ybar) ** 2).sum())
    res = minimize_scalar(negll, bounds=(0.0, 10.0), method='bounded')
    return max(float(res.x), 0.0)


def pool(triples, tau2_method='DL', ci='hk'):
    y = np.array([t[0] for t in triples]); v = np.array([t[1] for t in triples]); k = len(y)
    wfe = 1.0 / v; ybar_fe = (wfe * y).sum() / wfe.sum()
    Q = (wfe * (y - ybar_fe) ** 2).sum()
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    tau2 = {'FE': 0.0, 'DL': None, 'PM': None, 'REML': None}[tau2_method]
    if tau2 is None:
        tau2 = {'DL': _tau2_DL, 'PM': _tau2_PM, 'REML': _tau2_REML}[tau2_method](y, v)
    w = 1.0 / (v + tau2); sw = w.sum(); ybar = (w * y).sum() / sw
    se_normal = np.sqrt(1.0 / sw)
    if ci in ('fixed', 'wald'):
        se, crit = se_normal, norm.ppf(0.975)
    else:  # modified Hartung-Knapp (SE floored at classical RE SE), t_{k-1}
        se_hk = np.sqrt((w * (y - ybar) ** 2).sum() / ((k - 1) * sw))
        se, crit = max(se_hk, se_normal), student_t.ppf(0.975, k - 1)
    # 95% prediction interval (Higgins-Thompson-Spiegelhalter; t_{k-2}); random-effects only
    if ci not in ('fixed', 'wald') and k >= 3:
        tpi = student_t.ppf(0.975, k - 2)
        pi_lo = np.exp(ybar - tpi * np.sqrt(tau2 + se ** 2))
        pi_hi = np.exp(ybar + tpi * np.sqrt(tau2 + se ** 2))
    else:
        pi_lo = pi_hi = np.nan
    return dict(k=k, hr=np.exp(ybar), lo=np.exp(ybar - crit * se),
                hi=np.exp(ybar + crit * se), I2=I2, tau2=tau2, pi_lo=pi_lo, pi_hi=pi_hi)


# ── analysis assembly ───────────────────────────────────────────────────────
def endpoint_rows(ep):
    data = ENDPOINTS[ep]['data']
    return [(tr, data[tr]) for tr in TRIALS if data[tr]]


def run_endpoint(ep):
    rows = endpoint_rows(ep)
    trials = [tr for tr, _ in rows]
    triples = [_yv(d) for _, d in rows]
    out = [('Primary (all; DL + mod-HK)', pool(triples, 'DL', 'hk'), trials, False)]

    pub = [(tr, d) for tr, d in rows if tr not in SLIDE_ONLY]
    if len(pub) >= 2:
        out.append(('Full-text trials only',
                    pool([_yv(d) for _, d in pub], 'DL', 'hk'), [t for t, _ in pub], True))
    noprt = [(tr, d) for tr, d in rows if tr != 'POP-RT']
    if len(noprt) >= 2:
        out.append(('Exclude POP-RT (outlier)',
                    pool([_yv(d) for _, d in noprt], 'DL', 'hk'), [t for t, _ in noprt], False))
    out.append(('Fixed-effect (inverse-variance)', pool(triples, 'FE', 'fixed'), trials, False))
    out.append(('Random-effects, Paule-Mandel τ²', pool(triples, 'PM', 'hk'), trials, False))
    out.append(('Random-effects, REML τ²', pool(triples, 'REML', 'hk'), trials, False))
    out.append(('DL τ², classical Wald CI (no HK)', pool(triples, 'DL', 'wald'), trials, False))
    return out


def leave_one_out(ep):
    rows = endpoint_rows(ep)
    res = []
    for i, (tr, _) in enumerate(rows):
        sub = [_yv(d) for j, (t, d) in enumerate(rows) if j != i]
        if len(sub) >= 2:
            res.append((f'Omit {tr}', pool(sub, 'DL', 'hk')))
    return res


# ── console + CSV ────────────────────────────────────────────────────────────
def console_and_csv():
    rows_csv = [['endpoint', 'analysis', 'k', 'HR', 'CI_low', 'CI_high', 'I2_pct', 'tau2',
                 'PI_low', 'PI_high']]
    print('=' * 84)
    print('SENSITIVITY ANALYSES — WPRT vs PORT. HR<1 favours pelvic RT (WPRT).')
    print('=' * 84)
    for ep in ENDPOINTS:
        print(f'\n### {ep}  ({ENDPOINTS[ep]["note"] or "single defn"})')
        for label, r, trials, key in run_endpoint(ep):
            flag = '  <-- key' if key else ''
            pi = (f'  PI {r["pi_lo"]:.2f}-{r["pi_hi"]:.2f}'
                  if not np.isnan(r['pi_lo']) else '')
            print(f'  {label:<38} k={r["k"]}  HR {r["hr"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f})'
                  f'  I2={r["I2"]:.0f}%  tau2={r["tau2"]:.3f}{pi}{flag}')
            rows_csv.append([ep, label, r['k'], f'{r["hr"]:.3f}', f'{r["lo"]:.3f}',
                             f'{r["hi"]:.3f}', f'{r["I2"]:.0f}', f'{r["tau2"]:.4f}',
                             '' if np.isnan(r['pi_lo']) else f'{r["pi_lo"]:.3f}',
                             '' if np.isnan(r['pi_hi']) else f'{r["pi_hi"]:.3f}'])
        print('  -- leave-one-out --')
        for label, r in leave_one_out(ep):
            print(f'  {label:<38} k={r["k"]}  HR {r["hr"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f})  I2={r["I2"]:.0f}%')
            rows_csv.append([ep, 'LOO: ' + label, r['k'], f'{r["hr"]:.3f}', f'{r["lo"]:.3f}',
                             f'{r["hi"]:.3f}', f'{r["I2"]:.0f}', f'{r["tau2"]:.4f}',
                             '' if np.isnan(r['pi_lo']) else f'{r["pi_lo"]:.3f}',
                             '' if np.isnan(r['pi_hi']) else f'{r["pi_hi"]:.3f}'])
    path = os.path.join(OUT, 'sensitivity_summary.csv')
    with open(path, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows_csv)
    print(f'\nSaved: {path}')


# ── forest plots ──────────────────────────────────────────────────────────────
def _clamp(v):
    return min(max(v, XLIM[0]), EFFECT_MAX)


def _arrows(ax, lo, hi, y, color):
    if hi > EFFECT_MAX:
        ax.annotate('', xy=(EFFECT_MAX, y), xytext=(EFFECT_MAX / 1.7, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))
    if lo < XLIM[0]:
        ax.annotate('', xy=(XLIM[0], y), xytext=(XLIM[0] * 1.7, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))


def _diamond(ax, c, lo, hi, y, color, h=0.30):
    ax.add_patch(Polygon([[max(lo, XLIM[0]), y], [_clamp(c), y + h],
                          [min(hi, EFFECT_MAX), y], [_clamp(c), y - h]],
                         closed=True, facecolor=color, edgecolor='black', lw=0.7, zorder=5))
    _arrows(ax, lo, hi, y, color)


def _forest_panel(ax, items, title, letter):
    ax.axvline(1, color=C_NULL, ls='--', lw=0.9, zorder=1)
    ax.axvline(SEP_X, color='#e6e6e6', lw=0.8, zorder=1)
    ys = list(range(len(items) - 1, -1, -1))
    ax.text(TXT_X, ys[0] + 0.85, 'HR (95% CI)', ha='left', va='center',
            fontsize=7, style='italic', color='#888')
    for (label, r, color), y in zip(items, ys):
        _diamond(ax, r['hr'], r['lo'], r['hi'], y, color, h=0.28)
        ax.text(TXT_X, y, f'{r["hr"]:.2f} ({r["lo"]:.2f}–{r["hi"]:.2f})', ha='left', va='center',
                fontsize=7.4, color=color, fontweight='bold')
    ax.set_yticks(ys); ax.set_yticklabels([it[0] for it in items], fontsize=8)
    for lab, it in zip(ax.get_yticklabels(), items):
        lab.set_color(it[2])
    ax.set_xscale('log'); ax.set_xlim(*XLIM); ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_xticks(XTICKS); ax.set_xticklabels([str(x) for x in XTICKS], fontsize=8)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    td = blended_transform_factory(ax.transData, ax.transAxes)
    y_arr = -0.20; y_txt = -0.27
    ax.annotate('', xy=(0.07, y_arr), xytext=(0.9, y_arr), xycoords=td, textcoords=td,
                annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
    ax.annotate('', xy=(EFFECT_MAX, y_arr), xytext=(1.12, y_arr), xycoords=td, textcoords=td,
                annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
    ax.text(0.25, y_txt, 'favours WPRT', transform=td, ha='center', va='top',
            fontsize=7, color='#555', clip_on=False)
    ax.text(2.4, y_txt, 'favours PORT', transform=td, ha='center', va='top',
            fontsize=7, color='#555', clip_on=False)
    ax.set_title(f'{letter}   {title}', loc='left', fontsize=10, fontweight='bold', pad=10)


def plot_robustness():
    eps = list(ENDPOINTS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.82, bottom=0.24, wspace=0.55)
    for ax, letter, ep in zip(axes, ['a', 'b', 'c'], eps):
        items = []
        for label, r, trials, key in run_endpoint(ep):
            color = C_KEY if key else (C_POOL if label.startswith('Primary') else C_ALT)
            items.append((label, r, color))
        _forest_panel(ax, items, ep, letter)
    fig.suptitle('Sensitivity analyses — pelvic vs prostate-only radiotherapy '
                 '(pairwise random-effects MA)', x=0.5, y=0.96, fontsize=12.5, fontweight='bold')
    fig.text(0.15, 0.012,
             'Hazard ratio (95% CI), log scale. Key sensitivity (red) = full-text trials only, '
             'i.e. excluding trials available only as congress presentations (PEACE-2 ASCO GU / '
             'ESTRO 2026 and RTOG 0924 ASTRO 2025). '
             'No funnel/Egger/meta-regression (invalid at k=4–5); robustness checks, not confirmatory. '
             'Arrowhead = CI beyond axis. HR<1 favours pelvic RT (WPRT).',
             ha='left', fontsize=7.0, color='#666', va='bottom')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'forest_sensitivity_robustness.{ext}'),
                    dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_leaveoneout():
    eps = list(ENDPOINTS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.28, wspace=0.50)
    for ax, letter, ep in zip(axes, ['a', 'b', 'c'], eps):
        prim = pool([_yv(d) for _, d in endpoint_rows(ep)], 'DL', 'hk')
        items = [('Primary (all)', prim, C_POOL)]
        for label, r in leave_one_out(ep):
            items.append((label, r, '#3A3A3A'))
        _forest_panel(ax, items, ep, letter)
    fig.suptitle('Leave-one-out sensitivity — pelvic vs prostate-only radiotherapy',
                 x=0.5, y=0.96, fontsize=12.5, fontweight='bold')
    fig.text(0.12, 0.012, 'Hazard ratio (95% CI), log scale. Each row = pooled HR after omitting one trial '
             '(DL + modified Hartung-Knapp). Arrowhead = CI beyond axis. HR<1 favours pelvic RT (WPRT).',
             ha='left', fontsize=7.0, color='#666', va='bottom')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'forest_sensitivity_leaveoneout.{ext}'),
                    dpi=DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    console_and_csv()
    plot_robustness()
    plot_leaveoneout()
    print('Saved: forest_sensitivity_robustness.{png,pdf} and forest_sensitivity_leaveoneout.{png,pdf}')
