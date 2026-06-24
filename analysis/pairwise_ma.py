#!/usr/bin/env python3
"""
Pairwise random-effects meta-analysis — WPRT vs PORT in localized prostate cancer
=================================================================================
Aggregate-data synthesis of trial-reported hazard ratios (HR<1 favours WPRT/pelvic RT).
Five RCTs (extends Roy 2026, who pooled only the first four — PEACE-2 is added here).

Method (improves on Roy's DerSimonian-Laird + normal CI):
  - between-study variance tau^2 by DerSimonian-Laird (closed form)
  - pooled CI by *modified* Hartung-Knapp-Sidik-Jonkman (t_{k-1}; SE floored at classical RE SE)
  - I^2, tau^2, 95% prediction interval; pooled estimate excluding POP-RT (heterogeneity driver)

Generates BOTH layouts so they can be compared:
  - forest_pairwise_ma_combined.{png,pdf}  (one axis, outcomes stacked as sections)
  - forest_pairwise_ma_panels.{png,pdf}    (3 side-by-side panels, shared study rows)

Endpoints: OS (5), biochemical/progression (5, mixed defn), distant metastasis / MFS (4, mixed).
PCSM/PCSS not pooled (k=2) — printed for reference only.
"""

import os
import numpy as np
from scipy.stats import t as student_t
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import blended_transform_factory

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
DPI = 300

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
C_STUDY = '#3A3A3A'; C_POPRT = '#C0392B'; C_POOL = '#1F4E79'
C_POOLX = '#7F8C8D'; C_NULL = '#9AA0A6'
XLIM = (0.07, 90); EFFECT_MAX = 4.5; SEP_X = 5.6; TXT_X = 7.0
XTICKS = [0.1, 0.25, 0.5, 1, 2, 4]

TRIALS = ['GETUG-01', 'RTOG 9413', 'POP-RT', 'RTOG 0924', 'PEACE-2']
ENDPOINTS = {
    'Overall survival': {'note': '', 'data': {
        'GETUG-01': (0.88, 0.63, 1.22), 'RTOG 9413': (1.11, 0.98, 1.25),
        'POP-RT': (0.92, 0.41, 2.06), 'RTOG 0924': (1.01, 0.85, 1.20),
        'PEACE-2': (1.21, 0.84, 1.73)}},
    'Biochemical / progression': {'note': 'mixed defn: EFS / BCR / BFFS / bPFS', 'data': {
        'GETUG-01': (1.05, 0.83, 1.33), 'RTOG 9413': (0.90, 0.78, 1.04),
        'POP-RT': (0.23, 0.10, 0.52), 'RTOG 0924': (0.82, 0.65, 1.03),
        'PEACE-2': (0.84, 0.66, 1.07)}},
    'Distant metastasis / MFS': {'note': 'mixed defn: DM vs MFS (PEACE-2)', 'data': {
        'GETUG-01': None, 'RTOG 9413': (1.07, 0.84, 1.36),
        'POP-RT': (0.35, 0.15, 0.82), 'RTOG 0924': (1.04, 0.73, 1.49),
        'PEACE-2': (0.90, 0.69, 1.19)}},
}
PCSM = {'RTOG 0924': (1.33, 0.73, 2.42), 'PEACE-2': (0.95, 0.56, 1.64)}


# ── Meta core ────────────────────────────────────────────────────────────────────
def _yv(t):
    hr, lo, hi = t
    return np.log(hr), ((np.log(hi) - np.log(lo)) / (2 * 1.959964)) ** 2

def meta(triples):
    y = np.array([t[0] for t in triples]); v = np.array([t[1] for t in triples]); k = len(y)
    wfe = 1.0 / v; ybar_fe = np.sum(wfe * y) / np.sum(wfe)
    Q = np.sum(wfe * (y - ybar_fe) ** 2)
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    C = np.sum(wfe) - np.sum(wfe ** 2) / np.sum(wfe)            # DerSimonian-Laird
    tau2 = max((Q - (k - 1)) / C, 0.0) if C > 0 else 0.0
    w = 1.0 / (v + tau2); ybar = np.sum(w * y) / np.sum(w)
    se = max(np.sqrt(np.sum(w * (y - ybar) ** 2) / ((k - 1) * np.sum(w))), np.sqrt(1.0 / np.sum(w)))
    tcrit = student_t.ppf(0.975, k - 1)
    ci = (ybar - tcrit * se, ybar + tcrit * se)
    if k >= 3:
        tpi = student_t.ppf(0.975, k - 2)
        pi = (ybar - tpi * np.sqrt(tau2 + se ** 2), ybar + tpi * np.sqrt(tau2 + se ** 2))
    else:
        pi = (np.nan, np.nan)
    return dict(k=k, est=ybar, ci=ci, pi=pi, tau2=tau2, I2=I2, w=w)

def E(x):
    return np.exp(x)


# ── Compute + console report ──────────────────────────────────────────────────────
def compute_results():
    results = {}
    for ep, blob in ENDPOINTS.items():
        rows = [(tr, *_yv(blob['data'][tr]), blob['data'][tr]) for tr in TRIALS if blob['data'][tr]]
        pooled = meta([(r[1], r[2]) for r in rows])
        rows_x = [r for r in rows if r[0] != 'POP-RT']
        pooled_x = meta([(r[1], r[2]) for r in rows_x]) if len(rows_x) >= 2 else None
        results[ep] = dict(rows=rows, pooled=pooled, pooled_x=pooled_x, note=blob['note'])
    return results

def report(results):
    print('=' * 78)
    print('Pairwise random-effects MA (DL + modified Hartung-Knapp). HR<1 favours WPRT.')
    print('=' * 78)
    for ep, R in results.items():
        pooled = R['pooled']
        print(f'\n{ep}  ({R["note"] or "single defn"})')
        for tr, y, v, d in R['rows']:
            print(f'  {tr:<11} HR {d[0]:.2f} ({d[1]:.2f}-{d[2]:.2f})')
        e, (lo, hi) = pooled['est'], pooled['ci']
        print(f'  >> POOLED (k={pooled["k"]}): HR {E(e):.2f} ({E(lo):.2f}-{E(hi):.2f})  '
              f'I2={pooled["I2"]:.0f}%  tau2={pooled["tau2"]:.3f}')
        if not np.isnan(pooled['pi'][0]):
            print(f'     95% prediction interval: {E(pooled["pi"][0]):.2f}-{E(pooled["pi"][1]):.2f}')
        if R['pooled_x']:
            px = R['pooled_x']
            ex, (xl, xh) = px['est'], px['ci']
            print(f'  >> excl. POP-RT (k={px["k"]}): HR {E(ex):.2f} ({E(xl):.2f}-{E(xh):.2f})  I2={px["I2"]:.0f}%')
    print('\nPCSM/PCSS (k=2 — NOT pooled):')
    for tr, d in PCSM.items():
        print(f'  {tr:<11} HR {d[0]:.2f} ({d[1]:.2f}-{d[2]:.2f})')


def _clamp(v):
    return min(max(v, XLIM[0]), EFFECT_MAX)


def _arrows(ax, lo, hi, y, color):
    if hi > EFFECT_MAX:
        ax.annotate('', xy=(EFFECT_MAX, y), xytext=(EFFECT_MAX / 1.7, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))
    if lo < XLIM[0]:
        ax.annotate('', xy=(XLIM[0], y), xytext=(XLIM[0] * 1.7, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))


def _diamond(ax, c, lo, hi, ypos, color, h=0.30):
    ax.add_patch(Polygon([[max(lo, XLIM[0]), ypos], [_clamp(c), ypos + h],
                          [min(hi, EFFECT_MAX), ypos], [_clamp(c), ypos - h]],
                         closed=True, facecolor=color, edgecolor='black', lw=0.7, zorder=5))
    _arrows(ax, lo, hi, ypos, color)


def _direction_below(ax):
    td = blended_transform_factory(ax.transData, ax.transAxes)
    y_arr, y_txt = -0.08, -0.12
    ax.annotate('', xy=(0.09, y_arr), xytext=(0.9, y_arr), xycoords=td, textcoords=td,
                annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
    ax.annotate('', xy=(EFFECT_MAX, y_arr), xytext=(1.12, y_arr), xycoords=td, textcoords=td,
                annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
    ax.text(0.28, y_txt, 'favours WPRT', transform=td, ha='center', va='top',
            fontsize=7, color='#555', clip_on=False)
    ax.text(2.0, y_txt, 'favours PORT', transform=td, ha='center', va='top',
            fontsize=7, color='#555', clip_on=False)


# ── Layout 1: single combined forest ───────────────────────────────────────────────
def plot_combined():
    layout, cur = [], 0.0
    for ep, R in results.items():
        layout.append(('header', cur, ep, R['note'])); cur -= 1.0
        wmax = R['pooled']['w'].max()
        for (tr, y, v, d), w in zip(R['rows'], R['pooled']['w']):
            layout.append(('study', cur, tr, d, w / wmax)); cur -= 1.0
        layout.append(('pooled', cur, R['pooled'])); cur -= 1.0
        if R['pooled_x']:
            layout.append(('poolx', cur, R['pooled_x'])); cur -= 1.0
        cur -= 0.8
    ybottom = cur
    fig, ax = plt.subplots(figsize=(11.5, 0.46 * (-ybottom) + 2.8))
    fig.subplots_adjust(left=0.25, right=0.985, top=0.90, bottom=0.13)
    ax.set_xscale('log'); ax.set_xlim(*XLIM)
    ax.axvline(1, color=C_NULL, ls='--', lw=0.9, zorder=1)
    ax.axvline(SEP_X, color='#e6e6e6', lw=0.8, zorder=1)
    trans_ax = blended_transform_factory(ax.transAxes, ax.transData)
    yticks, ylabels, ycolors = [], [], []
    for item in layout:
        kind, yy = item[0], item[1]
        if kind == 'header':
            ep, note = item[2], item[3]
            ax.text(-0.23, yy, ep, transform=trans_ax, ha='left', va='center',
                    fontsize=10, fontweight='bold', color='#111111')
            if note:
                ax.text(TXT_X, yy, note, ha='left', va='center',
                        fontsize=6.8, style='italic', color='#888888')
        elif kind == 'study':
            tr, d, wrel = item[2], item[3], item[4]; hr, lo, hi = d
            col = C_POPRT if tr == 'POP-RT' else C_STUDY
            ax.plot([max(lo, XLIM[0]), min(hi, EFFECT_MAX)], [yy, yy], '-',
                    color=col, lw=1.4, solid_capstyle='round', zorder=3)
            ax.scatter([_clamp(hr)], [yy], s=30 + 300 * wrel, marker='s', color=col,
                       edgecolor='white', linewidth=0.6, zorder=4)
            _arrows(ax, lo, hi, yy, col)
            ax.text(TXT_X, yy, f'{hr:.2f} ({lo:.2f}–{hi:.2f})', ha='left', va='center',
                    fontsize=7.6, color=col)
            yticks.append(yy); ylabels.append('     ' + tr)
            ycolors.append(C_POPRT if tr == 'POP-RT' else '#333333')
        else:
            R = item[2]; c, lo, hi = E(R['est']), E(R['ci'][0]), E(R['ci'][1])
            color = C_POOL if kind == 'pooled' else C_POOLX
            _diamond(ax, c, lo, hi, yy, color)
            extra = f'   I²={R["I2"]:.0f}%' if kind == 'pooled' else ''
            ax.text(TXT_X, yy, f'{c:.2f} ({lo:.2f}–{hi:.2f})', ha='left', va='center',
                    fontsize=7.8, color=color, fontweight='bold')
            lbl = 'Pooled (DL+HK)' if kind == 'pooled' else 'Pooled excl. POP-RT'
            yticks.append(yy); ylabels.append('  ' + lbl + extra); ycolors.append(color)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8.8)
    for lab, c in zip(ax.get_yticklabels(), ycolors):
        lab.set_color(c)
        if 'Pooled' in lab.get_text():
            lab.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(ybottom + 0.4, 0.9)
    ax.set_xticks(XTICKS); ax.set_xticklabels([str(x) for x in XTICKS], fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    _direction_below(ax)
    fig.suptitle('Pelvic vs prostate-only radiotherapy in localized prostate cancer',
                 x=0.5, y=0.965, fontsize=12.5, fontweight='bold')
    ax.set_title('Random-effects meta-analysis of 5 randomized trials (DL + Hartung-Knapp)',
                 fontsize=9.5, color='#444444', pad=10)
    fig.text(0.25, 0.012,
             'Hazard ratio (95% CI), log scale. Squares sized by study weight; POP-RT (red) drives heterogeneity. '
             'Biochemical & DM/MFS pool mixed endpoint definitions (exploratory). PCSM/PCSS not shown (k=2). '
             'Arrowhead = CI beyond axis. HR<1 favours pelvic RT (WPRT).',
             ha='left', fontsize=7.0, color='#666', va='bottom')
    fig.savefig(os.path.join(OUT, 'forest_pairwise_ma_combined.png'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'forest_pairwise_ma_combined.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Layout 2: 3 side-by-side panels (shared study rows) ─────────────────────────────
def plot_panels():
    Y = {tr: yy for tr, yy in zip(TRIALS, [6, 5, 4, 3, 2])}
    Y_POOL, Y_POOLX = 0.55, -0.4
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.22, wspace=0.50)
    letters = ['a', 'b', 'c']
    for ax, letter, (ep, R) in zip(axes, letters, results.items()):
        for idx, tr in enumerate(TRIALS):
            if idx % 2 == 0:
                ax.axhspan(Y[tr] - 0.5, Y[tr] + 0.5, color='#F4F6F8', zorder=0)
        ax.axvline(1, color=C_NULL, ls='--', lw=0.9, zorder=1)
        ax.axvline(SEP_X, color='#e6e6e6', lw=0.8, zorder=1)
        present = {r[0]: (r[3], w) for r, w in zip(R['rows'], R['pooled']['w'])}
        wmax = R['pooled']['w'].max()
        for tr in TRIALS:
            if tr not in present:
                ax.text(TXT_X, Y[tr], 'NR', ha='left', va='center',
                        fontsize=7.2, color='#AAAAAA', style='italic')
                continue
            d, w = present[tr]; hr, lo, hi = d
            col = C_POPRT if tr == 'POP-RT' else C_STUDY
            ax.plot([max(lo, XLIM[0]), min(hi, EFFECT_MAX)], [Y[tr], Y[tr]], '-',
                    color=col, lw=1.4, solid_capstyle='round', zorder=3)
            ax.scatter([_clamp(hr)], [Y[tr]], s=28 + 300 * (w / wmax), marker='s', color=col,
                       edgecolor='white', linewidth=0.6, zorder=4)
            _arrows(ax, lo, hi, Y[tr], col)
            ax.text(TXT_X, Y[tr], f'{hr:.2f} ({lo:.2f}–{hi:.2f})', ha='left', va='center',
                    fontsize=7.2, color=col)
        ax.axhline(1.35, color='#DDDDDD', lw=0.7)
        for kind, ypos, color in [('pooled', Y_POOL, C_POOL), ('poolx', Y_POOLX, C_POOLX)]:
            RR = R['pooled'] if kind == 'pooled' else R['pooled_x']
            if RR is None:
                continue
            c, lo, hi = E(RR['est']), E(RR['ci'][0]), E(RR['ci'][1])
            _diamond(ax, c, lo, hi, ypos, color, h=0.26)
            ax.text(TXT_X, ypos, f'{c:.2f} ({lo:.2f}–{hi:.2f})', ha='left', va='center',
                    fontsize=7.4, color=color, fontweight='bold')
        ax.text(0.10, -1.35, f'I²={R["pooled"]["I2"]:.0f}%   τ²={R["pooled"]["tau2"]:.2f}',
                ha='left', va='center', fontsize=7.4, color='#333333')
        ax.set_xscale('log'); ax.set_xlim(*XLIM); ax.set_ylim(-1.9, 6.9)
        ax.set_xticks(XTICKS); ax.set_xticklabels([str(x) for x in XTICKS], fontsize=8.5)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        _direction_below(ax)
        ax.set_title(f'{letter}   {ep}', loc='left', fontsize=10.5, fontweight='bold', pad=16)
        if R['note']:
            ax.annotate(R['note'], xy=(0, 1.0), xytext=(0, 11), textcoords='offset points',
                        xycoords='axes fraction', fontsize=7.2, style='italic', color='#777777')
    yt = [Y[t] for t in TRIALS] + [Y_POOL, Y_POOLX]
    ylab = TRIALS + ['Pooled (DL+HK)', 'Pooled excl. POP-RT']
    axes[0].set_yticks(yt); axes[0].set_yticklabels(ylab, fontsize=9)
    for lab in axes[0].get_yticklabels():
        txt = lab.get_text()
        if txt == 'POP-RT':
            lab.set_color(C_POPRT)
        elif txt == 'Pooled excl. POP-RT':
            lab.set_color(C_POOLX); lab.set_fontweight('bold')
        elif txt == 'Pooled (DL+HK)':
            lab.set_color(C_POOL); lab.set_fontweight('bold')
    fig.suptitle('Pelvic vs prostate-only radiotherapy in localized prostate cancer — '
                 'random-effects meta-analysis of 5 randomized trials',
                 x=0.5, y=0.96, fontsize=12.5, fontweight='bold')
    fig.text(0.12, 0.012,
             'Hazard ratio (95% CI), log scale. DL + modified Hartung-Knapp. Squares sized by study weight; '
             'POP-RT (red) drives heterogeneity. Biochemical & DM/MFS pool mixed endpoint definitions. '
             'NR = not reported. Arrowhead = CI beyond axis. HR<1 favours pelvic RT (WPRT).',
             ha='left', fontsize=7.2, color='#666', va='bottom')
    fig.savefig(os.path.join(OUT, 'forest_pairwise_ma_panels.png'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'forest_pairwise_ma_panels.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    results = compute_results()
    report(results)
    plot_combined()
    plot_panels()
    print('\nSaved: forest_pairwise_ma_combined.{png,pdf} and forest_pairwise_ma_panels.{png,pdf}')
