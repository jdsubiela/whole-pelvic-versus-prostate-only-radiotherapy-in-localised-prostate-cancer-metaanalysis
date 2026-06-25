#!/usr/bin/env python3
"""Two-stage reconstructed-IPD meta-analysis: WPRT vs PORT.

Uses the validated pseudo-IPD (reconstructed/*.csv) from the trials with a
digitizable Kaplan-Meier curve: POP-RT, GETUG-01 (published) + PEACE-2 (slide).
RTOG 9413 (field x hormone-timing interaction plots) and RTOG 0924 (competing-
risks cumulative-incidence + OS KM not captured) have no poolable KM and remain
aggregate-only -- see reconstruction note.

For each endpoint group it:
  1. fits a per-trial Cox PH on the reconstructed IPD -> logHR, SE;
  2. random-effects pools (DerSimonian-Laird tau^2 + modified Hartung-Knapp CI),
     for PRIMARY (published trials only) and for ALL (+ slide, sensitivity layer);
  3. pools the RMST difference at a common horizon (per-trial bootstrap SE -> RE);
  4. checks proportional hazards on the pooled IPD (Schoenfeld) and reports a
     piecewise HR (<=60 vs >60 months) as a descriptive of time-varying effect.

Outputs (figures_ipd/):
  pooled_ipd_summary.csv
  forest_ipd_pooled.{png,pdf}
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import blended_transform_factory

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, 'reconstructed')
OUT = os.path.join(HERE, '..', 'figures_ipd')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260528)

# endpoint group -> list of (trial label, ipd filename, source). HR<1 favours WPRT.
GROUPS = {
    'Overall survival': [
        ('POP-RT', 'POP-RT_OS_ipd.csv', 'published'),
        ('GETUG-01', 'GETUG-01_OS_ipd.csv', 'published'),
        ('PEACE-2', 'PEACE-2_OS_ipd.csv', 'slide'),
        ('RTOG 0924', 'RTOG0924_OS_ipd.csv', 'slide'),
    ],
    'Biochemical / progression': [
        ('POP-RT', 'POP-RT_BFFS_ipd.csv', 'published'),
        ('GETUG-01', 'GETUG-01_EFS_ipd.csv', 'published'),
        ('PEACE-2', 'PEACE-2_BPFS_ipd.csv', 'slide'),
    ],
    'Distant metastasis / MFS': [
        ('POP-RT', 'POP-RT_DMFS_ipd.csv', 'published'),
        ('PEACE-2', 'PEACE-2_MFS_ipd.csv', 'slide'),
    ],
}


def cox_loghr(ipd):
    cph = CoxPHFitter().fit(ipd[['time', 'event', 'arm']], 'time', 'event')
    return float(cph.params_['arm']), float(cph.standard_errors_['arm'])


def km_rmst(time, event, tau):
    kmf = KaplanMeierFitter().fit(time, event)
    sf = kmf.survival_function_.reset_index(); sf.columns = ['t', 's']
    sf = sf[sf['t'] <= tau]
    edges = np.append(sf['t'].values, tau)
    return float(np.sum(sf['s'].values * np.diff(edges)))


def rmst_diff_se(ipd, tau, nboot=400):
    wp, po = ipd[ipd.arm == 1], ipd[ipd.arm == 0]
    base = km_rmst(wp.time.values, wp.event.values, tau) - km_rmst(po.time.values, po.event.values, tau)
    bs = []
    for _ in range(nboot):
        bw = wp.sample(len(wp), replace=True, random_state=RNG.integers(1 << 31))
        bp = po.sample(len(po), replace=True, random_state=RNG.integers(1 << 31))
        bs.append(km_rmst(bw.time.values, bw.event.values, tau) - km_rmst(bp.time.values, bp.event.values, tau))
    return base, float(np.std(bs, ddof=1))


def re_pool(y, s):
    y, v = np.asarray(y, float), np.asarray(s, float) ** 2
    k = len(y)
    wfe = 1 / v; ybar_fe = (wfe * y).sum() / wfe.sum()
    Q = (wfe * (y - ybar_fe) ** 2).sum()
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if (Q > 0 and k > 1) else 0.0
    C = wfe.sum() - (wfe ** 2).sum() / wfe.sum()
    tau2 = max((Q - (k - 1)) / C, 0.0) if (C > 0 and k > 1) else 0.0
    w = 1 / (v + tau2); ybar = (w * y).sum() / w.sum()
    se_re = np.sqrt(1 / w.sum())
    if k >= 2:
        se_hk = np.sqrt((w * (y - ybar) ** 2).sum() / ((k - 1) * w.sum()))
        se = max(se_hk, se_re); crit = student_t.ppf(0.975, k - 1)
    else:
        se = se_re; crit = 1.959964
    return dict(est=ybar, lo=ybar - crit * se, hi=ybar + crit * se, I2=I2, tau2=tau2, k=k)


def analyse():
    rows = []
    per_endpoint = {}
    for ep, trials in GROUPS.items():
        recs = []
        pooled_ipd = []
        taus = []
        for label, fn, source in trials:
            ipd = pd.read_csv(os.path.join(REC, fn))
            y, s = cox_loghr(ipd)
            taus.append(ipd.time.max())
            recs.append(dict(trial=label, source=source, y=y, s=s,
                             hr=np.exp(y), lo=np.exp(y - 1.96 * s), hi=np.exp(y + 1.96 * s)))
            pooled_ipd.append(ipd)
        tau = float(min(taus))  # common RMST horizon = shortest max-follow-up
        # per-trial RMST diff at common tau
        for r, (label, fn, source) in zip(recs, trials):
            ipd = pd.read_csv(os.path.join(REC, fn))
            d, se = rmst_diff_se(ipd, tau)
            r['rmst'] = d; r['rmst_se'] = se

        pub = [r for r in recs if r['source'] == 'published']
        allr = recs

        def pack(subset, name):
            p = re_pool([r['y'] for r in subset], [r['s'] for r in subset])
            pr = re_pool([r['rmst'] for r in subset], [r['rmst_se'] for r in subset])
            return dict(name=name, k=p['k'], hr=np.exp(p['est']), lo=np.exp(p['lo']), hi=np.exp(p['hi']),
                        I2=p['I2'], tau2=p['tau2'], rmst=pr['est'], rmst_lo=pr['lo'], rmst_hi=pr['hi'], tau=tau)

        pooled = {}
        if len(pub) >= 1:
            pooled['primary'] = pack(pub, f'Primary (published, k={len(pub)})')
        pooled['all'] = pack(allr, f'+ slide trials (k={len(allr)})')

        # PH test + piecewise HR on the combined IPD (all trials)
        comb = pd.concat(pooled_ipd, ignore_index=True)
        cph = CoxPHFitter().fit(comb[['time', 'event', 'arm']], 'time', 'event')
        ph_p = float(proportional_hazard_test(cph, comb[['time', 'event', 'arm']]).p_value[0])
        early = comb.copy(); early['event'] = np.where(early['time'] <= 60, early['event'], 0)
        early['time'] = np.minimum(early['time'], 60)
        hr_early = np.exp(CoxPHFitter().fit(early[['time', 'event', 'arm']], 'time', 'event').params_['arm'])
        late = comb[comb['time'] > 60].copy()
        hr_late = np.exp(CoxPHFitter().fit(late[['time', 'event', 'arm']], 'time', 'event').params_['arm']) if late['event'].sum() > 5 else np.nan

        per_endpoint[ep] = dict(recs=recs, pooled=pooled, ph_p=ph_p, hr_early=hr_early, hr_late=hr_late, tau=tau)

        print(f'\n### {ep}   (common RMST horizon tau={tau:.0f} mo)')
        for r in recs:
            print(f'  {r["trial"]:<9} [{r["source"]:<9}] HR {r["hr"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f})   RMST {r["rmst"]:+.1f} mo')
            rows.append(dict(endpoint=ep, level='trial', label=r['trial'], source=r['source'], k=1,
                             hr=round(r['hr'], 3), lo=round(r['lo'], 3), hi=round(r['hi'], 3),
                             rmst=round(r['rmst'], 2)))
        for key, P in pooled.items():
            print(f'  >> {P["name"]:<26} HR {P["hr"]:.2f} ({P["lo"]:.2f}-{P["hi"]:.2f})  '
                  f'I2={P["I2"]:.0f}%  RMST {P["rmst"]:+.1f} ({P["rmst_lo"]:+.1f},{P["rmst_hi"]:+.1f}) mo')
            rows.append(dict(endpoint=ep, level='pooled', label=P['name'], source=key, k=P['k'],
                             hr=round(P['hr'], 3), lo=round(P['lo'], 3), hi=round(P['hi'], 3),
                             rmst=round(P['rmst'], 2),
                             I2_pct=round(P['I2'], 1), tau2=round(P['tau2'], 4)))
        pe = f'PH Schoenfeld p={ph_p:.2f}; piecewise HR <=60mo {hr_early:.2f}, >60mo {hr_late:.2f}'
        print(f'  -- {pe}')
        rows.append(dict(endpoint=ep, level='diagnostic', label=pe, source='', k='',
                         hr='', lo='', hi='', rmst=''))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pooled_ipd_summary.csv'), index=False)
    forest(per_endpoint)
    print(f'\nSaved pooled_ipd_summary.csv + forest_ipd_pooled.{{png,pdf}} in {OUT}')


# ── forest ────────────────────────────────────────────────────────────────────
# Effect zone holds the markers (0.06-EFFECT_MAX); the blank zone to its right
# (up to XLIM hi) is reserved for the HR-text column so labels never overlap data.
XLIM = (0.06, 90); EFFECT_MAX = 4.5; SEP_X = 5.6; TXT_X = 7.0
XTICKS = [0.1, 0.25, 0.5, 1, 2, 4]
C_PUB = '#3A3A3A'; C_PRIM = '#1F4E79'; C_NULL = '#9AA0A6'


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
    ax.add_patch(Polygon([[max(lo, XLIM[0]), y], [_clamp(c), y + h], [min(hi, EFFECT_MAX), y],
                          [_clamp(c), y - h]], closed=True,
                         facecolor=color, edgecolor='black', lw=0.7, zorder=5))
    _arrows(ax, lo, hi, y, color)


def forest(per_endpoint):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.28, wspace=0.50)
    for ax, letter, (ep, D) in zip(axes, ['a', 'b', 'c'], per_endpoint.items()):
        items = [(r['trial'], r['hr'], r['lo'], r['hi'], C_PUB, False) for r in D['recs']]
        P = D['pooled']['all']
        items.append(('Pooled (random-effects)', P['hr'], P['lo'], P['hi'], C_PRIM, True))
        ys = list(range(len(items) - 1, -1, -1))
        ax.axvline(1, color=C_NULL, ls='--', lw=0.9)
        ax.axvline(SEP_X, color='#e6e6e6', lw=0.8)
        ax.text(TXT_X, ys[0] + 0.85, 'HR (95% CI)', ha='left', va='center', fontsize=7, style='italic', color='#888')
        for (lab, hr, lo, hi, col, pooled), y in zip(items, ys):
            if pooled:
                _diamond(ax, hr, lo, hi, y, col)
            else:
                ax.plot([max(lo, XLIM[0]), min(hi, EFFECT_MAX)], [y, y], '-', color=col, lw=1.5, zorder=3)
                ax.scatter([_clamp(hr)], [y], s=36, marker='s', color=col, edgecolor='white', lw=0.6, zorder=4)
                _arrows(ax, lo, hi, y, col)
            ax.text(TXT_X, y, f'{hr:.2f} ({lo:.2f}–{hi:.2f})', ha='left', va='center',
                    fontsize=7.6, color=col, fontweight='bold' if pooled else 'normal')
        ax.set_yticks(ys); ax.set_yticklabels([it[0] for it in items], fontsize=8.4)
        for lab, it in zip(ax.get_yticklabels(), items):
            lab.set_color(it[4]); lab.set_fontweight('bold' if it[5] else 'normal')
        ax.set_xscale('log'); ax.set_xlim(*XLIM); ax.set_ylim(-0.6, len(items) - 0.2)
        ax.set_xticks(XTICKS); ax.set_xticklabels([str(x) for x in XTICKS], fontsize=8)
        ax.tick_params(axis='y', length=0); ax.spines[['top', 'right', 'left']].set_visible(False)
        # per-panel direction of effect: arrows + labels BELOW the x-axis (axes-fraction y)
        td = blended_transform_factory(ax.transData, ax.transAxes)
        y_arr = -0.20; y_txt = -0.27
        ax.annotate('', xy=(0.085, y_arr), xytext=(0.9, y_arr), xycoords=td, textcoords=td,
                    annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
        ax.annotate('', xy=(EFFECT_MAX, y_arr), xytext=(1.12, y_arr), xycoords=td, textcoords=td,
                    annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.0))
        ax.text(0.27, y_txt, 'favours WPRT', transform=td, ha='center', va='top',
                fontsize=7, color='#555', clip_on=False)
        ax.text(2.0, y_txt, 'favours PORT', transform=td, ha='center', va='top',
                fontsize=7, color='#555', clip_on=False)
        sub = f'PH p={D["ph_p"]:.2f} · HR ≤60mo {D["hr_early"]:.2f} / >60mo {D["hr_late"]:.2f} · I²={P["I2"]:.0f}%'
        ax.set_title(f'{letter}  {ep}\n{sub}', loc='left', fontsize=9.5, fontweight='bold', pad=8)
    fig.suptitle('Reconstructed-IPD meta-analysis (two-stage, DL + Hartung-Knapp) — pelvic vs prostate-only RT',
                 x=0.5, y=0.96, fontsize=12.5, fontweight='bold')
    fig.text(0.10, 0.012, 'Hazard ratio (95% CI), log scale. Two-stage random-effects meta-analysis '
             '(DerSimonian-Laird τ² + Hartung-Knapp CI) of Guyot-reconstructed pseudo-IPD. Squares = trials, '
             'diamond = pooled. Arrowhead = CI beyond axis. HR<1 favours pelvic RT (WPRT).',
             ha='left', fontsize=7.0, color='#666')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'forest_ipd_pooled.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    analyse()
