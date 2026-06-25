#!/usr/bin/env python3
"""GRADE certainty-of-evidence assessment + RoB2 — pelvic vs prostate-only RT SR/MA.

For each pooled outcome evaluates the five GRADE domains, derives certainty,
computes anticipated absolute effects per 1,000 at 7 years (baseline from the
reconstructed pseudo-IPD PORT arm; intervention risk by applying the pooled HR
under proportional-hazards), and writes a Summary-of-Findings table.

The canonical RoB 2 assessment (per-outcome) lives in risk_of_bias/rob2_assessment.py;
this script no longer emits a RoB table, to avoid a divergent duplicate.

Notes
-----
- HR used in absolute effects is the aggregate-MA pooled HR (the one shown in the
  Effect column), applied to baseline risk derived from the reconstructed IPD.
- Publication bias is marked 'undetected' because k<10 invalidates Egger/funnel.
"""
import csv
import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
REC = os.path.join(HERE, 'ipd', 'reconstructed')
os.makedirs(OUT, exist_ok=True)

# ── GRADE encoding ───────────────────────────────────────────────────────────
JUDGE_DOWN = {'no': 0, 'serious': -1, 'very_serious': -2, 'undetected': 0}
JUDGE_LABEL = {'no': 'No concerns', 'serious': 'Serious', 'very_serious': 'Very serious',
               'undetected': 'Undetected'}
JUDGE_SYM = {'no': '✓', 'serious': '↓', 'very_serious': '↓↓', 'undetected': '–'}
JUDGE_COLOR = {'no': '#A6D785', 'serious': '#F6B26B', 'very_serious': '#E06666',
               'undetected': '#D9D9D9'}
CERTAINTY = {4: ('High', '⊕⊕⊕⊕'), 3: ('Moderate', '⊕⊕⊕○'),
             2: ('Low', '⊕⊕○○'), 1: ('Very low', '⊕○○○')}
CERT_COLOR = {'High': '#1F4E79', 'Moderate': '#5B9BD5', 'Low': '#F6B26B', 'Very low': '#C0392B'}

DOMAINS = ('rob', 'inconsistency', 'indirectness', 'imprecision', 'pubbias')
DOMAIN_HEAD = {'rob': 'Risk of bias', 'inconsistency': 'Inconsistency',
               'indirectness': 'Indirectness', 'imprecision': 'Imprecision',
               'pubbias': 'Publication\nbias'}

# ── Per-outcome data ─────────────────────────────────────────────────────────
HORIZON_MONTHS = 84  # 7 years
OUTCOMES = {
    'Overall survival': dict(
        k=5, n=5172,
        hr=1.07, lo=0.94, hi=1.21,
        effect='HR 1.07 (0.94 to 1.21)',
        ipd_port=['POP-RT_OS_ipd.csv', 'GETUG-01_OS_ipd.csv',
                  'PEACE-2_OS_ipd.csv', 'RTOG0924_OS_ipd.csv'],
        rob=('no', 'Phase-3 RCTs; RoB2 reviewer judgement Low / Some concerns (see RoB2 table).'),
        inconsistency=('no', 'I²=0%; directionally consistent across all 5 trials.'),
        indirectness=('serious', 'Applicability concern: trials span 30 years of practice change — RTOG 9413 (3D-CRT, no PSMA, short ADT) and GETUG-01 (3D-CRT) versus POP-RT/PEACE-2/RTOG 0924 (IMRT, partial PSMA, longer/intensified systemic therapy). Death endpoint uniformly defined, but the modern IMRT+PSMA patient is underrepresented.'),
        imprecision=('no', 'CI 0.94–1.21 narrow; excludes appreciable benefit (HR<0.85) and harm (HR>1.18).'),
        pubbias=('undetected', 'k=5 (<10) → Egger/funnel invalid; all identified phase-3 RCTs included.'),
    ),
    'Biochemical / progression (mixed defn)': dict(
        k=5, n=5172,
        hr=0.84, lo=0.54, hi=1.29,
        effect='HR 0.84 (0.54 to 1.29)',
        ipd_port=['POP-RT_BFFS_ipd.csv', 'GETUG-01_EFS_ipd.csv', 'PEACE-2_BPFS_ipd.csv'],
        rob=('serious', 'RTOG 9413 rated High risk of bias for this outcome (significant field × ADT-timing factorial interaction makes the collapsed WPRT-vs-PORT contrast uninterpretable as a single effect); the other four trials Some concerns (see RoB2 table).'),
        inconsistency=('very_serious', 'I²=69%, τ²=0.03; POP-RT (HR 0.23) opposes GETUG-01 (HR 1.05) and others. Pool dominated by one outlier; excluding POP-RT collapses heterogeneity to 0.'),
        indirectness=('serious', 'Outcome definitions differ (BFFS Phoenix / EFS / cPFS / bPFS / BCR); population/era heterogeneity (PSMA staging only in POP-RT) plausibly modifies the effect.'),
        imprecision=('serious', 'CI 0.54–1.29 wide; spans appreciable benefit (HR<0.85), no effect, and clinically meaningful harm.'),
        pubbias=('undetected', 'k=5 (<10) → tests invalid; all relevant phase-3 RCTs included.'),
    ),
    'Distant metastasis / MFS (mixed defn)': dict(
        k=4, n=4728,
        hr=0.92, lo=0.54, hi=1.57,
        effect='HR 0.92 (0.54 to 1.57)',
        ipd_port=['POP-RT_DMFS_ipd.csv', 'PEACE-2_MFS_ipd.csv'],
        rob=('no', 'Phase-3 RCTs; RoB2 Low / Some concerns (see RoB2 table).'),
        inconsistency=('serious', 'I²=55%; POP-RT (HR 0.35) drives heterogeneity; excluding it gives HR 1.00 with I²=0%.'),
        indirectness=('serious', 'Endpoint mixes MFS (POP-RT, PEACE-2 — death as competing event) and DM (RTOG 9413, RTOG 0924 — cause-specific) — distinct constructs pooled.'),
        imprecision=('serious', 'CI 0.54–1.57 very wide; spans large benefit and meaningful harm.'),
        pubbias=('undetected', 'k=4 (<10) → tests invalid.'),
    ),
}


def certainty_of(o):
    return max(1, 4 + sum(JUDGE_DOWN[o[d][0]] for d in DOMAINS))


# ── Anticipated absolute effects ─────────────────────────────────────────────
def baseline_risk(ipd_files, t):
    frames = [pd.read_csv(os.path.join(REC, f)) for f in ipd_files]
    port = pd.concat(frames, ignore_index=True)
    port = port[port.arm == 0]
    kmf = KaplanMeierFitter().fit(port.time, port.event)
    s = float(kmf.survival_function_at_times(t).iloc[0])
    return 1.0 - s  # event risk over t


def absolute_per_1000(o, t):
    r_c = baseline_risk(o['ipd_port'], t)
    # Convert HR to risk under proportional-hazards: R_i = 1 - (1-R_c)^HR
    r_i = 1.0 - (1.0 - r_c) ** o['hr']
    r_i_lo = 1.0 - (1.0 - r_c) ** o['lo']
    r_i_hi = 1.0 - (1.0 - r_c) ** o['hi']
    return dict(R_c=r_c, R_i=r_i, R_i_lo=r_i_lo, R_i_hi=r_i_hi,
                diff=(r_i - r_c), diff_lo=(r_i_lo - r_c), diff_hi=(r_i_hi - r_c))


# ── console + CSV ────────────────────────────────────────────────────────────
def report():
    rows = [['outcome', 'k', 'n', 'effect',
             'baseline_per1000_PORT_7y', 'intervention_per1000_WPRT_7y',
             'intervention_lo', 'intervention_hi',
             'rob', 'inconsistency', 'indirectness', 'imprecision', 'pubbias', 'certainty']]
    print('=' * 84)
    print('GRADE certainty of evidence — pelvic (WPRT) vs prostate-only RT (PORT)')
    print('=' * 84)
    for name, o in OUTCOMES.items():
        ae = absolute_per_1000(o, HORIZON_MONTHS)
        lv = certainty_of(o); lvl, symb = CERTAINTY[lv]
        print(f'\n{name}  (k={o["k"]}, n≈{o["n"]:,})')
        print(f'  Effect: {o["effect"]}')
        print(f'  Anticipated absolute effect at 7 yr (events per 1,000):')
        print(f'     PORT (baseline)      : {ae["R_c"]*1000:.0f}')
        print(f'     WPRT (intervention)  : {ae["R_i"]*1000:.0f} ({ae["R_i_lo"]*1000:.0f} to {ae["R_i_hi"]*1000:.0f})')
        print(f'     Difference (WPRT-PORT): {ae["diff"]*1000:+.0f} ({ae["diff_lo"]*1000:+.0f} to {ae["diff_hi"]*1000:+.0f}) per 1,000')
        for d in DOMAINS:
            j, why = o[d]
            print(f'  {DOMAIN_HEAD[d].replace(chr(10)," "):<18} {JUDGE_LABEL[j]:<14} {JUDGE_SYM[j]}  — {why}')
        print(f'  >> Certainty: {lvl} {symb}')
        rows.append([name, o['k'], o['n'], o['effect'],
                     round(ae['R_c'] * 1000, 0), round(ae['R_i'] * 1000, 0),
                     round(ae['R_i_lo'] * 1000, 0), round(ae['R_i_hi'] * 1000, 0),
                     JUDGE_LABEL[o['rob'][0]], JUDGE_LABEL[o['inconsistency'][0]],
                     JUDGE_LABEL[o['indirectness'][0]], JUDGE_LABEL[o['imprecision'][0]],
                     JUDGE_LABEL[o['pubbias'][0]], lvl])
    path = os.path.join(OUT, 'grade_summary.csv')
    with open(path, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f'\nSaved: {path}')


# ── Summary-of-Findings figure (with absolute effects) ───────────────────────
def sof_figure():
    headers = ['Outcome', 'No. of\ntrials (pts)', 'Relative effect\n(HR, 95% CI)',
               'Anticipated absolute effect\nat 7 yr (per 1,000)',
               *[DOMAIN_HEAD[d] for d in DOMAINS], 'Certainty']
    body = []
    for name, o in OUTCOMES.items():
        ae = absolute_per_1000(o, HORIZON_MONTHS)
        lv = certainty_of(o); lvl, symb = CERTAINTY[lv]
        body.append({'name': name, 'kn': f'{o["k"]} ({o["n"]:,})', 'effect': o['effect'],
                     'abs': ae,
                     'domains': [o[d][0] for d in DOMAINS], 'cert': (lvl, symb)})

    col_w = [3.0, 1.2, 1.9, 2.6, 1.35, 1.35, 1.35, 1.35, 1.35, 2.0]
    total_w = sum(col_w)
    n = len(body)
    row_h = 1.4
    fig_w = 19; fig_h = 1.8 + row_h * n
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, n * row_h + 1.4)
    ax.axis('off')

    # Header
    x0 = 0
    for h, w in zip(headers, col_w):
        ax.add_patch(Rectangle((x0, n * row_h + 0.4), w, 1.0,
                               facecolor='#1F4E79', edgecolor='white', lw=1.2))
        ax.text(x0 + w / 2, n * row_h + 0.9, h, ha='center', va='center',
                fontsize=8.6, fontweight='bold', color='white')
        x0 += w

    # Body
    for ri, r in enumerate(body):
        y = (n - ri - 1) * row_h + 0.4
        h = row_h
        x0 = 0
        # Outcome
        ax.add_patch(Rectangle((x0, y), col_w[0], h, facecolor='#F4F6F8', edgecolor='white', lw=1.2))
        ax.text(x0 + 0.15, y + h / 2, r['name'], ha='left', va='center',
                fontsize=8.4, fontweight='bold', color='#111')
        x0 += col_w[0]
        # k (n)
        ax.add_patch(Rectangle((x0, y), col_w[1], h, facecolor='white', edgecolor='white', lw=1.2))
        ax.text(x0 + col_w[1] / 2, y + h / 2, r['kn'], ha='center', va='center', fontsize=8.4)
        x0 += col_w[1]
        # Relative effect
        ax.add_patch(Rectangle((x0, y), col_w[2], h, facecolor='white', edgecolor='white', lw=1.2))
        ax.text(x0 + col_w[2] / 2, y + h / 2, r['effect'], ha='center', va='center', fontsize=8.4)
        x0 += col_w[2]
        # Absolute effect
        ae = r['abs']
        ax.add_patch(Rectangle((x0, y), col_w[3], h, facecolor='white', edgecolor='white', lw=1.2))
        line1 = f'PORT: {ae["R_c"]*1000:.0f} per 1,000'
        line2 = f'WPRT: {ae["R_i"]*1000:.0f} ({ae["R_i_lo"]*1000:.0f}–{ae["R_i_hi"]*1000:.0f})'
        line3 = f'Δ {ae["diff"]*1000:+.0f} ({ae["diff_lo"]*1000:+.0f} to {ae["diff_hi"]*1000:+.0f})'
        ax.text(x0 + col_w[3] / 2, y + h / 2 + 0.30, line1, ha='center', va='center', fontsize=7.8, color='#444')
        ax.text(x0 + col_w[3] / 2, y + h / 2, line2, ha='center', va='center', fontsize=8.2, fontweight='bold')
        ax.text(x0 + col_w[3] / 2, y + h / 2 - 0.30, line3, ha='center', va='center', fontsize=7.8, color='#555')
        x0 += col_w[3]
        # Domains
        for j, dw in zip(r['domains'], col_w[4:9]):
            ax.add_patch(Rectangle((x0, y), dw, h, facecolor=JUDGE_COLOR[j], edgecolor='white', lw=1.2))
            ax.text(x0 + dw / 2, y + h / 2 + 0.18, JUDGE_SYM[j], ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(x0 + dw / 2, y + h / 2 - 0.35, JUDGE_LABEL[j], ha='center', va='center', fontsize=6.8, color='#222')
            x0 += dw
        # Certainty
        lvl, symb = r['cert']
        ax.add_patch(Rectangle((x0, y), col_w[9], h, facecolor=CERT_COLOR[lvl], edgecolor='white', lw=1.2))
        ax.text(x0 + col_w[9] / 2, y + h / 2 + 0.18, symb, ha='center', va='center', fontsize=14, color='white', fontweight='bold')
        ax.text(x0 + col_w[9] / 2, y + h / 2 - 0.35, lvl, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    fig.suptitle('GRADE Summary of Findings — pelvic (WPRT) vs prostate-only RT (PORT)',
                 y=0.99, fontsize=12.5, fontweight='bold')
    fig.text(0.03, 0.02,
             'Anticipated absolute effects per 1,000 at 7 years: baseline = reconstructed-IPD PORT arm; '
             'WPRT = baseline transformed by the pooled HR under proportional hazards. '
             'Domains: ✓ no concerns · ↓ serious · ↓↓ very serious · – undetected. '
             'Publication bias not formally testable at k<10.',
             fontsize=7.2, color='#555')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'grade_sof_table.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: grade_sof_table.{{png,pdf}}')


if __name__ == '__main__':
    report()
    sof_figure()
