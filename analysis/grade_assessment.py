#!/usr/bin/env python3
"""GRADE certainty-of-evidence assessment + RoB2 — pelvic vs prostate-only RT SR/MA.

For each pooled outcome evaluates the five GRADE domains, derives certainty,
computes anticipated absolute effects per 1,000 at 7 years (baseline from the
reconstructed pseudo-IPD PORT arm; intervention risk by applying the pooled HR
under proportional-hazards), and writes a Summary-of-Findings table.

Also produces a RoB2 traffic-light table for the 5 trials (5 domains + overall).

Notes
-----
- HR used in absolute effects is the aggregate-MA pooled HR (the one shown in the
  Effect column), applied to baseline risk derived from the reconstructed IPD.
- Publication bias is marked 'undetected' because k<10 invalidates Egger/funnel.
- RoB2 judgements below are the reviewer's working draft — adjust per protocol.
"""
import csv
import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

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
        rob=('no', 'Phase-3 RCTs; RoB2 Low / Some concerns (see RoB2 table).'),
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


# ── RoB2 per-trial (reviewer working draft) ──────────────────────────────────
ROB_DOMAINS = ['D1: Randomization', 'D2: Deviations from intended\nintervention',
               'D3: Missing outcome\ndata', 'D4: Measurement of\nthe outcome',
               'D5: Selection of the\nreported result', 'Overall']
ROB_KEY = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
ROB_LEVEL = {'low': ('Low', '#A6D785', '+'),
             'some': ('Some concerns', '#F6B26B', '!'),
             'high': ('High', '#E06666', '−')}

TRIALS_ROB = {
    'RTOG 9413 (Roach 2018, Lancet Oncol)': {
        'D1': ('low', 'Computer-generated allocation, 2×2 stratified by risk; concealment described.'),
        'D2': ('some', 'Open-label (RT cannot be blinded). Field×hormone-timing interaction emphasized on update is recognised as the principal report but post-hoc relative to original 2003 design.'),
        'D3': ('low', '>90% follow-up at long-term update; minimal missingness.'),
        'D4': ('low', 'Death/PFS objectively ascertained.'),
        'D5': ('some', 'Multiple long-term updates (2003, 2007, 2013, 2018) with shifting endpoint emphasis.'),
        'Overall': 'some',
    },
    'GETUG-01 (Pommier 2016, IJROBP)': {
        'D1': ('low', 'Central randomization, stratified by risk group.'),
        'D2': ('some', 'Open-label RT trial.'),
        'D3': ('low', '15-year follow-up; vital status complete.'),
        'D4': ('low', 'OS objective; EFS centrally adjudicated.'),
        'D5': ('low', 'Long-term endpoints pre-specified; consistent reporting across updates.'),
        'Overall': 'low',
    },
    'POP-RT (Murthy 2021, JCO)': {
        'D1': ('low', 'Central allocation, stratified by Roach >35%; concealment ensured.'),
        'D2': ('some', 'Open-label. Single-centre delivery limits generalisability of technique.'),
        'D3': ('low', 'Median follow-up 68 mo; <5% loss.'),
        'D4': ('low', 'BFFS adjudicated, OS objective.'),
        'D5': ('low', 'Primary and key secondary outcomes pre-registered (NCT02302105).'),
        'Overall': 'low',
    },
    'PEACE-2 (Blanchard ESTRO 2026 — congress slides)': {
        'D1': ('low', '2×2 factorial (pelvic RT × cabazitaxel), central randomization.'),
        'D2': ('some', 'Open-label; cabazitaxel co-intervention may dilute/mask pelvic-RT effect estimate even with factorial analysis.'),
        'D3': ('some', 'Full per-protocol numbers and missing-data handling not yet in a peer-reviewed full report.'),
        'D4': ('low', 'cPFS, bPFS, MFS, OS pre-specified.'),
        'D5': ('some', 'Pre-publication: only the pelvic-RT randomization is publicly reported; full SAP and pre-specified subgroups await full publication.'),
        'Overall': 'some',
    },
    'RTOG 0924 (Roach ASTRO 2025 — congress slides)': {
        'D1': ('low', 'NRG central randomization, stratified.'),
        'D2': ('some', 'Open-label.'),
        'D3': ('some', '139 of 514 deaths still being adjudicated for cause; affects PCSM but not OS.'),
        'D4': ('low', 'OS objective.'),
        'D5': ('some', 'Pre-publication slide release; full report and pre-specified secondary endpoints pending.'),
        'Overall': 'some',
    },
}


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

    # RoB2 CSV
    rob_rows = [['trial', *ROB_KEY[:-1], 'Overall', 'D1_reason', 'D2_reason',
                 'D3_reason', 'D4_reason', 'D5_reason']]
    for tr, r in TRIALS_ROB.items():
        rob_rows.append([tr, ROB_LEVEL[r['D1'][0]][0], ROB_LEVEL[r['D2'][0]][0],
                         ROB_LEVEL[r['D3'][0]][0], ROB_LEVEL[r['D4'][0]][0],
                         ROB_LEVEL[r['D5'][0]][0], ROB_LEVEL[r['Overall']][0],
                         r['D1'][1], r['D2'][1], r['D3'][1], r['D4'][1], r['D5'][1]])
    with open(os.path.join(OUT, 'rob2_summary.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows(rob_rows)
    print(f'Saved: {os.path.join(OUT, "rob2_summary.csv")}')


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


# ── RoB2 traffic-light figure ────────────────────────────────────────────────
def rob2_figure():
    trials = list(TRIALS_ROB)
    n_t = len(trials); n_d = len(ROB_KEY)
    cell_w = [3.6, *([1.35] * 5), 1.5]
    total_w = sum(cell_w)
    row_h = 1.0
    fig_w = 14.5; fig_h = 1.8 + row_h * n_t
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, n_t * row_h + 1.4)
    ax.axis('off')

    # Header
    headers = ['Trial', *ROB_DOMAINS]
    x = 0
    for h, w in zip(headers, cell_w):
        ax.add_patch(Rectangle((x, n_t * row_h + 0.4), w, 1.0,
                               facecolor='#1F4E79', edgecolor='white', lw=1.2))
        ax.text(x + w / 2, n_t * row_h + 0.9, h, ha='center', va='center',
                fontsize=8.2, fontweight='bold', color='white')
        x += w

    for ri, tr in enumerate(trials):
        y = (n_t - ri - 1) * row_h + 0.4
        x = 0
        ax.add_patch(Rectangle((x, y), cell_w[0], row_h, facecolor='#F4F6F8', edgecolor='white', lw=1.2))
        ax.text(x + 0.12, y + row_h / 2, tr, ha='left', va='center', fontsize=8.2, fontweight='bold', color='#111')
        x += cell_w[0]
        for di, key in enumerate(ROB_KEY[:-1]):
            lvl = TRIALS_ROB[tr][key][0]
            label, color, sym = ROB_LEVEL[lvl]
            ax.add_patch(Rectangle((x, y), cell_w[1 + di], row_h, facecolor='white', edgecolor='white', lw=1.2))
            ax.add_patch(Circle((x + cell_w[1 + di] / 2, y + row_h / 2), 0.28,
                                facecolor=color, edgecolor='black', lw=0.8))
            ax.text(x + cell_w[1 + di] / 2, y + row_h / 2 + 0.005, sym,
                    ha='center', va='center', fontsize=12, fontweight='bold', color='black')
            x += cell_w[1 + di]
        ov = TRIALS_ROB[tr]['Overall']
        label, color, sym = ROB_LEVEL[ov]
        ax.add_patch(Rectangle((x, y), cell_w[-1], row_h, facecolor=color, edgecolor='white', lw=1.2))
        ax.text(x + cell_w[-1] / 2, y + row_h / 2 + 0.06, sym, ha='center', va='center',
                fontsize=14, fontweight='bold', color='black')
        ax.text(x + cell_w[-1] / 2, y + row_h / 2 - 0.28, label, ha='center', va='center',
                fontsize=7.4, color='black', fontweight='bold')

    fig.suptitle('Risk of bias (RoB 2) — per-trial judgement',
                 y=0.99, fontsize=12.5, fontweight='bold')
    fig.text(0.03, 0.02,
             'Symbols: + Low risk · ! Some concerns · − High risk. Reviewer working draft based on the available '
             'publications and congress reports; revise after the full reports of PEACE-2 and RTOG 0924 appear.',
             fontsize=7.2, color='#555')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'rob2_traffic_light.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: rob2_traffic_light.{{png,pdf}}')


if __name__ == '__main__':
    report()
    sof_figure()
    rob2_figure()
