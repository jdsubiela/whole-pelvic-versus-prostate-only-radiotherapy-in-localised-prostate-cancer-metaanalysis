#!/usr/bin/env python3
"""Descriptive (NOT pooled) adverse-events table — pelvic vs prostate-only RT SR/MA.

Each row reports the adverse-event numbers exactly as published / presented, with the
delivery technique, scale and timing preserved. No transformation across CTCAE
versions or timing windows is performed: structural heterogeneity precludes
quantitative pooling (see manuscript/limitations_notes.md, block L1).

Outputs (figures/):
  toxicity_descriptive.csv
  toxicity_descriptive_table.{png,pdf}
"""
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

# Per-trial extracted data. None / 'NR' = not reported in source available.
# Values are stored as the source presented them (% or n/N).
TRIALS_TOX = [
    {
        'trial': 'POP-RT',
        'reference': 'Murthy 2021, J Clin Oncol',
        'era': 'IMRT',
        'scale': 'RTOG late adverse events',
        'timing': 'Late only',
        'n_wprt': 110, 'n_port': 112,
        # Late, RTOG grade
        'GU_G2plus':  {'wprt': '20.0% (22/110)', 'port': '8.9% (10/112)',  'p': '0.02'},
        'GU_G3plus':  {'wprt': '1.8% (2/110)',   'port': '1.8% (2/112)',   'p': '—'},
        'GI_G2plus':  {'wprt': '8.2% (9/110)',   'port': '4.5% (5/112)',   'p': '0.28'},
        'GI_G3plus':  {'wprt': '1.8% (2/110)',   'port': '0% (0/112)',     'p': '—'},
        'notes': 'Only late adverse events tabulated in the long-term report. RTOG scale.',
    },
    {
        'trial': 'GETUG-01',
        'reference': 'Pommier 2016, Int J Radiat Oncol Biol Phys',
        'era': '2D / 3D-CRT',
        'scale': 'NCI-CTC v2 / RTOG (original 2007 report)',
        'timing': 'Acute + late (original 2007 report)',
        'n_wprt': 178, 'n_port': 178,
        'GU_G2plus':  {'wprt': 'See Pommier 2007', 'port': 'See Pommier 2007', 'p': 'NR'},
        'GU_G3plus':  {'wprt': 'See Pommier 2007', 'port': 'See Pommier 2007', 'p': 'NR'},
        'GI_G2plus':  {'wprt': 'See Pommier 2007', 'port': 'See Pommier 2007', 'p': 'NR'},
        'GI_G3plus':  {'wprt': 'See Pommier 2007', 'port': 'See Pommier 2007', 'p': 'NR'},
        'notes': 'The long-term update does not update adverse events, referring to the original Pommier 2007 (J Clin Oncol) report. Direction in 2007: acute ≥G2 GI higher with WPRT (2D-era, large field); late adverse events similar between arms.',
    },
    {
        'trial': 'RTOG 9413',
        'reference': 'Roach 2018, Lancet Oncol',
        'era': '3D-CRT',
        'scale': 'CTCAE',
        'timing': 'Late ≥G3 only (collapsed across hormone-timing)',
        'n_wprt': 633, 'n_port': 628,
        'GU_G2plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'GU_G3plus':  {'wprt': '6.3% (40/633)', 'port': '4.9% (31/628)', 'p': 'NS'},
        'GI_G2plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'GI_G3plus':  {'wprt': '5.1% (32/633)', 'port': '1.9% (12/628)', 'p': 'reported significant'},
        'notes': 'Numbers collapse the field comparison across hormone-timing (factorial design). 4-arm breakdown for ≥G3 GI: NHT+WPRT 7%, NHT+PORT 2%, WPRT+AHT 3%, PORT+AHT 2%. No acute adverse events in this long-term update.',
    },
    {
        'trial': 'PEACE-2',
        'reference': 'Blanchard 2026, ASCO GU / ESTRO LBA',
        'era': 'IMRT',
        'scale': 'CTCAE',
        'timing': 'Not yet reported',
        'n_wprt': 381, 'n_port': 380,
        'GU_G2plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'GU_G3plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'GI_G2plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'GI_G3plus':  {'wprt': 'NR', 'port': 'NR', 'p': 'NR'},
        'notes': 'Per-arm adverse events not yet reported. Trial-level data indicate ≥G3 GI/GU events are very low (~1–2%) with modern IMRT. Cabazitaxel co-intervention precludes a clean pelvic-RT-attributable adverse-events estimate.',
    },
    {
        'trial': 'RTOG 0924',
        'reference': 'Roach 2025, ASTRO LBA',
        'era': 'IMRT',
        'scale': 'CTCAE v4',
        'timing': 'Late (urinary frequency item)',
        'n_wprt': 1245, 'n_port': 1228,
        'GU_G2plus':  {'wprt': '27.2%', 'port': '25.3%', 'p': 'NS'},
        'GU_G3plus':  {'wprt': '0.7%',  'port': '0.5%',  'p': 'NS'},
        'GI_G2plus':  {'wprt': '22.1%', 'port': '18.1%', 'p': '0.0085 (one-sided)'},
        'GI_G3plus':  {'wprt': '3.3%',  'port': '3.4%',  'p': 'NS'},
        'notes': 'GU figures refer to the urinary-frequency item (only GU item publicly reported to date).',
    },
]

DOMAINS = ['GU_G2plus', 'GU_G3plus', 'GI_G2plus', 'GI_G3plus']
DOMAIN_HEAD = {'GU_G2plus': 'GU ≥G2', 'GU_G3plus': 'GU ≥G3',
               'GI_G2plus': 'GI ≥G2', 'GI_G3plus': 'GI ≥G3'}


# ── console + CSV ────────────────────────────────────────────────────────────
def report():
    rows = [['trial', 'reference', 'era', 'scale', 'timing', 'n_WPRT', 'n_PORT',
             'GU≥G2 WPRT', 'GU≥G2 PORT', 'GU≥G2 p',
             'GU≥G3 WPRT', 'GU≥G3 PORT', 'GU≥G3 p',
             'GI≥G2 WPRT', 'GI≥G2 PORT', 'GI≥G2 p',
             'GI≥G3 WPRT', 'GI≥G3 PORT', 'GI≥G3 p',
             'notes']]
    print('=' * 90)
    print('DESCRIPTIVE ADVERSE EVENTS — pelvic vs prostate-only RT (NOT pooled, see L1 limitation)')
    print('=' * 90)
    for t in TRIALS_TOX:
        print(f"\n{t['trial']}  [{t['era']}, {t['scale']}, {t['timing']}]")
        print(f"  source: {t['reference']}")
        for d in DOMAINS:
            v = t[d]
            print(f"  {DOMAIN_HEAD[d]:<7}  WPRT {v['wprt']:<22}  PORT {v['port']:<22}  p={v['p']}")
        print(f"  note: {t['notes']}")
        rows.append([t['trial'], t['reference'], t['era'], t['scale'], t['timing'],
                     t['n_wprt'], t['n_port'],
                     t['GU_G2plus']['wprt'], t['GU_G2plus']['port'], t['GU_G2plus']['p'],
                     t['GU_G3plus']['wprt'], t['GU_G3plus']['port'], t['GU_G3plus']['p'],
                     t['GI_G2plus']['wprt'], t['GI_G2plus']['port'], t['GI_G2plus']['p'],
                     t['GI_G3plus']['wprt'], t['GI_G3plus']['port'], t['GI_G3plus']['p'],
                     t['notes']])
    p = os.path.join(OUT, 'toxicity_descriptive.csv')
    with open(p, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f'\nSaved: {p}')


# ── Figure: descriptive table ────────────────────────────────────────────────
def _cell_color(v):
    if 'NR' in v or 'See ' in v:
        return '#ECECEC'
    return 'white'


def figure():
    headers = ['Trial', 'Era', 'Scale', 'Timing',
               'GU ≥G2\nPORT  /  WPRT', 'GU ≥G3\nPORT  /  WPRT',
               'GI ≥G2\nPORT  /  WPRT', 'GI ≥G3\nPORT  /  WPRT', 'Notes']
    col_w = [1.7, 1.4, 2.4, 2.3, 2.3, 2.3, 2.3, 2.3, 4.6]
    total_w = sum(col_w)
    n = len(TRIALS_TOX)
    row_h = 2.0  # tall rows for multi-line notes
    fig_w = 22; fig_h = 1.7 + row_h * n
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, n * row_h + 1.4)
    ax.axis('off')

    # Header
    x = 0
    for h, w in zip(headers, col_w):
        ax.add_patch(Rectangle((x, n * row_h + 0.4), w, 1.0,
                               facecolor='#1F4E79', edgecolor='white', lw=1.2))
        ax.text(x + w / 2, n * row_h + 0.9, h, ha='center', va='center',
                fontsize=8.4, fontweight='bold', color='white')
        x += w

    # Body
    for ri, t in enumerate(TRIALS_TOX):
        y = (n - ri - 1) * row_h + 0.4
        h = row_h
        x = 0
        # Trial
        ax.add_patch(Rectangle((x, y), col_w[0], h, facecolor='#F4F6F8', edgecolor='white', lw=1.2))
        ax.text(x + 0.12, y + h - 0.45, t['trial'], ha='left', va='top',
                fontsize=9, fontweight='bold', color='#111')
        ax.text(x + 0.12, y + h - 1.05, t['reference'], ha='left', va='top',
                fontsize=6.8, style='italic', color='#666', wrap=True)
        x += col_w[0]
        # Era
        ax.add_patch(Rectangle((x, y), col_w[1], h, facecolor='white', edgecolor='white', lw=1.2))
        ax.text(x + col_w[1] / 2, y + h / 2, t['era'], ha='center', va='center', fontsize=8.4)
        x += col_w[1]
        # Scale
        ax.add_patch(Rectangle((x, y), col_w[2], h, facecolor='white', edgecolor='white', lw=1.2))
        ax.text(x + col_w[2] / 2, y + h / 2, t['scale'], ha='center', va='center', fontsize=7.8, wrap=True)
        x += col_w[2]
        # Timing
        ax.add_patch(Rectangle((x, y), col_w[3], h, facecolor='white', edgecolor='white', lw=1.2))
        ax.text(x + col_w[3] / 2, y + h / 2, t['timing'], ha='center', va='center', fontsize=7.8, wrap=True)
        x += col_w[3]
        # Domains (4 cells)
        for d, dw in zip(DOMAINS, col_w[4:8]):
            v = t[d]
            ax.add_patch(Rectangle((x, y), dw, h,
                                   facecolor=_cell_color(v['port']) if _cell_color(v['port']) != 'white'
                                   else _cell_color(v['wprt']), edgecolor='white', lw=1.2))
            line1 = f"PORT: {v['port']}"
            line2 = f"WPRT: {v['wprt']}"
            line3 = f"p={v['p']}" if v['p'] not in ('NR', '—', '') else ''
            ax.text(x + dw / 2, y + h - 0.55, line1, ha='center', va='center', fontsize=7.6)
            ax.text(x + dw / 2, y + h - 1.05, line2, ha='center', va='center', fontsize=7.6, fontweight='bold')
            if line3:
                ax.text(x + dw / 2, y + h - 1.55, line3, ha='center', va='center', fontsize=7.0, color='#555')
            x += dw
        # Notes
        ax.add_patch(Rectangle((x, y), col_w[8], h, facecolor='#FFF8E1', edgecolor='white', lw=1.2))
        # Manual wrap of long notes to ~70 chars
        note = t['notes']
        wrapped = []
        words = note.split(); cur = ''
        for w in words:
            if len(cur) + len(w) + 1 > 60:
                wrapped.append(cur); cur = w
            else:
                cur = (cur + ' ' + w).strip()
        if cur:
            wrapped.append(cur)
        ax.text(x + 0.12, y + h - 0.3, '\n'.join(wrapped[:5]), ha='left', va='top',
                fontsize=6.8, color='#333', wrap=True)

    fig.suptitle('Adverse events by trial — descriptive table (NOT pooled): pelvic vs prostate-only RT',
                 y=0.99, fontsize=13, fontweight='bold')
    fig.text(0.02, 0.02,
             'Numbers reported verbatim as published or presented (PORT % / WPRT % or n/N; p where reported). '
             'NOT pooled across trials: 2D/3D-CRT vs IMRT delivery, different CTCAE versions, different acute/late timepoints — see Limitations L1. '
             '"NR" = not reported in source; grey cells = not extractable from available source.',
             fontsize=7.4, color='#555')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'toxicity_descriptive_table.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: toxicity_descriptive_table.{{png,pdf}}')


if __name__ == '__main__':
    report()
    figure()
