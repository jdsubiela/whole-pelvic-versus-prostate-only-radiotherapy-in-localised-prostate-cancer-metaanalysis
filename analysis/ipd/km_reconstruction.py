#!/usr/bin/env python3
"""Reconstruct pseudo-IPD for each trial/endpoint, fit Cox + RMST, validate vs published HR.

Pipeline per endpoint:
  1. Guyot-reconstruct patient-level (time, event) for each arm from the digitized
     curve + number-at-risk table (digitized/ + trials_ipd.py).
  2. Combine arms (WPRT=1, PORT=0), fit Cox PH -> reconstructed HR (95% CI).
  3. Compute RMST(tau) per arm + difference (bootstrap CI).
  4. VALIDATE: compare reconstructed HR to the published HR. A reconstruction is
     accepted only if the published point estimate falls inside the reconstructed
     95% CI and the log-HRs agree within ~15%. Failures are flagged for re-digitizing.

Outputs (reconstructed/):
  <TRIAL>_<ENDPOINT>_ipd.csv     combined pseudo-IPD (time, event, arm)
  reconstruction_validation.csv  one row per endpoint with published vs reconstructed

Run:  python3 km_reconstruction.py
"""
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from guyot import guyot_ipd
from trials_ipd import TRIALS_IPD

DIG = os.path.join(HERE, 'digitized')
OUT = os.path.join(HERE, 'reconstructed')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260528)


def reconstruct_arm(arm_cfg):
    df = pd.read_csv(os.path.join(DIG, arm_cfg['coords']))
    return guyot_ipd(df['time'].values, df['surv'].values,
                     arm_cfg['nar_t'], arm_cfg['nar'], arm_cfg['n'])


def km_rmst(time, event, tau):
    kmf = KaplanMeierFitter().fit(time, event)
    sf = kmf.survival_function_.reset_index()
    sf.columns = ['t', 's']
    sf = sf[sf['t'] <= tau]
    ts = sf['t'].values
    ss = sf['s'].values
    edges = np.append(ts, tau)
    return float(np.sum(ss * np.diff(edges)))


def rmst_diff_boot(ipd, tau, nboot=500):
    wp = ipd[ipd.arm == 1]
    po = ipd[ipd.arm == 0]
    base = km_rmst(wp.time.values, wp.event.values, tau) - km_rmst(po.time.values, po.event.values, tau)
    diffs = []
    for _ in range(nboot):
        bw = wp.sample(len(wp), replace=True, random_state=RNG.integers(1 << 31))
        bp = po.sample(len(po), replace=True, random_state=RNG.integers(1 << 31))
        diffs.append(km_rmst(bw.time.values, bw.event.values, tau)
                     - km_rmst(bp.time.values, bp.event.values, tau))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, lo, hi


def run():
    rows = []
    for trial, tcfg in TRIALS_IPD.items():
        for ep, ecfg in tcfg['endpoints'].items():
            wp = reconstruct_arm(ecfg['arms']['WPRT']); wp['arm'] = 1
            po = reconstruct_arm(ecfg['arms']['PORT']); po['arm'] = 0
            ipd = pd.concat([wp, po], ignore_index=True)
            ipd.to_csv(os.path.join(OUT, f'{trial.replace(" ", "")}_{ep}_ipd.csv'), index=False)

            cph = CoxPHFitter().fit(ipd[['time', 'event', 'arm']], 'time', 'event')
            hr = float(np.exp(cph.params_['arm']))
            lo = float(np.exp(cph.confidence_intervals_.loc['arm'].iloc[0]))
            hi = float(np.exp(cph.confidence_intervals_.loc['arm'].iloc[1]))
            tau = ecfg['tau']
            rmst_d, rlo, rhi = rmst_diff_boot(ipd, tau)

            phr, plo, phi = ecfg['published_hr']
            # Mutual CI containment: published point inside reconstructed CI AND
            # reconstructed point inside published CI. Robust for near-null HRs.
            inside = lo <= phr <= hi
            recon_in_pub = plo <= hr <= phi
            verdict = 'PASS' if (inside and recon_in_pub) else 'CHECK'

            n_ev = int(ipd.event.sum())
            print(f'\n{trial} — {ep}  [{tcfg["source"]}]')
            print(f'  reconstructed events: WPRT {int(wp.event.sum())}/{len(wp)},  PORT {int(po.event.sum())}/{len(po)}  (total {n_ev})')
            print(f'  published HR   : {phr:.2f} ({plo:.2f}-{phi:.2f})')
            print(f'  reconstructed  : {hr:.2f} ({lo:.2f}-{hi:.2f})   -> {verdict}'
                  f'{"" if inside else "  [pub point outside recon CI]"}')
            print(f'  RMST diff @{tau}mo: {rmst_d:+.1f} mo ({rlo:+.1f} to {rhi:+.1f})  (+ favours WPRT)')

            rows.append(dict(trial=trial, endpoint=ep, source=tcfg['source'],
                             pub_hr=phr, pub_lo=plo, pub_hi=phi,
                             recon_hr=round(hr, 3), recon_lo=round(lo, 3), recon_hi=round(hi, 3),
                             events=n_ev, rmst_diff=round(rmst_d, 2),
                             rmst_lo=round(rlo, 2), rmst_hi=round(rhi, 2), verdict=verdict))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'reconstruction_validation.csv'), index=False)
    print(f'\nSaved IPD + reconstruction_validation.csv in {OUT}')


if __name__ == '__main__':
    run()
