"""Guyot pseudo-IPD reconstruction from digitized Kaplan-Meier curves.

Vendored into this project (self-contained for the supplementary) from the
EV-P vs MVAC vs GC-Durva NMA framework. Algorithm: Guyot P et al., BMC Med Res
Methodol 2012;12:9. Given digitized (time, survival) points, the number-at-risk
table, and the arm total, it apportions events and censoring to recover patient-
level (time, event) data whose KM curve reproduces the published one.
"""
import numpy as np
import pandas as pd


def guyot_ipd(t_surv, s_surv, t_risk, n_risk, n_total):
    t_surv = np.array(t_surv, dtype=float)
    s_surv = np.array(s_surv, dtype=float)
    t_risk = np.array(t_risk, dtype=float)
    n_risk = np.array(n_risk, dtype=float)

    if s_surv[0] > 1:
        s_surv = s_surv / 100.0

    n_int = len(t_surv) - 1
    n_hat = np.zeros(n_int + 1)
    n_hat[0] = n_total
    d = np.zeros(n_int)
    cens = np.zeros(n_int)

    for i in range(n_int):
        if s_surv[i] > 0:
            d[i] = round(n_hat[i] * (1 - s_surv[i + 1] / s_surv[i]))
        idx = np.argmin(np.abs(t_risk - t_surv[i + 1]))
        n_hat_next = n_risk[idx]
        cens[i] = max(0, round(n_hat[i] - d[i] - n_hat_next))
        n_hat[i + 1] = n_hat[i] - d[i] - cens[i]

    times, events = [], []
    for i in range(n_int):
        ne = int(d[i])
        nc = int(cens[i])
        if ne > 0:
            et = np.linspace(t_surv[i] + 0.01, t_surv[i + 1], ne, endpoint=False)
            times.extend(et)
            events.extend([1] * ne)
        if nc > 0:
            ct = np.linspace(t_surv[i] + 0.01, t_surv[i + 1], nc, endpoint=False)
            times.extend(ct)
            events.extend([0] * nc)

    n_remaining = int(n_hat[n_int])
    if n_remaining > 0:
        times.extend([t_surv[-1]] * n_remaining)
        events.extend([0] * n_remaining)

    return pd.DataFrame({'time': times, 'event': events})
