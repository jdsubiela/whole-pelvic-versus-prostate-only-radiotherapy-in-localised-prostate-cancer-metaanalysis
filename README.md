# Whole-pelvic versus prostate-only radiotherapy in localised prostate cancer — meta-analysis

Reproducible analysis code for the systematic review and meta-analysis:

> **Whole-pelvic versus prostate-only radiotherapy in clinically node-negative high-risk localised prostate adenocarcinoma: a systematic review and meta-analysis with reconstructed individual-patient data.**

- Open Science Framework Project: [osf.io/ajhek](https://osf.io/ajhek/)
- Open Science Framework Registration: [osf.io/7kfdp](https://osf.io/7kfdp/)
- PROSPERO: [CRD420261433121](https://www.crd.york.ac.uk/PROSPERO/view/CRD420261433121)

---

## What this repository contains

This repository releases the **analysis pipeline only** (Python). Per-trial digitised Kaplan-Meier coordinates and reconstructed pseudo-individual-patient datasets are included so the analyses can be re-run end to end. Source PDFs of the included trials are **not** redistributed (copyright); each is cited in the methods so they can be obtained from the original publishers.

| Path | What's inside |
|---|---|
| `analysis/pairwise_ma.py` | Aggregate-data random-effects meta-analysis (DerSimonian-Laird τ² + modified Hartung-Knapp 95% CI) of the five trials across OS, biochemical/progression-free and metastasis-free outcomes; forest plot generator. |
| `analysis/sensitivity_analysis.py` | Six pre-specified sensitivity analyses: leave-one-out, full-text trials only, excluding POP-RT, alternative τ² estimators (Paule-Mandel, REML), fixed-effect, and classical Wald confidence interval; robustness + leave-one-out forest plots. |
| `analysis/transitivity_dashboard.py` | Trial-level transitivity dashboard (PSMA-PET staging, common-iliac coverage, IMRT use, single-centre status, ADT duration, Roach-formula nodal-risk distribution). |
| `analysis/toxicity_table.py` | Descriptive (NOT pooled) per-trial adverse-events table (late ≥G2 / ≥G3 GU and GI). |
| `analysis/grade_assessment.py` | GRADE Summary-of-Findings table and the per-trial RoB 2 traffic-light figure. |
| `analysis/figures/` | Output figures and CSVs of the aggregate-data layer (forest plots, sensitivity panels, transitivity dashboard, GRADE SoF, RoB 2 traffic-light, descriptive adverse-events table). |
| `analysis/ipd/guyot.py` | Vendored implementation of the Guyot algorithm for pseudo-individual-patient-data reconstruction from a digitised Kaplan-Meier curve (Guyot et al., BMC Med Res Methodol 2012;12:9). |
| `analysis/ipd/trials_ipd.py` | Per-trial × per-outcome registry of digitised KM input files, number-at-risk tables and arm totals. |
| `analysis/ipd/digitized/` | Per-curve digitised (time, survival) CSVs — one per trial × outcome × arm. |
| `analysis/ipd/km_reconstruction.py` | Runs Guyot reconstruction per trial × outcome, fits per-trial Cox PH, computes RMST, validates against the published HR by mutual 95%-CI containment. |
| `analysis/ipd/reconstructed/` | Reconstructed pseudo-individual-patient data per trial × outcome + `reconstruction_validation.csv`. |
| `analysis/ipd/pool_ipd.py` | Two-stage random-effects meta-analysis of the reconstructed pseudo-IPD; Schoenfeld proportional-hazards diagnostic; piecewise hazard ratio at ≤60 vs >60 months. |
| `analysis/ipd/km_pooled.py` | Pooled reconstructed Kaplan-Meier curves (linearly-interpolated, with 95% CI bands). |
| `analysis/ipd/km_overlay.py` | QC overlay: reconstructed KM (line) versus digitised input points (dots) per trial × outcome. |
| `analysis/figures_ipd/` | Output figures of the reconstructed-IPD layer. |

## Reproducing the analyses

```bash
# Aggregate-data meta-analysis + sensitivities + diagnostics
cd analysis
python3 pairwise_ma.py
python3 sensitivity_analysis.py
python3 transitivity_dashboard.py
python3 toxicity_table.py
python3 grade_assessment.py

# Reconstructed-IPD pipeline
cd ipd
python3 km_reconstruction.py
python3 pool_ipd.py
python3 km_pooled.py
python3 km_overlay.py
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `lifelines`. Tested on Python 3.14.

## Methods summary

- **PRISMA 2020** + Vickers SR/MA reporting standards.
- Phase-3 RCTs of whole-pelvic vs prostate-only radiotherapy in cN0 high-risk localised prostate cancer.
- Five trials included (n = 5,177): RTOG 9413, GETUG-01, POP-RT, PEACE-2, RTOG 0924.
- **Aggregate-data meta-analysis**: random-effects, DerSimonian-Laird τ² + modified Hartung-Knapp 95% CI.
- **Reconstructed-IPD meta-analysis**: Guyot algorithm validated by mutual 95%-confidence-interval containment against published hazard ratios; two-stage random-effects pooling.
- **Risk of bias**: Cochrane RoB 2, one global rating per trial.
- **Certainty of evidence**: GRADE per outcome.
- **Adverse events**: descriptive per trial; not pooled (structural heterogeneity in delivery technique and CTCAE version).
- Six pre-specified sensitivity analyses: leave-one-out, full-text trials only, excluding POP-RT, alternative τ² estimators (Paule-Mandel and REML), fixed-effect, classical Wald CI.

## Citation

When the manuscript is published, cite the published paper. To cite this code release before publication, use the OSF Registration DOI: pending DataCite minting at `osf.io/7kfdp`.

## Licence

MIT (see `LICENSE`). Digitised Kaplan-Meier coordinates and reconstructed pseudo-IPD in `analysis/ipd/` are released under CC0 1.0 for reproducibility audit; each carries its source trial citation.

## Authors

- **José Daniel Subiela, MD, PhD** — Department of Urology, Hospital Universitario Ramón y Cajal, Madrid, Spain. *Corresponding author; co-first.* jdsubiela [at] gmail.com.
- **Elías Gomis Sellés, MD, PhD** — Department of Radiation Oncology, Hospital Universitario y Politécnico La Fe (Valencia) / GenesisCare Valencia / Universidad Internacional de La Rioja (UNIR), Spain. *Co-first author.*
- **Júlia Aumatell, MD, PhD** — Department of Urology, Hospital Universitario Rey Juan Carlos, Universidad Rey Juan Carlos, Móstoles, Madrid, Spain.
- **Fernando López Campos, MD, PhD** — Department of Radiation Oncology, Hospital Universitario Ramón y Cajal, Madrid, Spain.
- **David López-Curtis, MD** — Department of Urology, Hospital Universitario Ramón y Cajal, Madrid, Spain.
- **Victoria Vera, MD** — *Affiliation pending.*
- **Juan Gómez Rivas, MD, PhD** — Department of Urology, Hospital Clínico San Carlos, Madrid, Spain.
- **Luis San-José Manso, MD, PhD** — Department of Urology, Hospital Universitario de La Princesa, IIS Princesa, Universidad Autónoma de Madrid, Madrid, Spain.
- **Francesco Barletta, MD** — Division of Experimental Oncology, Department of Urology, Urological Research Institute, Vita-Salute San Raffaele University, Milan, Italy.
- **Òscar Buisan, MD, PhD** — Department of Urology, Hospital Universitari de Bellvitge, L'Hospitalet de Llobregat, Barcelona, Spain. *Co-senior author.*
- **Felipe Couñago, MD, PhD** — Department of Radiation Oncology, San Francisco de Asís Hospital / La Milagrosa Hospital, Madrid; National Chair of Research, GenesisCare, Madrid, Spain. *Co-senior author.*

Additional international co-authors and the full byline + numbered affiliations live in [`manuscript/authors.md`](https://osf.io/ajhek/) (OSF Project) and in the PROSPERO record above.

## What is NOT in this repository

- PDF reprints of the included trials (copyright). Original publications are cited in the manuscript.
- Source congress slides and personal working materials.
- The screening, extraction, risk-of-bias, protocol, PROSPERO and OSF working files — those are deposited at the OSF Project ([osf.io/ajhek](https://osf.io/ajhek/)).
