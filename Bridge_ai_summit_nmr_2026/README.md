# Metabolic Disease Screening — Honest experiments (real cohorts, held-out PR-AUC)

BDI Hackathon 2026 | PhenoInsure Tech | Calvin (KMITL AI Engineering) | env `superaiss6`

Three cleanly-separated, poster-ready experiments. Each follows
**`ข้อมูลจากไหน → ใช้วิธีอะไร → ผลยังไง`** (Data source → Method → Results), and every
PR figure writes its **data provenance** and **evaluation method** directly on the plot
so a reviewer never has to ask "measured how?".

Every XGBoost model uses **monotonic constraints assigned from disease/metabolomics
literature**, not mined from each dataset's own correlations. A feature is only constrained
when there's a citable, literature-backed reason for its direction; anything uncertain —
including every unidentified NMR chemical-shift bucket, which has no chemical identity to
reason about at all — is left unconstrained rather than guessed. Headline metric is
**PR-AUC (average precision)** rather than ROC-AUC, since it's the more honest read under the
class imbalance present in Experiments B and C.

| Notebook | Disease | Data (real) | Eval | Result |
|---|---|---|---|---|
| [`exp_obesity_nmr_mtbls242.ipynb`](exp_obesity_nmr_mtbls242.ipynb) | Obesity (NMR) | MTBLS242 serum ¹H-NMR, n=177 | stratified 5-fold CV | **PR-AUC 0.984 ± 0.014** |
| [`exp_t2dm_clinical_nhanes.ipynb`](exp_t2dm_clinical_nhanes.ipynb) | T2DM (clinical) | NHANES 2013–14 Cycle H, n=6,113 | stratified 5-fold CV | **PR-AUC 0.416 ± 0.030** (baseline 0.143) |
| [`exp_t2dm_nmr_mtbls1.ipynb`](exp_t2dm_nmr_mtbls1.ipynb) | T2DM (NMR) | MTBLS1 urine ¹H-NMR, n=132 (42 subj) | **subject-grouped** 5-fold CV | **PR-AUC 0.976 ± 0.022** |

Figures live in [`poster_figures/`](poster_figures/) (300 DPI, print-ready) and are also
embedded inline in each notebook. `run_approach6.py` is the script form of Experiment A;
its CSV outputs are in [`results/`](results/).

**Data:** the notebooks read the source cohorts (MTBLS242, MTBLS1, NHANES) by absolute path
from the parent BDI project; the raw data is **not** included in this repo. This repo holds
the analysis notebooks, figures, and documentation.

**📊 For the poster:** [`poster_figures/`](poster_figures/) holds print-ready 300 DPI
figures + [`POSTER.md`](poster_figures/POSTER.md) documenting every experiment and figure
in detail (data source, method, results, captions, honest caveats, judge Q&A prep).

## Run

```bash
conda run -n superaiss6 jupyter nbconvert --to notebook --execute --inplace \
  exp_obesity_nmr_mtbls242.ipynb exp_t2dm_clinical_nhanes.ipynb
# or open them in Jupyter with the superaiss6 kernel
```

## Design principles (why these are defensible)

1. **Real cohorts only.** Obesity uses measured MTBLS242 NMR; T2DM uses the NHANES
   national survey. No simulated spectra. A "leak-free" pipeline on *synthetic* data would
   only be **pipeline validation**, not evidence a biomarker is real — so we don't show that.
2. **Held-out PR-AUC, stated on the figure.** Every PR curve is stratified 5-fold cross-validation
   (mean ± std), annotated as held-out — never training performance. PR-AUC (average precision) is
   the headline, not permutation/feature importance (which shows *which* feature matters, not *how
   well* it predicts).
3. **Leakage-free labels.** T2DM excludes HbA1c **and** glucose (they define the label); an
   `assert` in the notebook enforces it. Honest PR-AUC ≈ 0.42 (vs a 0.14 prevalence baseline), not a
   circular ~1.0.
4. **Monotonic constraints from literature, not from the data.** Each constrained feature's
   direction is a citable disease/metabolomics-literature finding — never the sign of that
   feature's own correlation with this specific sample. A feature is only constrained when there's
   a defensible reason; everything else (including all 105 unidentified NMR chemical-shift buckets
   in Experiment C, which have no chemical identity to reason about) is left unconstrained. This is
   a smaller, more conservative constraint set than a purely data-driven rule would produce, and
   every non-zero direction can be traced to a named source below.

## Experiment A — Obesity (MTBLS242 NMR)

- **Data:** bariatric cohort; **obese = `preop` (n=106)** vs **healthy = `12 months after
  surgery` (n=71)**; 21 metabolites, `log1p`. Mid time points (3/6/9 mo) excluded.
- **Method:** XGBoost (`max_depth=3`) with monotonic constraints from obesity/insulin-resistance
  metabolomics literature (Newgard 2009, Wang 2011 *Nat Med*, Würtz et al.), stratified 5-fold
  CV, headline PR-AUC; SHAP for interpretation. **11 of 21** metabolites are constrained: BCAAs
  L-valine/L-leucine/L-allo-Isoleucine, aromatic amino acids L-tyrosine/D-phenylalanine,
  L-alanine, lipoproteins, L-Lactic acid ↑; glycine, L-glutamine, histidine ↓ (robust, replicated
  inverse markers). The rest — including the three known **exogenous artifacts** below — are
  deliberately left unconstrained.
- **Honest caveats:** preop vs 12mo are cohort *extremes* (high PR-AUC expected). Top SHAP
  drivers `dimethyl sulfone / isopropanol / methanol` are exogenous — **isopropanol is a
  surgical skin-prep contaminant** — so they're intentionally *not* constrained: doing so would
  encode a sample-context confound as if it were obesity biology. Real obesity markers
  (`L-valine`/BCAA, `lipoproteins`, `L-tyrosine`, lactate) rank just below. Patient IDs aren't
  recoverable → same-patient fold leakage can't be fully excluded.

## Experiment B — Type-2 Diabetes (NHANES clinical)

- **Data:** NHANES 2013–14, 7 files merged on `SEQN`; adults ≥18; label = self-reported
  diagnosis **or** HbA1c ≥ 6.5%; prevalence 14.3%.
- **Method:** 13 routine features (age, sex, race, BP, cholesterol/HDL, TC/HDL, creatinine,
  albumin, BUN, triglycerides, uric acid) — **HbA1c + glucose excluded**; XGBoost with
  monotonic constraints from metabolic-syndrome/T2DM risk literature, 5-fold CV, headline
  PR-AUC. **11 of 13** features are constrained: age, systolic + diastolic BP, TC/HDL ratio,
  creatinine, BUN, triglycerides, uric acid, total cholesterol ↑; HDL, albumin ↓. Sex and race
  are left unconstrained — nominal demographic categories, not a dose-response a monotonic
  constraint can meaningfully encode.
- **Honest caveat:** PR-AUC ≈ 0.42 against a 0.14 prevalence baseline (~2.9x lift) is the
  genuine screening number from non-diagnostic signals. This is *lower* than a purely
  data-driven (correlation-sign) constraint set would score (~0.46) — the honest cost of
  choosing directions for defensibility rather than fit to this particular sample.

## Experiment C — Type-2 Diabetes (MTBLS1 urine NMR)

- **Data:** MTBLS1, *"urinary changes in type 2 diabetes"*; **48 diabetes vs 84 control** samples
  from 42 people; 220 NMR variables. Same person has NMR **and** label (a real merge). Glucose
  region excluded from the NMR — no shortcut.
- **Method:** XGBoost (`max_depth=3`) with monotonic constraints from urinary diabetes-
  metabolomics literature, constraining **32 of 220** variables (the other 188, including all
  105 unidentified chemical-shift buckets, are left unconstrained — no chemical identity to
  reason about, or no confident literature direction): hippurate, citrate ↓ (this dataset's own
  published finding, Salek et al. 2007); ketone bodies, α-hydroxybutyrate (Gall et al. 2010),
  allantoin, N-acetyl glycoproteins ↑ (high confidence); a broader aminoaciduria amino-acid
  pattern (BCAAs, aromatic amino acids, alanine, glutamine/glutamate, pyruvate, lactate) ↑
  (moderate confidence); indoxyl sulfate/sulphate ↑ (lower confidence, uremic-toxin/renal
  literature). **StratifiedGroupKFold** on subjects reconstructed from consecutive sample-number
  runs (controls → 12 clean blocks of 7) so no person spans train/test. Headline PR-AUC.
- **Result:** subject-grouped **PR-AUC 0.976 ± 0.022** (naive 0.969 → leakage negligible). SHAP
  recovers BCAAs (2-oxoisovalerate, isoleucine), hippurate region, N-methylnicotinamide, allantoin.
- **⚠ Known limitation:** every control sample name has prefix `ADG19007u` and every diabetes
  sample has prefix `ADG10003u` — the label is **100% predictable from acquisition batch alone**.
  Subject-grouped CV only rules out same-*person* leakage; it does nothing about same-*batch*
  leakage, since every control subject sits in one batch and every diabetic subject in the other.
  **This PR-AUC is a ceiling, not confirmed biology** — it cannot be distinguished from a
  scan-session/collection artifact with this cohort alone.
- **Other caveats:** small (42 people, 26 subject-groups); **urine** not serum; diabetic subject
  boundaries approximate; gender is imbalanced between groups (controls skew male) and urine
  metabolome is sex-dependent.
