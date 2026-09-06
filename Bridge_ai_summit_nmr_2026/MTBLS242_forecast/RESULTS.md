# Initial held-out results

The pipeline found **71 subjects** with both pre-operative and 12-month serum
NMR measurements and evaluated 21 metabolites using nested 5-fold
cross-validation.

| Metric | Result |
|---|---:|
| Forecast MAE (log1p) | 0.326 |
| Training-fold mean baseline MAE | 0.334 |
| Persistence baseline MAE | 0.512 |
| Overall R² across all values | 0.931 |
| Median per-metabolite R² | -0.013 |

## Interpretation

The ridge forecast improves MAE only slightly over predicting the training-fold
12-month mean (about **2.4% lower MAE**) and the median per-metabolite R² is near
zero and slightly negative. The high pooled R² is driven largely by differences
in scale between metabolites, so it should **not** be presented as strong
individual-level forecasting performance.

The honest conclusion is that baseline NMR contains limited signal for the
individual 12-month metabolite profile in this small complete-case cohort. The
model is a valid proof-of-concept pipeline, but it needs clinical covariates,
treatment type, a larger cohort, and external validation before practical use.
