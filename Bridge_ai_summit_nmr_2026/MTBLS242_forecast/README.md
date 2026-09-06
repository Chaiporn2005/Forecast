# MTBLS242 12-month forecast

## Forecast question

Use the **pre-operative serum ¹H-NMR profile** to forecast the **same subject's
serum NMR profile at 12 months after bariatric surgery**.

This is a genuine longitudinal forecast because sample IDs encode a stable
four-digit subject identifier. The public metadata contains 106 pre-operative
samples and 71 twelve-month samples; **71 subjects have both time points**.

This is **not** a forecast of obesity, diabetes, weight loss, or clinical events.
MTBLS242 does not provide a future disease-incidence outcome or BMI/weight-loss
target in the supplied metadata. A clinical risk forecast requires those future
outcomes to be linked at subject level.

Two models are included:

1. `forecast_mtbls242.py` forecasts the continuous 12-month NMR profile.
2. `forecast_responder_pr_auc.py` forecasts a binary **strong metabolic
   response**, enabling PR-AUC evaluation with monotonic constraints.

The binary outcome is fixed before fitting: at least **9 of 11**
literature-directed obesity markers (≥80%) must move toward the expected
healthier direction by 12 months.

## Model and evaluation

- Inputs: 21 pre-operative metabolite abundances, transformed with `log1p`.
- Targets: the same 21 metabolite abundances at 12 months.
- Model: multi-output ridge regression.
- Tuning: ridge penalty selected inside each training fold.
- Evaluation: nested 5-fold cross-validation across 71 distinct subjects.
- Comparators:
  - training-fold mean at 12 months;
  - persistence forecast (future value equals baseline value).
- Metrics: MAE, RMSE, overall R², and per-metabolite R².

Monotonic constraints are not imposed in this first forecast model. The earlier
classification constraints describe a metabolite's direction relative to an
obesity label; they do not establish that a higher baseline concentration must
produce a higher or lower concentration after surgery. Applying those signs to
this different input-output relationship would not be scientifically justified.

For the binary responder model, monotonic signs encode an explicit
"opportunity to improve" assumption: a more obesity-associated baseline profile
cannot reduce the modeled probability of a strong response. This is a modeling
guardrail, not a causal claim, and still requires external validation.

## Run

```powershell
python forecast_mtbls242.py
python forecast_responder_pr_auc.py
```

Only NumPy and pandas are required. Install matplotlib to also create the plot:

```powershell
python -m pip install -r requirements.txt
python forecast_mtbls242.py
```

## Outputs

Files are written to `outputs/`:

- `summary.json` — headline held-out metrics;
- `fold_metrics.csv` — results for each outer fold;
- `metabolite_metrics.csv` — MAE, RMSE, and R² for each metabolite;
- `heldout_predictions.csv` — observed and held-out forecast values;
- `paired_longitudinal_dataset.csv` — the 71 paired subject records;
- `forecast_model.npz` — final model fitted on all paired subjects;
- `forecast_observed_vs_predicted.png` — evaluation figure when matplotlib is installed;
- `forecast_observed_vs_predicted.svg` — dependency-free fallback figure otherwise.

Binary responder results are written to `outputs_pr_auc/`:

- `summary_pr_auc.json` — held-out PR-AUC and monotonicity checks;
- `fold_pr_auc.csv` — PR-AUC and prevalence baseline for each fold;
- `heldout_responder_predictions.csv` — held-out probabilities and outcomes;
- `monotone_coefficients.csv` — constraint and fitted coefficient per feature;
- `monotone_responder_model.npz` — final fitted model;
- `heldout_pr_curve.svg` — held-out precision–recall curve.

## Limitations

- The cohort contains only patients receiving bariatric surgery, so it does not
  estimate outcomes under no surgery or general-population risk.
- With 71 complete subjects, estimates have substantial uncertainty and require
  external validation.
- Forecasting a metabolic profile is not equivalent to forecasting a clinical
  outcome.
- Treatment type and clinical covariates are not included in the public sample
  sheet used here.

## Data provenance

The included ISA-Tab sample sheet and metabolite abundance table were downloaded
from the official EMBL-EBI MetaboLights public MTBLS242 directory.
