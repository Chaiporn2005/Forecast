"""
Approach 6 - Obesity Screener (MTBLS242, NMR serum -> obese vs healthy)
=======================================================================
BDI Hackathon 2026 | PhenoInsure Tech

MTBLS242 is a bariatric (weight-loss) surgery cohort. We use the time-point
factor as a natural obesity signal and frame a binary classification:

    obese   (class 1) = preop                     (before operation, n=106)
    healthy (class 0) = 12 months after surgery   (weight-normalized, n=71)

The intermediate 3/6/9-month samples (patients mid-weight-loss, neither clearly
obese nor healthy) are excluded to sharpen the obese-vs-healthy contrast.

Model      : XGBoost with literature-informed monotonic constraints,
             stratified 5-fold cross-validation
Headline   : pr_auc_cv.png  -> per-fold PR + mean PR (+/-1 std band) + PR-AUC
Interpret  : shap_summary.png -> which NMR metabolites drive the obese call

Honesty note: preop vs 12mo are the two extremes of the cohort, so expect a
very high PR-AUC. The CV +/-std band shows the real variance. This measures
how separable the extremes are, not proof of general (population) screening
power.

Run:  conda run -n superaiss6 python run_approach6.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Highly_Imagination", "real_data"))
NPZ_PATH = os.path.join(DATA_DIR, "MTBLS242_parsed.npz")
SAMPLES_PATH = os.path.join(DATA_DIR, "MTBLS242_samples.txt")

OBESE_TIMEPOINTS = {"preop"}                       # class 1
HEALTHY_TIMEPOINTS = {"12 months after surgery"}   # class 0
# To use "healthy = all post-op" instead, set:
# HEALTHY_TIMEPOINTS = {"3 months after surgery", "6 months after surgery",
#                       "9 months after surgery", "12 months after surgery"}

N_SPLITS = 5
SEED = 42

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,          # shallow: only 177 samples, guard against overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# --------------------------------------------------------------------------- #
# Monotonic direction per metabolite, from obesity/insulin-resistance serum
# metabolomics literature (not computed from this dataset's own correlations,
# so it can't just re-encode sampling noise or a surgical-context artifact).
#
#   +1 (obesity/preop-associated):
#     BCAAs L-valine/L-leucine/L-allo-Isoleucine (Newgard 2009, Wang 2011 NatMed);
#     aromatic amino acids L-tyrosine/D-phenylalanine (same BCAA+AAA signature);
#     L-alanine (gluconeogenic, rises with insulin resistance); lipoproteins
#     (NMR particle burden, dyslipidemia); L-Lactic acid (impaired oxidative
#     metabolism in insulin resistance).
#   -1 (healthy/12mo-postop-associated):
#     glycine, L-glutamine, histidine — robust, repeatedly-replicated INVERSE
#     markers of obesity/insulin resistance (Wurtz et al.).
#   0 (left unconstrained — no confident literature direction, or known to be
#     an artifact rather than biology):
#     Pyruvic acid, citrate, acetoacetate, (R)-3-Hydroxybutyric acid, acetate,
#     hypoxanthine, creatinine — mixed/insufficient evidence, and creatinine is
#     confounded by the surgical-vs-outpatient sample context here.
#     Dimethyl sulfone, isopropanol, methanol — these are the notebook's own
#     flagged EXOGENOUS artifacts (isopropanol = surgical skin-prep contaminant),
#     not obesity biology, so constraining their direction would just encode
#     the confound as if it were a biological finding.
# --------------------------------------------------------------------------- #
MONOTONE_DIRECTIONS = {
    "L-valine": 1,
    "L-leucine": 1,
    "L-allo-Isoleucine": 1,
    "L-tyrosine": 1,
    "D-phenylalanine": 1,
    "L-alanine": 1,
    "lipoproteins": 1,
    "L-Lactic acid": 1,
    "glycine": -1,
    "L-glutamine": -1,
    "histidine": -1,
}


def compute_monotone_constraints(df):
    return {col: MONOTONE_DIRECTIONS.get(col, 0) for col in df.columns}


# --------------------------------------------------------------------------- #
# 1. Load data + attach time points
# --------------------------------------------------------------------------- #
def load_dataset():
    d = np.load(NPZ_PATH, allow_pickle=True)
    metabolites = [str(m) for m in d["metabolites"]]
    conc = d["concentrations"].astype(float)          # (465, 21)
    sample_ids = [str(s) for s in d["sample_ids"]]

    # sample_name -> time point
    tp_map = {}
    with open(SAMPLES_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tp_map[row["Sample Name"]] = row["Factor Value[time point]"]
    timepoints = np.array([tp_map.get(s, "MISSING") for s in sample_ids])

    # keep only obese/healthy anchor time points
    is_obese = np.isin(timepoints, list(OBESE_TIMEPOINTS))
    is_healthy = np.isin(timepoints, list(HEALTHY_TIMEPOINTS))
    keep = is_obese | is_healthy

    X = conc[keep]
    y = is_obese[keep].astype(int)                    # 1=obese(preop), 0=healthy
    X_log = np.log1p(X)                               # NMR abundances -> log scale

    df = pd.DataFrame(X_log, columns=metabolites)
    print(f"[data] samples kept: {keep.sum()}  (obese={y.sum()}, healthy={(y==0).sum()})")
    print(f"[data] features: {len(metabolites)} NMR metabolites")
    return df, y, metabolites


# --------------------------------------------------------------------------- #
# 2. Stratified 5-fold CV -> collect PR curve per fold (headline metric: PR-AUC,
#    i.e. average precision, more informative than ROC-AUC under class imbalance)
# --------------------------------------------------------------------------- #
def cross_validate(df, y, monotone_constraints):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    mean_recall = np.linspace(0, 1, 200)
    precisions, pr_aucs, fold_rows = [], [], []
    params = dict(XGB_PARAMS, monotone_constraints=monotone_constraints)

    for fold, (tr, te) in enumerate(skf.split(df, y), start=1):
        model = XGBClassifier(**params)
        model.fit(df.iloc[tr], y[tr])
        prob = model.predict_proba(df.iloc[te])[:, 1]

        fold_pr_auc = average_precision_score(y[te], prob)
        pr_aucs.append(fold_pr_auc)

        precision, recall, _ = precision_recall_curve(y[te], prob)
        # precision_recall_curve returns recall descending -> reverse to ascending for interp
        interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp)

        fold_rows.append({"fold": fold, "n_test": len(te), "pr_auc": fold_pr_auc})
        print(f"[cv] fold {fold}: n_test={len(te):3d}  PR-AUC={fold_pr_auc:.4f}")

    return mean_recall, np.array(precisions), np.array(pr_aucs), pd.DataFrame(fold_rows)


# --------------------------------------------------------------------------- #
# 3. Plot the Precision-Recall curve  (the headline deliverable)
# --------------------------------------------------------------------------- #
def plot_pr(mean_recall, precisions, pr_aucs, n_obese, n_healthy, out_path):
    mean_precision = precisions.mean(axis=0)
    mean_pr_auc = pr_aucs.mean()
    std_pr_auc = pr_aucs.std()
    std_precision = precisions.std(axis=0)
    upper = np.minimum(mean_precision + std_precision, 1)
    lower = np.maximum(mean_precision - std_precision, 0)
    chance = n_obese / (n_obese + n_healthy)   # baseline precision = prevalence

    fig, ax = plt.subplots(figsize=(7.6, 7))

    for i, (prec, a) in enumerate(zip(precisions, pr_aucs), start=1):
        ax.plot(mean_recall, prec, lw=1, alpha=0.35,
                label=f"Fold {i}  (PR-AUC = {a:.3f})")

    ax.plot(mean_recall, mean_precision, color="#b2182b", lw=3,
            label=f"Mean PR  (PR-AUC = {mean_pr_auc:.3f} $\\pm$ {std_pr_auc:.3f})")
    ax.fill_between(mean_recall, lower, upper, color="#b2182b", alpha=0.18,
                    label=r"$\pm$ 1 std. dev.")
    ax.axhline(chance, linestyle="--", lw=1.2, color="grey",
               label=f"Chance (PR-AUC = {chance:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Approach 6 — Obesity Screener (MTBLS242 NMR)\n"
        f"XGBoost + monotonic constraints, stratified 5-fold CV  |  obese/preop n={n_obese}  vs  "
        f"healthy/12mo n={n_healthy}",
        fontsize=12.5,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}   mean PR-AUC = {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}")
    return mean_pr_auc, std_pr_auc


# --------------------------------------------------------------------------- #
# 4. SHAP -> which metabolites drive the obese prediction
# --------------------------------------------------------------------------- #
def plot_shap(df, y, monotone_constraints, out_path):
    model = XGBClassifier(**dict(XGB_PARAMS, monotone_constraints=monotone_constraints))
    model.fit(df, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    plt.figure()
    shap.summary_plot(shap_values, df, show=False, plot_size=(8, 7))
    plt.title("Approach 6 — SHAP: NMR metabolites driving 'obese' call",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_path}")

    # rank metabolites by mean |SHAP|
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1]
    ranking = pd.DataFrame(
        {"metabolite": df.columns[order], "mean_abs_shap": imp[order]}
    )
    print("[shap] top drivers:")
    print(ranking.head(8).to_string(index=False))
    return ranking


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("Approach 6 - Obesity Screener (MTBLS242 NMR -> obese vs healthy)")
    print("=" * 70)

    df, y, _ = load_dataset()
    n_obese, n_healthy = int(y.sum()), int((y == 0).sum())

    monotone_constraints = compute_monotone_constraints(df)
    n_pos = sum(1 for v in monotone_constraints.values() if v == 1)
    n_neg = sum(1 for v in monotone_constraints.values() if v == -1)
    print(f"[monotone] {n_pos} increasing, {n_neg} decreasing, "
          f"{len(monotone_constraints) - n_pos - n_neg} unconstrained "
          f"(literature-based, see MONOTONE_DIRECTIONS)")

    mean_recall, precisions, pr_aucs, fold_df = cross_validate(df, y, monotone_constraints)

    pr_path = os.path.join(HERE, "pr_auc_cv.png")
    mean_pr_auc, std_pr_auc = plot_pr(mean_recall, precisions, pr_aucs, n_obese, n_healthy, pr_path)

    shap_path = os.path.join(HERE, "shap_summary.png")
    ranking = plot_shap(df, y, monotone_constraints, shap_path)

    # persist results
    results_path = os.path.join(HERE, "approach6_monotonic_pr_auc_results.csv")
    ranking_path = os.path.join(HERE, "approach6_monotonic_shap_ranking.csv")
    fold_df.to_csv(results_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    print(f"[csv] wrote {results_path}")
    print(f"[csv] wrote {ranking_path}")

    print("-" * 70)
    print(f"SUMMARY  mean PR-AUC = {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}  "
          f"(5-fold CV, n={n_obese + n_healthy})")
    print("Results: approach6_monotonic_pr_auc_results.csv")
    print("Figures: pr_auc_cv.png (PR curve), shap_summary.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
