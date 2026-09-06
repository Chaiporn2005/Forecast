"""Repeated nested CV for L2 + clinical predictors + NMR on MTBLS242.

This script is intentionally fail-closed: it does not create synthetic clinical
values.  Supply a real patient-level CSV before running the analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
COMPARISON = DATA / "Monotonic vs L2 vs Elastic Net vs SVM"
sys.path.insert(0, str(COMPARISON))
import run_comparison as base  # noqa: E402
from validate_setup import CATEGORICAL, NUMERIC, REQUIRED, normalize_subject_id  # noqa: E402

RESULTS = ROOT / "Results"
GRAPH = ROOT / "Graph"
REPEATS = 20
FOLDS = 5
L2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
MODELS = ("L2 + Clinical + NMR", "L2 (NMR only)", "Monotonic + L2 (NMR only)")


def load_clinical(path: Path, subject_ids: pd.Series) -> pd.DataFrame:
    """Read and order one row per MTBLS242 subject; never infer missing values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Clinical file not found: {path}. Supply a real CSV with columns: {', '.join(REQUIRED)}"
        )
    df = pd.read_csv(path, dtype={"subject_id": "string"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")
    df = df.copy()
    df["subject_id"] = normalize_subject_id(df["subject_id"])
    if df["subject_id"].isna().any() or df["subject_id"].duplicated().any():
        raise ValueError("Clinical CSV must contain one non-empty row per unique subject_id.")
    expected = set(subject_ids.astype(str))
    observed = set(df["subject_id"].astype(str))
    missing_ids = sorted(expected - observed)
    unexpected_ids = sorted(observed - expected)
    if missing_ids or unexpected_ids:
        raise ValueError(
            f"Clinical subject IDs must match MTBLS242 exactly; missing={missing_ids}, unexpected={unexpected_ids}"
        )
    ordered = pd.DataFrame({"subject_id": subject_ids.astype(str)}).merge(
        df[REQUIRED], on="subject_id", how="left", validate="one_to_one"
    )
    for col in NUMERIC:
        numeric = pd.to_numeric(ordered[col], errors="coerce")
        bad = ordered[col].notna() & numeric.isna()
        if bad.any():
            raise ValueError(f"Clinical numeric column {col!r} contains non-numeric values.")
        ordered[col] = numeric
    all_missing = [c for c in NUMERIC + CATEGORICAL if ordered[c].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"Clinical predictors are completely missing: {all_missing}")
    return ordered


def make_transformer() -> ColumnTransformer:
    """Create a fold-local imputer/scaler/encoder for clinical predictors."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2 compatibility
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)],
        remainder="drop", sparse_threshold=0,
    )


def fit_l2_clinical(nmr_train: np.ndarray, clinical_train: pd.DataFrame, y: np.ndarray,
                    nmr_test: np.ndarray, clinical_test: pd.DataFrame, lam: float):
    """Fit clinical+NMR L2 model with every preprocessing step fit on train only."""
    nmr_scaler = StandardScaler().fit(nmr_train)
    clinical_transformer = make_transformer().fit(clinical_train[NUMERIC + CATEGORICAL])
    z = np.hstack([nmr_scaler.transform(nmr_train), clinical_transformer.transform(clinical_train[NUMERIC + CATEGORICAL])])
    zz = np.hstack([nmr_scaler.transform(nmr_test), clinical_transformer.transform(clinical_test[NUMERIC + CATEGORICAL])])
    if not np.isfinite(z).all() or not np.isfinite(zz).all():
        raise ValueError("Clinical preprocessing produced non-finite values.")
    model = LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0 / (len(y) * lam),
        max_iter=10000, tol=1e-8, random_state=base.SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(z, y)
    score = model.predict_proba(zz)[:, 1]
    detail = {
        "iterations": int(model.n_iter_[0]),
        "nmr_features": int(nmr_train.shape[1]),
        "clinical_features_after_encoding": int(z.shape[1] - nmr_train.shape[1]),
        "nonzero_features": int((np.abs(model.coef_[0]) > 1e-8).sum()),
        "lambda": float(lam),
    }
    return score, detail


def candidate_score(name: str, lam: float, nmr_train: np.ndarray, clinical_train: pd.DataFrame,
                    y: np.ndarray, nmr_test: np.ndarray, clinical_test: pd.DataFrame,
                    directions: np.ndarray):
    if name == "L2 + Clinical + NMR":
        score, detail = fit_l2_clinical(nmr_train, clinical_train, y, nmr_test, clinical_test, lam)
        return score, detail
    if name == "L2 (NMR only)":
        score, detail, _ = base.fit_score("L2", {"lambda": lam}, nmr_train, y, nmr_test, directions)
        return score, {**detail, "lambda": float(lam), "nmr_features": int(nmr_train.shape[1]), "clinical_features_after_encoding": 0}
    score, detail, _ = base.fit_score("Monotonic", {"lambda": lam}, nmr_train, y, nmr_test, directions)
    return score, {**detail, "lambda": float(lam), "nmr_features": int(nmr_train.shape[1]), "clinical_features_after_encoding": 0}


def weighted_ap(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Tie-aware average precision for subject-bootstrap multiplicities."""
    order = np.argsort(-scores, kind="stable")
    yy, ss = y[order], scores[order]
    ww = weights[:, order]
    endpoints = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ww * yy, axis=1)[:, endpoints]
    total = np.cumsum(ww, axis=1)[:, endpoints]
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    increments = np.diff(np.c_[np.zeros(len(ww)), tp], axis=1)
    positives = tp[:, -1]
    return np.divide((increments * precision).sum(axis=1), positives,
                     out=np.zeros(len(ww)), where=positives > 0)


def run_external(path: Path, pairs: pd.DataFrame, nmr: np.ndarray, clinical: pd.DataFrame,
                 y: np.ndarray, names: list[str], directions: np.ndarray) -> pd.DataFrame:
    """Fit on all development patients after inner-CV tuning and score an independent CSV."""
    external = pd.read_csv(path, dtype={"subject_id": "string"})
    target = "strong_responder" if "strong_responder" in external.columns else "target"
    required = ["subject_id", target, *NUMERIC, *CATEGORICAL, *names]
    missing = [c for c in required if c not in external.columns]
    if missing:
        raise ValueError(f"External CSV is missing required columns: {missing}")
    if external["subject_id"].duplicated().any():
        raise ValueError("External CSV must have unique subject_id values.")
    if external["subject_id"].isna().any():
        raise ValueError("External CSV must have non-empty subject_id values.")
    y_ext = pd.to_numeric(external[target], errors="raise").astype(int).to_numpy()
    if not set(np.unique(y_ext)).issubset({0, 1}) or len(np.unique(y_ext)) < 2:
        raise ValueError("External target must contain both 0 and 1 classes.")
    clinical_ext = external[["subject_id", *NUMERIC, *CATEGORICAL]].copy()
    for col in NUMERIC:
        clinical_ext[col] = pd.to_numeric(clinical_ext[col], errors="raise")
    nmr_ext = external[names].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(nmr_ext).all() or (nmr_ext < 0).any():
        raise ValueError("External NMR columns must be finite and non-negative raw abundances.")
    nmr_ext = np.log1p(nmr_ext)
    # Final-model lambda is selected with a separate inner CV on all development patients.
    inner = base.splits(y, FOLDS, base.SEED + 9000)
    selected = {}
    for name in MODELS:
        means = []
        for lam in L2_GRID:
            vals = []
            for valid in inner:
                train = np.setdiff1d(np.arange(len(y)), valid)
                pred, _ = candidate_score(name, lam, nmr[train], clinical.iloc[train], y[train],
                                          nmr[valid], clinical.iloc[valid], directions)
                vals.append(average_precision_score(y[valid], pred))
            means.append(float(np.mean(vals)))
        selected[name] = float(L2_GRID[int(np.argmax(means))])
    rows = []
    for name in MODELS:
        pred, detail = candidate_score(name, selected[name], nmr, clinical, y, nmr_ext, clinical_ext, directions)
        rows.append({"model": name, "n_external": len(y_ext), "external_positives": int(y_ext.sum()),
                     "external_prevalence": float(y_ext.mean()), "pr_auc_ap": float(average_precision_score(y_ext, pred)),
                     "selected_lambda": selected[name], "parameters": json.dumps({"lambda": selected[name]}), **detail})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "external_validation.csv", index=False)
    (RESULTS / "external_protocol.json").write_text(json.dumps({
        "external_file": str(path), "target": target, "independent_cohort_required": True,
        "development_subjects": len(y), "selection": "lambda selected by inner 5-fold CV on all development subjects",
        "metric": "average precision (PR-AUC)", "n_external": len(y_ext),
    }, indent=2), encoding="utf-8")
    return out


def run(clinical_path: Path, external_path: Path | None = None) -> None:
    pairs, nmr, y, names, directions, marker_count = base.load_data()
    clinical = load_clinical(clinical_path, pairs.subject_id)
    RESULTS.mkdir(exist_ok=True)
    GRAPH.mkdir(exist_ok=True)
    all_folds, all_tuning, all_memberships, all_predictions = [], [], [], []
    for repeat in range(REPEATS):
        seed = base.SEED + 1000 * repeat
        outer = base.splits(y, FOLDS, seed)
        for fold, test in enumerate(outer, 1):
            train = np.setdiff1d(np.arange(len(y)), test)
            inner = base.splits(y[train], FOLDS, seed + fold)
            for j, valid in enumerate(inner, 1):
                for local, idx in enumerate(train):
                    all_memberships.append({"repeat": repeat + 1, "outer_fold": fold, "inner_fold": j,
                        "subject_id": pairs.subject_id.iloc[idx], "role": "validation" if local in valid else "train"})
            for name in MODELS:
                candidate_means = []
                for candidate, lam in enumerate(L2_GRID):
                    scores = []
                    for j, valid in enumerate(inner, 1):
                        inner_train = np.setdiff1d(np.arange(len(train)), valid)
                        pred, detail = candidate_score(name, lam, nmr[train[inner_train]], clinical.iloc[train[inner_train]],
                                                       y[train[inner_train]], nmr[train[valid]], clinical.iloc[train[valid]], directions)
                        ap = average_precision_score(y[train[valid]], pred)
                        scores.append(ap)
                        all_tuning.append({"repeat": repeat + 1, "outer_fold": fold, "inner_fold": j,
                            "model": name, "candidate": candidate, "lambda": lam,
                            "validation_ap": float(ap), **detail})
                    candidate_means.append(float(np.mean(scores)))
                winner = int(np.argmax(candidate_means)); lam = L2_GRID[winner]
                pred, detail = candidate_score(name, lam, nmr[train], clinical.iloc[train], y[train],
                                               nmr[test], clinical.iloc[test], directions)
                row = {"repeat": repeat + 1, "seed": seed, "outer_fold": fold, "model": name,
                       "n_train": len(train), "n_test": len(test), "test_positives": int(y[test].sum()),
                       "prevalence": float(y[test].mean()), "pr_auc_ap": float(average_precision_score(y[test], pred)),
                       "best_inner_ap": candidate_means[winner], "candidate": winner, "lambda": lam, **detail}
                all_folds.append(row)
                for idx, score in zip(test, pred):
                    all_predictions.append({"repeat": repeat + 1, "outer_fold": fold, "model": name,
                        "subject_index": int(idx), "subject_id": pairs.subject_id.iloc[idx], "y": int(y[idx]), "score": float(score)})
            print(f"repeat {repeat + 1:02d}/{REPEATS} fold {fold}/{FOLDS} complete", flush=True)

    folds = pd.DataFrame(all_folds); predictions = pd.DataFrame(all_predictions)
    pd.DataFrame(all_tuning).to_csv(RESULTS / "inner_tuning_scores.csv", index=False)
    pd.DataFrame(all_memberships).to_csv(RESULTS / "inner_fold_membership.csv", index=False)
    folds.to_csv(RESULTS / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "heldout_predictions.csv", index=False)

    rng = np.random.default_rng(20260906)
    weights = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y)), size=2000).astype(float)
    valid = (weights @ y > 0) & (weights @ (1 - y) > 0); weights = weights[valid]
    boot = {name: np.zeros(len(weights)) for name in MODELS}
    for (repeat, fold, name), group in predictions.groupby(["repeat", "outer_fold", "model"], sort=True):
        idx = group.subject_index.to_numpy(int)
        boot[name] += weighted_ap(group.y.to_numpy(int), group.score.to_numpy(float), weights[:, idx])
    for name in MODELS: boot[name] /= REPEATS * FOLDS
    pd.DataFrame(boot).to_csv(RESULTS / "subject_bootstrap_draws.csv", index=False)
    reference = boot["L2 (NMR only)"]; summaries = []
    for name in MODELS:
        subset = folds[folds.model.eq(name)]
        rep_means = subset.groupby("repeat").pr_auc_ap.mean()
        delta = boot[name] - reference
        lo, hi = np.quantile(boot[name], [0.025, 0.975]); dlo, dhi = np.quantile(delta, [0.025, 0.975])
        summaries.append({"model": name, "mean_outer_fold_ap": float(subset.pr_auc_ap.mean()),
            "repeat_mean_sd": float(rep_means.std(ddof=1)), "conditional_ci95_low": float(lo),
            "conditional_ci95_high": float(hi), "difference_vs_l2_nmr": float(subset.pr_auc_ap.mean() - folds[folds.model.eq("L2 (NMR only)")].pr_auc_ap.mean()),
            "paired_ci95_low_difference_vs_l2_nmr": float(dlo), "paired_ci95_high_difference_vs_l2_nmr": float(dhi)})
    summary = pd.DataFrame(summaries); summary.to_csv(RESULTS / "summary.csv", index=False)
    folds.groupby(["repeat", "model"]).pr_auc_ap.mean().unstack().to_csv(RESULTS / "repeat_means.csv")

    colors = ["#147d84", "#b1192e", "#7756a5"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in summary.iterrows():
        ax.plot([row.conditional_ci95_low, row.conditional_ci95_high], [i, i], lw=3, color="#147d84")
        ax.plot(row.mean_outer_fold_ap, i, "o", color=colors[i])
        ax.text(row.conditional_ci95_high + 0.01, i, f"{row.mean_outer_fold_ap:.3f}", va="center")
    ax.axvline(y.mean(), ls="--", color="gray", label=f"Prevalence = {y.mean():.3f}")
    ax.set(yticks=range(len(summary)), yticklabels=summary.model, xlim=(0, 1), xlabel="Mean held-out PR-AUC (Average Precision)", title="MTBLS242 | L2 + clinical + NMR")
    ax.invert_yaxis(); ax.legend(loc="lower right"); fig.text(.5, .015, "95% paired subject-bootstrap intervals; conditional on saved CV fits", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(GRAPH / "confidence_intervals.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODELS:
        vals = folds[folds.model.eq(name)].groupby("repeat").pr_auc_ap.mean()
        ax.plot(vals.index, vals.values, marker=".", label=name)
    ax.set(xlabel="Repeat", ylabel="Mean outer-fold PR-AUC", ylim=(0, 1), title="Paired split stability across 20 repeats"); ax.legend(); fig.tight_layout(); fig.savefig(GRAPH / "repeat_stability.png", dpi=220); plt.close(fig)

    protocol = {"repeats": REPEATS, "outer_folds": FOLDS, "inner_folds": FOLDS,
        "outer_seeds": [base.SEED + 1000 * r for r in range(REPEATS)], "n_subjects": len(y), "n_responders": int(y.sum()),
        "nmr_features": names, "clinical_features": REQUIRED[1:], "metric": "average precision (step integral, tie-aware)",
        "preprocessing": "NMR and clinical imputation/encoding/scaling fit inside each inner/outer training partition",
        "target": "at least 9 of 11 specified markers improve at 12 months", "models": MODELS,
        "lambda_grid": L2_GRID, "external_validation": "only run when --external points to an independent cohort",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": base.sklearn.__version__},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA / "s_MTBLS242.txt", DATA / "m_MTBLS242_v2_maf.tsv"]}}
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    checks = {"one_prediction_per_subject_repeat_model": bool(predictions.groupby(["repeat", "model", "subject_id"]).size().eq(1).all()),
              "expected_predictions": int(len(predictions)) == REPEATS * FOLDS * len(y) * len(MODELS),
              "all_scores_finite": bool(np.isfinite(predictions.score).all()), "external_validation": "PERFORMED" if external_path else "NOT PERFORMED"}
    (RESULTS / "run_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if external_path:
        print(run_external(external_path, pairs, nmr, clinical, y, names, directions).to_string(index=False))
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True, help="Real patient-level MTBLS242 clinical CSV")
    parser.add_argument("--external", type=Path, help="Independent cohort CSV with target, 21 NMR columns, and clinical columns")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.clinical, args.external)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
