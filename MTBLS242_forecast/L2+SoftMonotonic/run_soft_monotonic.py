"""Repeated nested CV for L2-regularized logistic regression with soft monotonic penalties.

The soft constraint adds rho/2 * sum(max(0, -sign_j * w_j)**2) to the
mean logistic loss. L2 and rho are selected inside five inner folds. The same
patient split is used for the soft model, the hard Monotonic+L2 comparator and
the unconstrained L2 comparator.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Compatibility alias for the bundled runtime's historical version attribute spelling.
if not callable(getattr(np, "__version", None)):
    setattr(np, "__version", lambda: getattr(np, "__version__", "unknown"))
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

COMPARISON = Path(__file__).resolve().parent.parent / "Monotonic vs L2 vs Elastic Net vs SVM"
sys.path.insert(0, str(COMPARISON))
import run_comparison as base  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
GRAPH = ROOT / "Graph"
MODELS = ("Soft Monotonic + L2", "Monotonic + L2", "L2")
L2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
RHO_GRID = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)


def fit_soft(x, y, test, directions, l2, rho):
    """Fit an unconstrained L-BFGS-B model with a differentiable hinge-square penalty."""
    scaler = base.StandardScaler().fit(x)
    z, zz = scaler.transform(x), scaler.transform(test)
    n, p = z.shape
    d = np.asarray(directions, dtype=float)

    def objective(wb):
        w, b = wb[:-1], wb[-1]
        score = z @ w + b
        err = expit(score) - y
        violation = np.maximum(0.0, -d * w)
        # d == 0 means the feature is unconstrained.
        violation[d == 0] = 0.0
        loss = (np.mean(np.logaddexp(0.0, score) - y * score)
                + 0.5 * l2 * np.sum(w * w)
                + 0.5 * rho * np.sum(violation * violation))
        grad = z.T @ err / n + l2 * w
        grad += -rho * d * violation
        return loss, np.r_[grad, err.mean()]

    prevalence = np.clip(float(y.mean()), 1e-6, 1 - 1e-6)
    init = np.r_[np.zeros(p), np.log(prevalence / (1.0 - prevalence))]
    result = minimize(objective, init, jac=True, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    coef, intercept = result.x[:-1], result.x[-1]
    score = expit(zz @ coef + intercept)
    violation = np.maximum(0.0, -d * coef)
    violation[d == 0] = 0.0
    detail = {
        "iterations": int(result.nit),
        "nonzero_features": int((np.abs(coef) > 1e-8).sum()),
        "violating_features": int((violation > 1e-8).sum()),
        "max_violation": float(violation.max(initial=0.0)),
        "l2": float(l2),
        "soft_penalty": float(rho),
    }
    return score, detail, coef, scaler, intercept


def weighted_ap(y, scores, weights):
    """Tie-aware step AP for a matrix of subject-bootstrap multiplicities."""
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


def candidate_score(name, params, x, y, test, directions):
    if name == "Soft Monotonic + L2":
        return fit_soft(x, y, test, directions, params["l2"], params["soft_penalty"])
    if name == "Monotonic + L2":
        score, detail, coef = base.fit_score(
            "Monotonic", {"lambda": params["l2"]}, x, y, test, directions)
        detail = {**detail, "l2": float(params["l2"]), "soft_penalty": None,
                  "violating_features": int(detail.get("violations") or 0),
                  "max_violation": 0.0}
        return score, detail, coef, None, None
    score, detail, coef = base.fit_score("L2", {"lambda": params["l2"]}, x, y, test, directions)
    detail = {**detail, "l2": float(params["l2"]), "soft_penalty": None,
              "violating_features": None, "max_violation": None}
    return score, detail, coef, None, None


def grids():
    return {
        "Soft Monotonic + L2": [{"l2": l, "soft_penalty": r} for l in L2_GRID for r in RHO_GRID],
        "Monotonic + L2": [{"l2": l} for l in L2_GRID],
        "L2": [{"l2": l} for l in L2_GRID],
    }


def run():
    repeats, bootstrap_draws = 20, 2000
    RESULTS.mkdir(exist_ok=True)
    GRAPH.mkdir(exist_ok=True)
    pairs, x, y, names, directions, marker_count = base.load_data()
    grid = grids()
    all_folds, all_tuning, all_memberships, all_predictions = [], [], [], []
    for repeat in range(repeats):
        seed = base.SEED + 1000 * repeat
        outer = base.splits(y, 5, seed)
        for fold, test in enumerate(outer, 1):
            train = np.setdiff1d(np.arange(len(y)), test)
            inner = base.splits(y[train], 5, seed + fold)
            for j, valid in enumerate(inner, 1):
                for local, idx in enumerate(train):
                    all_memberships.append({"repeat": repeat + 1, "outer_fold": fold,
                        "inner_fold": j, "subject_id": pairs.subject_id.iloc[idx],
                        "role": "validation" if local in valid else "train"})
            for name in MODELS:
                candidate_means = []
                for candidate, params in enumerate(grid[name]):
                    scores = []
                    for j, valid in enumerate(inner, 1):
                        inner_train = np.setdiff1d(np.arange(len(train)), valid)
                        pred, detail, _, _, _ = candidate_score(
                            name, params, x[train[inner_train]], y[train[inner_train]],
                            x[train[valid]], directions)
                        ap = average_precision_score(y[train[valid]], pred)
                        scores.append(ap)
                        all_tuning.append({"repeat": repeat + 1, "outer_fold": fold,
                            "inner_fold": j, "model": name, "candidate": candidate,
                            "parameters": json.dumps(params), "validation_ap": float(ap), **detail})
                    candidate_means.append(float(np.mean(scores)))
                winner = int(np.argmax(candidate_means))
                params = grid[name][winner]
                pred, detail, coef, _, _ = candidate_score(name, params, x[train], y[train], x[test], directions)
                ap = average_precision_score(y[test], pred)
                row = {"repeat": repeat + 1, "seed": seed, "outer_fold": fold,
                       "model": name, "n_train": len(train), "n_test": len(test),
                       "test_positives": int(y[test].sum()), "prevalence": float(y[test].mean()),
                       "pr_auc_ap": float(ap), "best_inner_ap": candidate_means[winner],
                       "candidate": winner, "parameters": json.dumps(params), **detail}
                all_folds.append(row)
                for idx, score in zip(test, pred):
                    all_predictions.append({"repeat": repeat + 1, "outer_fold": fold,
                        "model": name, "subject_index": int(idx),
                        "subject_id": pairs.subject_id.iloc[idx], "y": int(y[idx]),
                        "score": float(score)})
            print(f"repeat {repeat + 1:02d}/20 fold {fold}/5 complete", flush=True)

    folds = pd.DataFrame(all_folds)
    predictions = pd.DataFrame(all_predictions)
    tuning = pd.DataFrame(all_tuning)
    memberships = pd.DataFrame(all_memberships)
    folds.to_csv(RESULTS / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "heldout_predictions.csv", index=False)
    tuning.to_csv(RESULTS / "inner_tuning_scores.csv", index=False)
    memberships.to_csv(RESULTS / "inner_fold_membership.csv", index=False)

    rng = np.random.default_rng(20260906)
    weights = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y)), size=bootstrap_draws).astype(float)
    valid_draw = (weights @ y > 0) & (weights @ (1 - y) > 0)
    weights = weights[valid_draw]
    boot = {name: np.zeros(len(weights)) for name in MODELS}
    for (repeat, fold, name), group in predictions.groupby(["repeat", "outer_fold", "model"], sort=True):
        idx = group.subject_index.to_numpy(int)
        boot[name] += weighted_ap(group.y.to_numpy(int), group.score.to_numpy(float), weights[:, idx])
    for name in MODELS:
        boot[name] /= repeats * 5
    boot_df = pd.DataFrame(boot)
    boot_df.to_csv(RESULTS / "subject_bootstrap_draws.csv", index=False)

    summaries = []
    soft_boot = boot["Soft Monotonic + L2"]
    for name in MODELS:
        subset = folds[folds.model.eq(name)]
        rep_means = subset.groupby("repeat").pr_auc_ap.mean()
        delta = boot[name] - soft_boot
        lo, hi = np.quantile(boot[name], [0.025, 0.975])
        dlo, dhi = np.quantile(delta, [0.025, 0.975])
        summaries.append({"model": name, "mean_outer_fold_ap": subset.pr_auc_ap.mean(),
            "repeat_mean_sd": rep_means.std(ddof=1), "conditional_ci95_low": lo,
            "conditional_ci95_high": hi, "difference_vs_soft": subset.pr_auc_ap.mean() - folds[folds.model.eq("Soft Monotonic + L2")].pr_auc_ap.mean(),
            "paired_ci95_low_difference_vs_soft": dlo, "paired_ci95_high_difference_vs_soft": dhi,
            "soft_penalty_selected_fraction": (subset.soft_penalty.fillna(0) > 0).mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    folds.groupby(["repeat", "model"]).pr_auc_ap.mean().unstack().to_csv(RESULTS / "repeat_means.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in summary.iterrows():
        ax.plot([row.conditional_ci95_low, row.conditional_ci95_high], [i, i], lw=3, color="#147d84")
        ax.plot(row.mean_outer_fold_ap, i, "o", color="#b1192e")
        ax.text(row.conditional_ci95_high + 0.01, i, f"{row.mean_outer_fold_ap:.3f}", va="center")
    ax.axvline(y.mean(), ls="--", color="gray", label=f"Prevalence = {y.mean():.3f}")
    ax.set(yticks=range(len(MODELS)), yticklabels=MODELS, xlim=(0, 1),
           xlabel="Mean held-out PR-AUC (Average Precision)",
           title="MTBLS242 | Repeated nested 5 × 5 CV")
    ax.invert_yaxis(); ax.legend(loc="lower right")
    fig.text(.5, .015, "95% paired subject-bootstrap intervals; conditional on saved CV fits", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(GRAPH / "soft_monotonic_confidence_intervals.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODELS:
        vals = folds[folds.model.eq(name)].groupby("repeat").pr_auc_ap.mean()
        ax.plot(vals.index, vals.values, marker=".", label=name)
    ax.set(xlabel="Repeat", ylabel="Mean outer-fold PR-AUC", ylim=(0, 1),
           title="Split stability across 20 repeats")
    ax.legend(); fig.tight_layout(); fig.savefig(GRAPH / "soft_monotonic_repeat_stability.png", dpi=220); plt.close(fig)

    protocol = {"repeats": repeats, "outer_folds": 5, "inner_folds": 5,
        "outer_seeds": [base.SEED + 1000 * r for r in range(repeats)],
        "bootstrap_draws_requested": bootstrap_draws, "bootstrap_draws_valid": int(len(weights)),
        "bootstrap_seed": 20260906, "n_subjects": len(y), "n_responders": int(y.sum()),
        "features": names, "constraints": base.SIGNS, "grids": grid,
        "metric": "average precision (step integral, tie-aware)",
        "soft_objective": "mean logistic loss + l2/2*sum(w^2) + soft_penalty/2*sum(max(0,-direction*w)^2)",
        "interval": "95% percentile paired subject bootstrap of fixed held-out predictions; conditional on fitted models",
        "external_validation": "not performed; no eligible independent cohort supplied",
        "software": {"python": platform.python_version(), "numpy": np.__version(), "pandas": pd.__version(),
                     "scipy": base.scipy.__version__, "sklearn": base.sklearn.__version__},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(base.__file__), ROOT / "run_soft_monotonic.py", base.DATA / "s_MTBLS242.txt",
            base.DATA / "m_MTBLS242_v2_maf.tsv"]}}
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    checks = {"one_prediction_per_subject_repeat_model": bool(predictions.groupby(["repeat", "model", "subject_id"]).size().eq(1).all()),
        "expected_predictions": int(len(predictions)) == repeats * 5 * len(y) * len(MODELS),
        "all_scores_finite": bool(np.isfinite(predictions.score).all()),
        "all_fits_converged": bool(folds.iterations.notna().all()),
        "soft_monotonic_violation_count": int(folds[folds.model.eq("Soft Monotonic + L2")].violating_features.sum()),
        "inner_fit_count": int(len(tuning)), "external_validation": "NOT PERFORMED"}
    (RESULTS / "run_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
