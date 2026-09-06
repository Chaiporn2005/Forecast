"""Forecast strong 12-month metabolic response using monotonic constraints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_mtbls242 import load_longitudinal_pairs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_pr_auc"
SEED = 42
N_SPLITS = 5
MIN_IMPROVED_MARKERS = 9
L2_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)

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


def make_outcome(x_preop, y_12mo, metabolites):
    """Strong response = >=9/11 markers improve in the literature direction."""
    marker_idx = [metabolites.index(name) for name in MONOTONE_DIRECTIONS]
    signs = np.array([MONOTONE_DIRECTIONS[name] for name in MONOTONE_DIRECTIONS])
    change = y_12mo[:, marker_idx] - x_preop[:, marker_idx]
    improved_count = np.sum(change * signs < 0, axis=1)
    return (improved_count >= MIN_IMPROVED_MARKERS).astype(int), improved_count


def stratified_folds(y, n_splits, seed):
    rng = np.random.default_rng(seed)
    fold_parts = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for fold_no, part in enumerate(np.array_split(idx, n_splits)):
            fold_parts[fold_no].append(part)
    return [np.sort(np.concatenate(parts)).astype(int) for parts in fold_parts]


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def fit_monotone_logistic(x, y, constraints, l2, max_iter=6000):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    weights = np.zeros(z.shape[1], dtype=float)
    prevalence = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    intercept = float(np.log(prevalence / (1 - prevalence)))

    for iteration in range(max_iter):
        prob = sigmoid(z @ weights + intercept)
        error = prob - y
        grad_w = z.T @ error / len(y) + l2 * weights
        grad_b = float(error.mean())
        step = 0.12 / np.sqrt(1.0 + iteration / 250.0)
        next_weights = weights - step * grad_w
        next_weights[constraints == 1] = np.maximum(next_weights[constraints == 1], 0.0)
        next_weights[constraints == -1] = np.minimum(next_weights[constraints == -1], 0.0)
        next_intercept = intercept - step * grad_b
        delta = max(np.max(np.abs(next_weights - weights)), abs(next_intercept - intercept))
        weights, intercept = next_weights, next_intercept
        if delta < 1e-8:
            break

    return {
        "x_mean": mean,
        "x_std": std,
        "weights": weights,
        "intercept": intercept,
        "l2": float(l2),
    }


def predict_probability(model, x):
    z = (x - model["x_mean"]) / model["x_std"]
    return sigmoid(z @ model["weights"] + model["intercept"])


def average_precision(y, score):
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    ranked = y[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * ranked) / positives)


def select_l2(x, y, constraints, seed):
    inner = stratified_folds(y, 4, seed)
    all_idx = np.arange(len(y))
    scores = {}
    for l2 in L2_VALUES:
        aps = []
        for valid_idx in inner:
            train_idx = np.setdiff1d(all_idx, valid_idx, assume_unique=True)
            model = fit_monotone_logistic(x[train_idx], y[train_idx], constraints, l2)
            aps.append(average_precision(y[valid_idx], predict_probability(model, x[valid_idx])))
        scores[l2] = float(np.nanmean(aps))
    return max(scores, key=scores.get)


def precision_recall_points(y, score):
    ranked = y[np.argsort(-score, kind="mergesort")]
    tp = np.cumsum(ranked)
    fp = np.cumsum(1 - ranked)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    return np.r_[0.0, recall], np.r_[1.0, precision]


def save_pr_svg(y, score, path, ap):
    recall, precision = precision_recall_points(y, score)
    prevalence = float(y.mean())
    width, height, margin = 760, 650, 85

    def sx(value):
        return margin + value * (width - 2 * margin)

    def sy(value):
        return height - margin - value * (height - 2 * margin)

    points = " ".join(f"{sx(r):.1f},{sy(p):.1f}" for r, p in zip(recall, precision))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="21" font-weight="bold">MTBLS242 strong metabolic response forecast</text>
<text x="{width/2}" y="62" text-anchor="middle" font-family="Arial" font-size="14">Held-out stratified 5-fold predictions · PR-AUC (AP) = {ap:.3f}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>
<line x1="{margin}" y1="{sy(prevalence):.1f}" x2="{width-margin}" y2="{sy(prevalence):.1f}" stroke="#777" stroke-dasharray="7 5"/>
<polyline points="{points}" fill="none" stroke="#b2182b" stroke-width="4"/>
<text x="{width-margin-5}" y="{sy(prevalence)-8:.1f}" text-anchor="end" font-family="Arial" font-size="13">No-skill baseline = {prevalence:.3f}</text>
<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="16">Recall</text>
<text x="24" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 24 {height/2})">Precision</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paired, x, future, metabolites = load_longitudinal_pairs()
    y, improved_count = make_outcome(x, future, metabolites)
    constraints = np.array([MONOTONE_DIRECTIONS.get(name, 0) for name in metabolites])
    outer = stratified_folds(y, N_SPLITS, SEED)
    all_idx = np.arange(len(y))
    oof = np.full(len(y), np.nan)
    rows = []

    for fold_no, test_idx in enumerate(outer, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        l2 = select_l2(x[train_idx], y[train_idx], constraints, SEED + fold_no)
        model = fit_monotone_logistic(x[train_idx], y[train_idx], constraints, l2)
        prob = predict_probability(model, x[test_idx])
        oof[test_idx] = prob
        rows.append({
            "fold": fold_no,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_positive_test": int(y[test_idx].sum()),
            "l2": l2,
            "pr_auc_average_precision": average_precision(y[test_idx], prob),
            "baseline_prevalence": float(y[test_idx].mean()),
        })

    fold_metrics = pd.DataFrame(rows)
    oof_ap = average_precision(y, oof)
    final_l2 = select_l2(x, y, constraints, SEED + 100)
    final_model = fit_monotone_logistic(x, y, constraints, final_l2)
    weights = np.asarray(final_model["weights"])
    violations = int(
        np.sum((constraints == 1) & (weights < -1e-12))
        + np.sum((constraints == -1) & (weights > 1e-12))
    )
    summary = {
        "task": "forecast strong 12-month metabolic response from preop NMR",
        "outcome_definition": "at least 9 of 11 literature-directed markers improve",
        "n_subjects": int(len(y)),
        "n_responders": int(y.sum()),
        "prevalence_baseline": float(y.mean()),
        "metric": "PR-AUC calculated as average precision",
        "evaluation": "nested stratified 5-fold cross-validation; held-out subjects only",
        "oof_pr_auc": oof_ap,
        "mean_fold_pr_auc": float(fold_metrics["pr_auc_average_precision"].mean()),
        "std_fold_pr_auc": float(fold_metrics["pr_auc_average_precision"].std(ddof=0)),
        "monotone_positive": int(np.sum(constraints == 1)),
        "monotone_negative": int(np.sum(constraints == -1)),
        "unconstrained": int(np.sum(constraints == 0)),
        "monotonicity_violations": violations,
    }

    predictions = paired[["subject_id", "Sample Name_preop", "Sample Name_12mo"]].copy()
    predictions["improved_marker_count"] = improved_count
    predictions["strong_responder"] = y
    predictions["heldout_probability"] = oof
    coefficients = pd.DataFrame({
        "metabolite": metabolites,
        "constraint": constraints,
        "standardized_coefficient": weights,
    }).sort_values("standardized_coefficient", ascending=False)

    np.savez_compressed(
        OUTPUT_DIR / "monotone_responder_model.npz",
        **final_model,
        metabolites=np.array(metabolites, dtype=object),
        constraints=constraints,
    )
    predictions.to_csv(OUTPUT_DIR / "heldout_responder_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_pr_auc.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "monotone_coefficients.csv", index=False)
    (OUTPUT_DIR / "summary_pr_auc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_pr_svg(y, oof, OUTPUT_DIR / "heldout_pr_curve.svg", oof_ap)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
