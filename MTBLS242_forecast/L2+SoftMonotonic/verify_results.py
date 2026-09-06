"""Independent audit of the saved L2 + soft monotonic repeated-CV results."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT.parent / "Monotonic vs L2 vs Elastic Net vs SVM"
sys.path.insert(0, str(COMPARISON))
import run_comparison as base  # noqa: E402
from run_soft_monotonic import weighted_ap, MODELS

def main():
    results = ROOT / "Results"
    protocol = json.loads((results / "protocol.json").read_text())
    pairs, x, y, names, directions, count = base.load_data()
    folds = pd.read_csv(results / "outer_fold_metrics.csv")
    pred = pd.read_csv(results / "heldout_predictions.csv", dtype={"subject_id": str})
    tuning = pd.read_csv(results / "inner_tuning_scores.csv")
    membership = pd.read_csv(results / "inner_fold_membership.csv", dtype={"subject_id": str})
    assert len(pred) == protocol["repeats"] * len(y) * len(MODELS)
    assert pred.groupby(["repeat", "model", "subject_id"]).size().eq(1).all()
    assert np.isfinite(pred.score).all()
    for repeat in range(1, protocol["repeats"] + 1):
        seed = protocol["outer_seeds"][repeat - 1]
        outer = base.splits(y, 5, seed)
        for fold, test in enumerate(outer, 1):
            train = np.setdiff1d(np.arange(len(y)), test)
            inner = base.splits(y[train], 5, seed + fold)
            test_ids = set(pairs.subject_id.iloc[test])
            for name in MODELS:
                p = pred[(pred.repeat == repeat) & (pred.outer_fold == fold) & pred.model.eq(name)]
                assert set(p.subject_id) == test_ids
                row = folds[(folds.repeat == repeat) & (folds.outer_fold == fold) & folds.model.eq(name)].iloc[0]
                np.testing.assert_allclose(row.pr_auc_ap, average_precision_score(p.y, p.score), atol=1e-12)
                if name == "Soft Monotonic + L2":
                    n_candidates = 30
                else:
                    n_candidates = 5
                t = tuning[(tuning.repeat == repeat) & (tuning.outer_fold == fold) & tuning.model.eq(name)]
                assert len(t) == n_candidates * 5
                means = t.groupby("candidate").validation_ap.mean()
                winner = int(np.argmax(means.to_numpy()))
                assert int(row.candidate) == winner
                np.testing.assert_allclose(row.best_inner_ap, means.iloc[winner], atol=1e-12)
            for j, valid in enumerate(inner, 1):
                m = membership[(membership.repeat == repeat) & (membership.outer_fold == fold) & (membership.inner_fold == j)]
                assert len(m) == len(train) and m.subject_id.is_unique
                valid_ids = set(pairs.subject_id.iloc[train[valid]])
                assert set(m[m.role.eq("validation")].subject_id) == valid_ids
                assert set(m[m.role.eq("train")].subject_id) == set(pairs.subject_id.iloc[train]) - valid_ids
    draws = pd.read_csv(results / "subject_bootstrap_draws.csv")
    summary = pd.read_csv(results / "summary.csv")
    for _, row in summary.iterrows():
        d = draws[row.model]
        np.testing.assert_allclose([row.conditional_ci95_low, row.conditional_ci95_high], d.quantile([.025, .975]))
        delta = draws[row.model] - draws["Soft Monotonic + L2"]
        np.testing.assert_allclose([row.paired_ci95_low_difference_vs_soft, row.paired_ci95_high_difference_vs_soft], delta.quantile([.025, .975]))
    report = {"status": "PASS", "repeats": protocol["repeats"],
        "checks": ["patient split membership", "same outer patients across models", "one prediction per patient/repeat/model",
                   "held-out AP recomputed", "inner candidate winner recomputed", "bootstrap intervals recomputed"],
        "external_validation": "NOT PERFORMED", "interval_scope": "conditional on fitted CV models"}
    (results / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
