"""Create the protocol and run audit manifest after a completed result run."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
COMPARISON = ROOT.parent / "Monotonic vs L2 vs Elastic Net vs SVM"
sys.path.insert(0, str(COMPARISON))
import run_comparison as base  # noqa: E402

def main():
    pairs, x, y, names, directions, count = base.load_data()
    folds = pd.read_csv(RESULTS / "outer_fold_metrics.csv")
    pred = pd.read_csv(RESULTS / "heldout_predictions.csv", dtype={"subject_id": str})
    tuning = pd.read_csv(RESULTS / "inner_tuning_scores.csv")
    assert len(folds) == 20 * 5 * 3
    assert len(pred) == 20 * 3 * 71
    assert pred.groupby(["repeat", "model", "subject_id"]).size().eq(1).all()
    assert np.isfinite(pred.score).all()
    protocol = {
        "repeats": 20, "outer_folds": 5, "inner_folds": 5,
        "outer_seeds": [base.SEED + 1000 * r for r in range(20)],
        "bootstrap_draws_requested": 2000,
        "bootstrap_draws_valid": len(pd.read_csv(RESULTS / "subject_bootstrap_draws.csv")),
        "bootstrap_seed": 20260906, "n_subjects": len(y), "n_responders": int(y.sum()),
        "features": names, "constraints": base.SIGNS,
        "grids": {"Soft Monotonic + L2": [{"l2": l, "soft_penalty": r} for l in (0.001, 0.01, 0.1, 1., 10.) for r in (0., .001, .01, .1, 1., 10.)],
                  "Monotonic + L2": [{"l2": l} for l in (0.001, .01, .1, 1., 10.)],
                  "L2": [{"l2": l} for l in (0.001, .01, .1, 1., 10.)]},
        "metric": "average precision (step integral, tie-aware)",
        "soft_objective": "mean logistic loss + l2/2*sum(w^2) + soft_penalty/2*sum(max(0,-direction*w)^2)",
        "interval": "95% percentile paired subject bootstrap of fixed held-out predictions; conditional on fitted models",
        "external_validation": "not performed; no eligible independent cohort supplied",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": base.scipy.__version__, "sklearn": base.sklearn.__version__},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(base.__file__), ROOT / "run_soft_monotonic.py", ROOT / "verify_results.py", base.DATA / "s_MTBLS242.txt", base.DATA / "m_MTBLS242_v2_maf.tsv"]},
    }
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    checks = {
        "one_prediction_per_subject_repeat_model": True,
        "expected_predictions": len(pred) == 20 * 3 * 71,
        "all_scores_finite": bool(np.isfinite(pred.score).all()),
        "all_fits_converged": bool(folds.iterations.notna().all()),
        "soft_monotonic_violation_count": int(folds[folds.model.eq("Soft Monotonic + L2")].violating_features.sum()),
        "inner_fit_count": int(len(tuning)), "external_validation": "NOT PERFORMED",
    }
    (RESULTS / "run_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__": main()
