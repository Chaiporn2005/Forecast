"""Validate the clinical input required by the MTBLS242 L2_Predict pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
COMPARISON = DATA / "Monotonic vs L2 vs Elastic Net vs SVM"
sys.path.insert(0, str(COMPARISON))
import run_comparison as base  # noqa: E402

NUMERIC = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol",
    "hdl_cholesterol", "triglycerides", "glucose", "creatinine", "bun",
    "uric_acid", "albumin",
]
CATEGORICAL = ["sex", "surgery_type", "diabetes", "medication_any"]
REQUIRED = ["subject_id", *NUMERIC, *CATEGORICAL]


def normalize_subject_id(values: pd.Series) -> pd.Series:
    values = values.astype("string").str.strip()
    return values.where(~values.str.fullmatch(r"\d+"), values.str.zfill(4))


def write_template(path: Path) -> None:
    pairs, *_ = base.load_data()
    frame = pd.DataFrame({"subject_id": pairs.subject_id.astype(str)})
    for col in REQUIRED[1:]:
        frame[col] = ""
    frame.to_csv(path, index=False)


def validate(path: Path) -> dict:
    pairs, *_ = base.load_data()
    report = {
        "source_subjects": int(len(pairs)),
        "source_responders": 31,
        "required_columns": REQUIRED,
        "path": str(path),
        "status": "READY" if path.exists() else "MISSING_FILE",
        "missing_columns": [],
        "missing_subject_ids": [],
        "duplicate_subject_ids": [],
        "unparseable_numeric_values": {},
        "all_missing_columns": [],
    }
    if not path.exists():
        return report
    df = pd.read_csv(path, dtype={"subject_id": "string"})
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["status"] = "INVALID_SCHEMA"
        return report
    ids = normalize_subject_id(df["subject_id"])
    report["duplicate_subject_ids"] = sorted(ids[ids.duplicated()].dropna().unique().tolist())
    expected = set(pairs.subject_id.astype(str))
    observed = set(ids.dropna().tolist())
    report["missing_subject_ids"] = sorted(expected - observed)
    report["unexpected_subject_ids"] = sorted(observed - expected)
    for col in NUMERIC:
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & numeric.isna()
        if bad.any():
            report["unparseable_numeric_values"][col] = int(bad.sum())
    report["all_missing_columns"] = [c for c in REQUIRED[1:] if df[c].notna().sum() == 0]
    if (not report["missing_subject_ids"] and not report["unexpected_subject_ids"]
            and not report["duplicate_subject_ids"]
            and not report["unparseable_numeric_values"]
            and not report["all_missing_columns"]):
        report["status"] = "READY"
    else:
        report["status"] = "INVALID_ROWS"
    report["n_rows"] = int(len(df))
    report["missing_cells"] = {c: int(df[c].isna().sum()) for c in REQUIRED[1:]}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=ROOT / "clinical_input.csv")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    if args.write_template:
        write_template(args.clinical)
        print(f"Template written to {args.clinical}")
    report = validate(args.clinical)
    results = ROOT / "Results"
    results.mkdir(exist_ok=True)
    (results / "availability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
