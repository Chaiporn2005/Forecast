# External validation status: NOT PERFORMED

No independent patient-level cohort satisfying the fixed MTBLS242 input/target specification was available for evaluation on 2026-09-06. No external PR-AUC or clinical efficacy result is claimed. Internal resampling does not substitute for external validation.

## Candidate cohort screening

This was a targeted feasibility search, not an exhaustive systematic review. Queries included `bariatric surgery serum NMR 12 months longitudinal MetaboLights`, `bariatric 12 months metabolomics data availability`, and searches for candidate study data-availability statements. Sources were searched on 2026-09-06.

| Candidate / primary source | Evidence | Decision for this run |
|---|---|---|
| [Han et al., 2022, Korean Obesity Surgical Treatment Study](https://dom-pubs.onlinelibrary.wiley.com/doi/full/10.1111/dom.14689) | 52 participants, serum NMR, baseline and follow-up through 12 months. Data-availability section requires application and corresponding-author approval. | Promising candidate; subject-level metabolite table not available here. Exact 21-feature coverage, units and independent patients must be checked after access. ENA sequencing data alone are insufficient. |
| [Nassif et al., 2026, sleeve gastrectomy](https://www.mdpi.com/2218-1989/16/7/497) | Reported cohort of 45 patients with preoperative, 3- and 12-month serum/fecal measurements. | Candidate only. Publisher full-text fetch returned HTTP 429 and PMC browser challenge; downloadable compatible patient-level table and availability terms not verified. |
| [By-Band-Sleeve NMR data report, 2025 preprint](https://www.medrxiv.org/content/10.1101/2025.11.11.25339993v1.full) | 1,045 individuals and 250 metabolic traits after filtering; individual data subject to investigator request and conditions. | Not available for this run; exact 12-month availability, assay mapping and target coverage unresolved. Preprint evidence. |
| [MTBLS218 / Narath et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5008721/) | Public bariatric longitudinal metabolomics measured using LC-HRMS. | Different assay; not a drop-in validation of the fixed 21 NMR inputs. No outcome-driven cross-platform mapping attempted. |
| [AlpsNMR case study, 2020](https://academic.oup.com/bioinformatics/article/36/9/2943/5701646) | Reuses MTBLS242. | Same underlying cohort; cannot count as external validation. |

MTBLS1 and NHANES from the sibling project have different outcomes and do not provide the required paired bariatric serum profile for the same target.

## Files prepared

- `development_lock.json`: fixed feature order, original-data/code hashes, development-only parameter search results and primary model choice. Development tuning AP is NOT held-out external performance.
- `development_tuning.csv`: all candidate tuning scores used to make that lock.
- `external_input_template.csv`: header only; no fabricated observations.
- `provenance_template.json`: evidence fields to complete when data arrive.
- `../external_validation.py`: scorer that refits the locked models on the 71 development subjects and evaluates an eligible independent table once.

The lock selects a primary model using a prespecified development-only five-fold split (seed 90210). All four model families remain secondary comparisons. External outcomes do not tune parameters, select variables, choose a model, set a threshold, calibrate scores or determine preprocessing. SVM emits ranking margins, not probabilities.

## Required data and review

One row per independent patient with a stable, cohort-namespaced identifier. Provide 21 baseline measurements as `preop::<exact metabolite name>` and the 11 fixed target markers at 12 months as `month12::<exact metabolite name>`. The file template defines the exact names. Values must be nonnegative, finite and in the same measurement scale as development data before `log1p`. The original feature labels include stereoisomer names and a broad lipoprotein signal; similar names in another assay are not automatically equivalent.

Document patient independence, serum collection, feature identity, units, laboratory harmonization, exact follow-up timing, outcome-blind processing and data-use permission in the provenance JSON. Name matching and numeric validation cannot prove biological comparability. A qualified dataset/assay owner must substantiate the evidence. Do not normalize the external cohort to optimize its target performance or replace absent markers with guessed values. Missing markers require a separately developed reduced-feature model and a new locked protocol.

From the comparison folder, after eligible data and provenance are supplied:

```sh
python external_validation.py --data paired_external.csv --provenance evidence.json
```

External results, if this command succeeds on real data, go to `External_validation/evaluated/`. The script blocks empty evidence, duplicate/overlapping patient IDs, the development cohort, missing features, invalid values, single-class outcomes, changed development inputs and overwriting an existing external result. The evaluator reports AP with ordinary patient-bootstrap percentile intervals conditional on the fixed trained model. No evaluation results have been generated in this run.

## Clinical interpretation still pending

The fixed response definition (at least 9/11 metabolite changes) is an experimental surrogate. It depends on baseline values that are also predictors, so mathematical coupling and regression to the mean remain possible explanations. Even successful independent AP would establish discrimination for this surrogate, not weight-loss benefit, diabetes remission or improved patient care. Clinical endpoint linkage, calibration, prespecified decision thresholds, clinical utility and prospective evaluation require a subsequent protocol.
