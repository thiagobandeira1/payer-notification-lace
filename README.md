# payer-notification-lace

Analysis code for the readmission-prediction part of:

> Bandeira TBN. **From Payer Notifications to Primary-Care Follow-up: Retrospective Evaluation of a
> Notification-Derived LACE Score and a Transitions-of-Care Program.** Preprint, Zenodo, 2026.
> doi:[10.5281/zenodo.24090922](https://doi.org/10.5281/zenodo.24090922)

The code rebuilds hospital stays from a health plan's encounter notification (ENS) file, labels 30-day
readmissions, scores each stay with a notification-derived adaptation of the LACE index, and compares LACE with
penalized logistic regression and XGBoost using patient-grouped cross-validation.

**No patient data is included.** The tests use synthetic patients only.

## What is included

| Paper item | Where in the code |
|---|---|
| Stay reconstruction (merging same-day and overlapping stays, cancellations, events in date order) | `readmit/stays.py`, `readmit/normalize.py` |
| Readmission outcome (1–30 days, strict and broad definitions) and exclusions | `readmit/labels.py`, `readmit/config.py` |
| LACE adaptation (Supplement S1, Table S1) and Charlson comorbidity (Quan ICD-10 coding, original Charlson weights) | `readmit/lace.py`, `readmit/charlson.py` |
| Compact (17) and full (68) notification-derived feature sets (Table S8) | `readmit/features.py` |
| 5-fold patient-grouped cross-validation with nested 3-fold tuning, and the tuning grids | `readmit/model.py` |
| Metrics, calibration intercept and slope, 2,000 patient-clustered bootstrap resamples, four repeat seeds | `readmit/evaluate.py`, `readmit/experiment.py` |
| Unit tests for stays, labels, LACE and leakage rules, on synthetic patients | `tests/` |

## What is not included

- **Patient data** of any kind, including derived row-level files. These cannot be shared.
- **The transitions-of-care program** (task boards, deadlines, EHR matching) described in Section 2.5 of the paper.
- **The ED claims analysis** (Section 2.6) and the interrupted time series. Monthly aggregate ED counts are in the
  paper's Supplement S2.
- **The operational risk-tier figures** (Section 3.2), which came from the practice's operational database.

## Setup and tests

Python 3.12.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest -q
```

## Running on your own notification file

The input is a payer ENS CSV with the columns listed in `readmit/ingest.py` (`ENS_COLUMNS`). Put it at
`data/raw/ens_notifications.csv` or pass `--input PATH`.

```
python -m readmit build      # rebuild stays and label readmissions
python -m readmit lace       # LACE AUROC with patient-bootstrap confidence intervals
python -m readmit model      # LACE vs logistic regression vs XGBoost, calibration
```

Names, dates of birth, addresses, MRNs and free text are dropped at normalization, and patients are tracked by a
hashed key. Console output is aggregate only. Files written to `data/processed/` and `outputs/` are still row-level
health information (they keep dates), so keep them on systems approved for patient data. `.gitignore` excludes
`data/`, `outputs/` and all CSV, Excel and PDF files.

Results depend on how complete and consistent your payer's feed is. Check your own feed and validate the score
locally before using it to direct care; the paper did not establish prospective performance or clinical benefit.

## Citation

See `CITATION.cff`, or cite the paper DOI above.

## License

Code: MIT (see `LICENSE`). The paper itself is under CC BY-NC-ND 4.0 on Zenodo.
