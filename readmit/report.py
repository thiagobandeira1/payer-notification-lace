"""Aggregate summaries. Everything here returns or prints counts, rates and distributions only."""
import pandas as pd

from .config import ReadmitConfig
from .normalize import parse_dates
from .stays import is_admission


def data_quality(raw: pd.DataFrame, events: pd.DataFrame) -> dict:
    out = {"rows": len(raw), "rows_without_patient_key": len(raw) - len(events),
           "patients": events["patient_key"].nunique()}
    for c in ["EventTime", "Admit Timestamp", "Discharge Timestamp"]:
        parsed = pd.to_datetime(raw[c], errors="coerce", format="mixed", dayfirst=False)
        out[f"{c}: 1900 placeholder"] = int((parsed.dt.year <= 1900).sum())
        out[f"{c}: unparseable"] = int((raw[c].notna() & parsed.isna()).sum())
    cancels = events.loc[events["event"].str.startswith("Cancel", na=False), "event"].value_counts()
    out.update({f"events: {k}": int(v) for k, v in cancels.items()})
    return out


def summarize(stays: pd.DataFrame, idx: pd.DataFrame, cfg: ReadmitConfig) -> dict:
    adm = stays[is_admission(stays, cfg.definition)]
    discharged = adm[adm["discharge"].notna()]
    el = idx[idx["eligible"]]
    n, k = len(el), int(el["readmit30"].sum())
    return {
        "admissions with discharge": len(discharged),
        "patients with a discharged admission": int(discharged["patient_key"].nunique()),
        "admissions without a recorded discharge": int(adm["discharge"].isna().sum()),
        "negative LOS dropped": int((adm["los_days"] < 0).sum()),
        "index candidates": len(idx),
        "episodes merged from transfers": int((idx["n_stays"] > 1).sum()),
        "exclusions": idx.loc[~idx["eligible"], "exclusion"].value_counts().to_dict(),
        "eligible index stays": n,
        "eligible patients": int(el["patient_key"].nunique()),
        "same-day next admission": int((el["days_to_next"] == 0).sum()),
        "30-day readmissions": k,
        "readmission rate": k / max(n, 1),
        "ER-only visit within 30 days": int(el["er_visit_30d"].sum()),
        "with full 6-month look-back": int(el["full_lookback"].sum()),
        "primary dx complete": el["primary_dx"].notna().mean() if n else float("nan"),
        "all dx codes complete": el["dx_codes"].notna().mean() if n else float("nan"),
        "by disposition": el.groupby("disposition")["readmit30"].agg(["count", "sum", "mean"]).round(3),
        "days to readmission": el.loc[el["readmit30"], "days_to_next"].describe().round(1).to_dict(),
    }


def print_summary(s: dict, title: str = "") -> None:
    if title:
        print(f"\n==== {title} ====")
    for key, val in s.items():
        if isinstance(val, pd.DataFrame):
            print(f"{key}:\n{val.to_string()}")
        elif isinstance(val, float):
            print(f"{key}: {val:.1%}")
        else:
            print(f"{key}: {val}")
