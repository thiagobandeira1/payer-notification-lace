"""Discharge-time feature matrix for index episodes.

Leakage rules: every feature uses only events on or before the index discharge (diagnosis codes up to
cfg.dx_grace_days after it, because scoring runs the morning after discharge). History counts look strictly before
the index admission. Labels (readmit30, days_to_next, er_visit_30d) and anything dated after discharge are never
features.
"""
import re

import pandas as pd

from .lace import compute_lace

DISPOSITIONS = ["Home", "Home Health", "SNF", "ALF", "AMA", "Other", "Missing"]
# ICD-10 chapters by code prefix (first letter, with a numeric split where a letter spans two chapters)
CHAPTERS = [
    ("infectious", "A00", "B99"), ("neoplasm", "C00", "D49"), ("blood", "D50", "D89"),
    ("endocrine", "E00", "E89"), ("mental", "F01", "F99"), ("nervous", "G00", "G99"),
    ("eye_ear", "H00", "H95"), ("circulatory", "I00", "I99"), ("respiratory", "J00", "J99"),
    ("digestive", "K00", "K95"), ("skin", "L00", "L99"), ("musculoskeletal", "M00", "M99"),
    ("genitourinary", "N00", "N99"), ("pregnancy", "O00", "O9A"), ("symptoms", "R00", "R99"),
    ("injury", "S00", "T88"),
]
TOP_FACILITIES = 8
LEAKY = {"readmit30", "days_to_next", "er_visit_30d", "next_admit", "eligible", "exclusion"}
# Pre-specified small set for ~100-200 events (about 10 events per feature): LACE components plus history,
# age and discharge setting. The full matrix (~68 columns) is reported as a secondary analysis.
COMPACT_FEATURES = [
    "lace", "los_days", "acute", "charlson", "er_visits_6m", "age", "prior_adm_6m", "days_since_last_discharge",
    "n_dx_codes", "dx_missing", "disp_Home", "disp_Home_Health", "disp_SNF", "disp_AMA", "inpatient_class",
    "weekend_discharge", "full_lookback",
]


def dx_chapter(code) -> str:
    if not isinstance(code, str) or len(code.strip()) < 3:
        return "missing"
    c = code.strip().upper().replace(".", "")[:3]
    for name, lo, hi in CHAPTERS:
        if lo <= c <= hi:
            return name
    return "other"


def prior_admissions(idx: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Admission episodes that started before the index admission date: counts and days since last discharge."""
    m = idx[["patient_key", "admit"]].rename_axis("episode_key").reset_index().merge(
        episodes[["patient_key", "admit", "discharge"]], on="patient_key", suffixes=("", "_prior"))
    days_before = (m["admit"].dt.normalize() - m["admit_prior"].dt.normalize()).dt.days
    m = m[days_before > 0]
    out = pd.DataFrame(index=idx.index)
    for name, hi in [("prior_adm_6m", 182), ("prior_adm_12m", 365)]:
        out[name] = m[days_before[m.index] <= hi].groupby("episode_key").size().reindex(idx.index, fill_value=0)
    gap = (m["admit"].dt.normalize() - m["discharge"].dt.normalize()).dt.days
    out["days_since_last_discharge"] = gap[gap >= 0].groupby(m.loc[gap >= 0, "episode_key"]).min().reindex(idx.index)
    return out


def one_hot(s: pd.Series, levels: list[str], prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_{lv}": (s == lv).astype(int) for lv in levels}, index=s.index)


def build_features(idx: pd.DataFrame, episodes: pd.DataFrame, stays: pd.DataFrame,
                   facilities: list[str] | None = None) -> pd.DataFrame:
    """Feature matrix (one row per index episode). `facilities` fixes the facility columns (top-N by default)."""
    lace = compute_lace(idx, stays)
    n_codes = idx["dx_codes"].map(lambda s: len(s.split(";")) if isinstance(s, str) else 0)
    if facilities is None:
        facilities = idx["facility"].value_counts().head(TOP_FACILITIES).index.tolist()
    fac = idx["facility"].where(idx["facility"].isin(facilities), "other")
    X = pd.concat([
        pd.DataFrame({
            "age": idx["age"], "los_days": idx["los_days"], "acute": lace["acute"].astype(int),
            "er_visits_30d": lace["er_visits_30d"], "er_visits_6m": lace["er_visits_6m"],
            "charlson": lace["charlson"], "lace": lace["lace"], "n_dx_codes": n_codes,
            "dx_missing": (n_codes == 0).astype(int), "n_stays": idx["n_stays"],
            "n_facilities": idx["n_facilities"], "inpatient_class": idx["has_inpatient_class"].astype(int),
            "weekend_discharge": (idx["discharge"].dt.day_of_week >= 5).astype(int),
            "full_lookback": idx["full_lookback"].astype(int),
        }, index=idx.index),
        lace.filter(like="cci_"),
        prior_admissions(idx, episodes),
        one_hot(idx["disposition"], DISPOSITIONS, "disp"),
        one_hot(idx["primary_dx"].map(dx_chapter), [c[0] for c in CHAPTERS] + ["other", "missing"], "dxch"),
        one_hot(fac, facilities + ["other"], "facility"),
    ], axis=1)
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_") for c in X.columns]  # safe for XGBoost
    assert not LEAKY & set(X.columns)
    return X
