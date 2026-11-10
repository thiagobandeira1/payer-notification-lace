"""LACE index (van Walraven et al., CMAJ 2010) per index episode, from information available at discharge.

L = length of stay, A = acute (emergent) admission, C = Charlson score, E = ER visits in the prior 6 months.
In this feed an ER contact is any stay with an Emergency-class event, dated by its admit date:
- A: an ER contact on the admission day or the day before (an ER visit at another facility the day before the
  transfer counts).
- E: distinct days with an ER contact 2 to 182 days before admission (excluding the index ER visit).
"""
import numpy as np
import pandas as pd

from .charlson import charlson_frame

ER_LOOKBACK_DAYS = 182
HIGH_RISK_LACE = 10  # conventional high-risk cut-off


def los_points(days: pd.Series) -> np.ndarray:
    d = days.to_numpy()
    return np.select([d < 1, d == 1, d == 2, d == 3, d <= 6, d <= 13], [0, 1, 2, 3, 4, 5], default=7)


def charlson_points(score: pd.Series) -> np.ndarray:
    return np.where(score >= 4, 5, score)


def er_points(n: pd.Series) -> np.ndarray:
    return np.minimum(n, 4)


def er_contact_days(stays: pd.DataFrame) -> pd.DataFrame:
    er = stays.loc[stays["has_er_class"], ["patient_key", "admit"]].dropna()
    return er.assign(day=er["admit"].dt.normalize())[["patient_key", "day"]].drop_duplicates()


def er_history(idx: pd.DataFrame, stays: pd.DataFrame, lookback_days: int = ER_LOOKBACK_DAYS) -> pd.DataFrame:
    """acute (ER contact on admit day or the day before), ER visit counts in the prior 30 days and 6 months."""
    m = idx[["patient_key", "admit"]].rename_axis("episode_key").reset_index().merge(
        er_contact_days(stays), on="patient_key")
    days_before = (m["admit"].dt.normalize() - m["day"]).dt.days
    out = pd.DataFrame(index=idx.index)
    out["acute"] = idx.index.isin(m.loc[days_before.between(0, 1), "episode_key"])
    for name, hi in [("er_visits_30d", 30), ("er_visits_6m", lookback_days)]:
        counts = m[days_before.between(2, hi)].groupby("episode_key")["day"].nunique()
        out[name] = counts.reindex(idx.index, fill_value=0).astype(int)
    return out


def compute_lace(idx: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """LACE components, total, Charlson flags and ER history for each index episode."""
    out = er_history(idx, stays).join(charlson_frame(idx["dx_codes"]))
    out["lace_l"] = los_points(idx["los_days"])
    out["lace_a"] = np.where(out["acute"], 3, 0)
    out["lace_c"] = charlson_points(out["charlson"])
    out["lace_e"] = er_points(out["er_visits_6m"])
    out["lace"] = out[["lace_l", "lace_a", "lace_c", "lace_e"]].sum(axis=1)
    return out
