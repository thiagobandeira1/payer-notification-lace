"""Merge transfers into episodes of care and label 30-day readmissions."""
import numpy as np
import pandas as pd

from .config import ReadmitConfig
from .stays import final_disposition, is_admission, is_er_only

INDEX_COLUMNS = [
    "patient_key", "facility", "age", "n_stays", "n_facilities", "admit", "discharge", "los_days", "via_er",
    "has_inpatient_class", "has_admission_event", "disposition", "primary_dx", "dx_codes",
    "days_to_next", "readmit30", "er_visit_30d", "full_lookback", "eligible", "exclusion",
]


def episode_numbers(adm: pd.DataFrame) -> np.ndarray:
    """Same-day or overlapping admissions of one patient share an episode number. `adm` sorted by patient, admit.

    An admission whose own discharge is unknown ends the episode with an unknown discharge.
    """
    out = np.empty(len(adm), dtype=int)
    n, cur_patient, cur_end = -1, None, pd.NaT
    for i, (p, a, d) in enumerate(zip(adm["patient_key"], adm["admit"], adm["discharge"])):
        if p == cur_patient and pd.notna(cur_end) and pd.notna(a) and a.normalize() <= cur_end.normalize():
            cur_end = pd.NaT if pd.isna(d) else max(cur_end, d)
        else:
            n, cur_patient, cur_end = n + 1, p, d
        out[i] = n
    return out


def to_episodes(adm: pd.DataFrame, merge: bool) -> pd.DataFrame:
    """One row per episode of care, indexed by its first stay_key. Without merging, each stay is its own episode."""
    adm = adm.sort_index().sort_values(["patient_key", "admit"], kind="stable")
    ep = episode_numbers(adm) if merge else np.arange(len(adm))
    g = adm.groupby(ep, sort=False)
    eps = pd.DataFrame({
        "episode_key": adm.index.to_series().groupby(ep, sort=False).first(),
        "patient_key": g["patient_key"].first(),
        "facility": g["facility"].first(),
        "age": g["age"].min(),
        "n_stays": g.size(),
        "n_facilities": g["facility"].nunique(),
        "admit": g["admit"].min(),
        # unknown if any stay in the episode has no recorded discharge
        "discharge": g["discharge"].max().where(g["discharge"].count() == g.size()),
        "via_er": g["has_er_class"].first(),
        "has_inpatient_class": g["has_inpatient_class"].any(),
        "has_admission_event": g["has_admission_event"].any(),
        # a hospice or death discharge anywhere in a transfer chain is the episode's outcome
        "disposition": g["disposition"].agg(final_disposition, skip_missing=False),
        "primary_dx": g["primary_dx"].last(),
        "dx_codes": g["dx_codes"].agg(_union_codes),
    })
    eps["los_days"] = (eps["discharge"].dt.normalize() - eps["admit"].dt.normalize()).dt.days
    return eps.set_index("episode_key")


def _union_codes(s: pd.Series):
    codes = [c for v in s.dropna() for c in v.split(";")]
    return ";".join(dict.fromkeys(codes)) if codes else np.nan


def build_episodes(stays: pd.DataFrame, cfg: ReadmitConfig) -> pd.DataFrame:
    """All admission episodes under the outcome definition, with each patient's next admission date."""
    adm = stays[is_admission(stays, cfg.outcome_definition)]
    adm = adm[~(adm["los_days"] < 0)]  # negative length of stay = data error
    if cfg.outcome_requires_discharge:
        adm = adm[adm["discharge"].notna()]
    eps = to_episodes(adm, cfg.merge_transfers)
    eps["next_admit"] = eps.groupby("patient_key")["admit"].shift(-1)
    return eps


def label_readmissions(stays: pd.DataFrame, cfg: ReadmitConfig, eps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Index episodes (discharged, meeting the admission definition) with readmission labels and exclusions."""
    eps = build_episodes(stays, cfg) if eps is None else eps
    idx = eps[eps["discharge"].notna() & is_admission(eps, cfg.definition)].copy()
    idx["days_to_next"] = (idx["next_admit"].dt.normalize() - idx["discharge"].dt.normalize()).dt.days
    idx["readmit30"] = idx["days_to_next"].between(cfg.window_min_days, cfg.window_max_days)

    data_start, data_end = stays["first_event"].min(), stays["last_event"].max()
    cutoff = data_end - pd.Timedelta(days=cfg.followup_days)
    idx["exclusion"] = np.select(
        [idx["disposition"].isin(cfg.excluded_dispositions), idx["discharge"] > cutoff],
        ["disposition " + idx["disposition"], f"discharged in last {cfg.followup_days} days"],
        default="",
    )
    idx["eligible"] = idx["exclusion"] == ""
    idx["full_lookback"] = idx["admit"] >= data_start + pd.Timedelta(days=cfg.lookback_days)

    # secondary outcome: an ER-only visit within the window after discharge
    er = stays.loc[is_er_only(stays, cfg.outcome_definition, cfg), ["patient_key", "admit"]]
    m = idx.reset_index()[["episode_key", "patient_key", "discharge"]].merge(er, on="patient_key")
    gap = m["admit"] - m["discharge"]
    hit = m.loc[(gap > pd.Timedelta(0)) & (gap.dt.days <= cfg.window_max_days), "episode_key"]
    idx["er_visit_30d"] = idx.index.isin(hit)
    return idx[INDEX_COLUMNS]
