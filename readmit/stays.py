"""Collapse normalized events into one row per hospital stay."""
import numpy as np
import pandas as pd

from .config import ReadmitConfig
from .normalize import parse_codes

ADMISSION_EVENTS = ("Admission", "Transfer to Inpatient")
TERMINAL_DISPOSITIONS = ("Expired", "Hospice")


def final_disposition(s: pd.Series, skip_missing: bool = True) -> str:
    """Expired or Hospice anywhere wins over later values; otherwise the last (known) value."""
    for t in TERMINAL_DISPOSITIONS:
        if (s == t).any():
            return t
    if skip_missing:
        s = s[s != "Missing"]
    return s.iloc[-1] if len(s) else "Missing"


def apply_cancels(ev: pd.DataFrame, cfg: ReadmitConfig) -> pd.DataFrame:
    """Reverse cancel events, then drop the cancel rows. `ev` must be in chronological order with a `pos` column.

    - Cancel Admission: the whole stay is dropped (always applied).
    - Cancel Discharge: removes the latest Discharge event at or before the cancel, and clears the discharge
      timestamps on rows up to the cancel (plus any later copy of the cancelled timestamp).
    - Cancel Transfer: removes the latest Transfer event at or before the cancel.
    """
    ev = ev.copy()
    is_cancel = ev["event"].str.startswith("Cancel", na=False)
    cancelled_stays = set(ev.loc[ev["event"] == "Cancel Admission", "stay_key"])
    removed: set = set()
    if cfg.apply_all_cancels:
        for _, c in ev[ev["event"].isin(["Cancel Discharge", "Cancel Transfer"])].iterrows():
            in_stay = ev["stay_key"] == c["stay_key"]
            prior = ev[in_stay & ~is_cancel & (ev["pos"] <= c["pos"]) & ~ev.index.isin(removed)]
            if c["event"] == "Cancel Transfer":
                transfers = prior[prior["event"].str.startswith("Transfer", na=False)]
                if len(transfers):
                    removed.add(transfers.index[-1])
                continue
            discharges = prior[prior["event"] == "Discharge"]
            cancelled_ts = c["discharge_ts"]
            if len(discharges):
                removed.add(discharges.index[-1])
                if pd.isna(cancelled_ts):
                    cancelled_ts = discharges["discharge_ts"].iloc[-1]
            clear = in_stay & ((ev["pos"] <= c["pos"]) | (ev["discharge_ts"] == cancelled_ts))
            ev.loc[clear, "discharge_ts"] = pd.NaT
    keep = ~is_cancel & ~ev["stay_key"].isin(cancelled_stays) & ~ev.index.isin(removed)
    return ev[keep]


def build_stays(events: pd.DataFrame, cfg: ReadmitConfig) -> pd.DataFrame:
    """One row per stay (patient + facility + admit date), indexed by stay_key."""
    order = ["event_date", "row"] if cfg.chronological_order else ["row"]
    ev = events.sort_values(order, na_position="last", kind="stable").reset_index(drop=True)
    ev["pos"] = np.arange(len(ev))
    ev = apply_cancels(ev, cfg)
    ev = ev.assign(
        is_admission_event=ev["event"].isin(ADMISSION_EVENTS),
        is_discharge_event=ev["event"].eq("Discharge"),
        is_inpatient_class=ev["patient_class"].eq("Inpatient"),
        is_er_class=ev["patient_class"].eq("Emergency"),
        has_class=ev["patient_class"].notna(),
    )
    ev["discharge_event_date"] = ev["event_date"].where(ev["is_discharge_event"])
    ev["disposition_known"] = ev["disposition"].where(ev["disposition"] != "Missing")
    # parse per row so an empty list (";") never wins over a row that has codes
    codes = ev["all_dx"].map(parse_codes)
    ev["dx_codes"] = codes.map(";".join).where(codes.map(len) > 0)
    ev["primary_dx"] = ev["primary_dx"].str.strip().replace("", None)

    st = ev.groupby("stay_key").agg(
        patient_key=("patient_key", "first"),
        facility=("facility", "first"),
        admit_ts=("admit_ts", "min"),
        first_event=("event_date", "min"),
        last_event=("event_date", "max"),
        discharge_ts=("discharge_ts", "max"),
        discharge_event_date=("discharge_event_date", "max"),
        has_inpatient_class=("is_inpatient_class", "any"),
        has_admission_event=("is_admission_event", "any"),
        has_discharge_event=("is_discharge_event", "any"),
        has_er_class=("is_er_class", "any"),
        n_class=("has_class", "sum"),
        n_er_class=("is_er_class", "sum"),
        disposition=("disposition_known", "last"),
        age=("age", "min"),
        n_events=("row", "size"),
    )
    st["admit"] = st["admit_ts"].fillna(st["first_event"])
    # missing discharge timestamp: fall back to the Discharge event's date when one exists
    fallback = st["discharge_event_date"] if cfg.discharge_from_event else st["last_event"]
    st["discharge"] = st["discharge_ts"].fillna(fallback.where(st["has_discharge_event"]))
    st["los_days"] = (st["discharge"].dt.normalize() - st["admit"].dt.normalize()).dt.days
    st["all_er"] = (st["n_class"] > 0) & (st["n_er_class"] == st["n_class"])
    st["no_class"] = st["n_class"] == 0
    st["disposition"] = st["disposition"].fillna("Missing")
    if cfg.terminal_disposition_wins:
        st["disposition"] = ev.groupby("stay_key")["disposition"].agg(final_disposition)

    if cfg.dx_from_last:
        # last codes recorded by the discharge date + grace days (later updates were not available at scoring time)
        cutoff = st["discharge"].dt.normalize() + pd.Timedelta(days=cfg.dx_grace_days)
        cutoff = ev[["stay_key"]].join(cutoff.rename("cutoff"), on="stay_key")["cutoff"]
        dx_rows = ev[cutoff.isna() | ev["event_date"].isna() | (ev["event_date"] <= cutoff)]
        dx = dx_rows.groupby("stay_key")[["primary_dx", "dx_codes"]].last()
    else:
        dx = ev.groupby("stay_key")[["primary_dx", "dx_codes"]].first()
    st = st.join(dx)

    cols = ["patient_key", "facility", "age", "admit", "discharge", "los_days", "first_event", "last_event",
            "has_inpatient_class", "has_admission_event", "has_er_class", "all_er", "no_class",
            "has_discharge_event", "disposition", "primary_dx", "dx_codes", "n_events"]
    return st[cols].sort_values(["patient_key", "admit"], kind="stable")


def is_admission(df: pd.DataFrame, definition: str) -> pd.Series:
    """broad: Inpatient class or an Admission / Transfer to Inpatient event. strict: Inpatient class only."""
    if definition == "strict":
        return df["has_inpatient_class"]
    return df["has_inpatient_class"] | df["has_admission_event"]


def is_er_only(stays: pd.DataFrame, definition: str, cfg: ReadmitConfig) -> pd.Series:
    """ER visit that did not become an admission under `definition`."""
    er_class = stays["all_er"] | (stays["no_class"] & cfg.blank_class_is_er)
    return ~is_admission(stays, definition) & er_class
