"""Turn raw ENS rows into a standard event table keyed by a hashed patient ID.

Identifiers (name, DOB, address, MRN, physician and PCP names, free-text complaint) are dropped here, so nothing
downstream ever sees them. Other payers will get their own normalizer that outputs the same EVENT_COLUMNS.
"""
import hashlib
import re

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "row", "patient_key", "stay_key", "episode_id", "facility", "event", "patient_class", "event_date",
    "admit_ts", "discharge_ts", "disposition", "primary_dx", "all_dx", "age",
]
NULL_PLACEHOLDER_YEAR = 1900
MAX_AGE = 90  # HIPAA Safe Harbor: ages over 89 are grouped
_CODE_SPLIT = re.compile(r"[,;|^\s]+")  # the feed mostly uses ^ and ;


def patient_key(member: pd.Series, dob: pd.Series) -> pd.Series:
    """SHA-256 of MEMBER (uppercased, trimmed) + "|" + DOB, first 12 hex chars. Missing name or DOB gives NaN."""
    name = member.fillna("").astype(str).str.upper().str.strip()
    dob = dob.fillna("").astype(str)
    key = (name + "|" + dob).map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:12])
    return key.where((name != "") & (dob.str.strip() != ""))


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse month/day/year dates (with or without times). 1/1/1900 is a null placeholder."""
    d = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    return d.where(d.dt.year > NULL_PLACEHOLDER_YEAR)


def age_at(dob: pd.Series, when: pd.Series) -> pd.Series:
    """Whole years from DOB to `when`, capped at MAX_AGE. The DOB itself is never kept."""
    born = pd.to_datetime(dob, errors="coerce", format="mixed", dayfirst=False)
    had_birthday = (when.dt.month > born.dt.month) | ((when.dt.month == born.dt.month) & (when.dt.day >= born.dt.day))
    years = when.dt.year - born.dt.year - (~had_birthday).astype(int)
    return years.clip(upper=MAX_AGE)


def disposition_group(x) -> str:
    if pd.isna(x):
        return "Missing"
    u = str(x).upper()
    if "HOSPICE" in u:
        return "Hospice"
    if "SNF" in u or "SKILLED" in u or "NURSING" in u:
        return "SNF"
    if "HHS" in u or "HEALTH" in u:
        return "Home Health"
    if "ALF" in u or "ASSISTED" in u:
        return "ALF"
    if "STILL" in u:
        return "Still a patient"
    if "EXPIRED" in u or "DIED" in u or "DEATH" in u:
        return "Expired"
    if "AMA" in u or "AGAINST" in u:
        return "AMA"
    if u.startswith("HOM") or "HOME" in u or u.strip() == "1":
        return "Home"
    return "Other"


def parse_codes(s) -> tuple[str, ...]:
    """Split a diagnosis code list into normalized ICD-10 codes (uppercase, no dots), keeping order."""
    if pd.isna(s):
        return ()
    codes = (c.replace(".", "").upper() for c in _CODE_SPLIT.split(str(s)))
    return tuple(dict.fromkeys(c for c in codes if c))


def normalize_ens(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Aetna ENS rows to EVENT_COLUMNS. Rows without a name or DOB are dropped (no patient key)."""
    ev = pd.DataFrame({
        "row": np.arange(len(raw)),
        "patient_key": patient_key(raw["MEMBER"], raw["DOB"]).to_numpy(),
        "episode_id": raw["Episode Id"].to_numpy(),
        "facility": raw["Source Facility"].to_numpy(),
        "event": raw["Event"].to_numpy(),
        "patient_class": raw["Source Ptclass"].to_numpy(),
        "event_date": parse_dates(raw["EventTime"]).to_numpy(),
        "admit_ts": parse_dates(raw["Admit Timestamp"]).to_numpy(),
        "discharge_ts": parse_dates(raw["Discharge Timestamp"]).to_numpy(),
        "disposition": raw["Discharge Disposition"].map(disposition_group).to_numpy(),
        "primary_dx": raw["Primary Diagnosis Code"].to_numpy(),
        "all_dx": raw["All Diagnosis Codes"].to_numpy(),
    })
    ev["age"] = age_at(raw["DOB"], parse_dates(raw["EventTime"])).to_numpy()
    ev = ev[ev["patient_key"].notna()].copy()
    # Episode Id is reused across separate hospitalizations, so a stay = patient + facility + admit date,
    # falling back to the Episode Id only when the admit date is missing
    admit_day = ev["admit_ts"].dt.strftime("%Y%m%d")
    ev["stay_key"] = (ev["patient_key"] + "_" + ev["facility"].fillna("NA") + "_"
                      + admit_day.fillna("EP" + ev["episode_id"].fillna("NA")))
    return ev[EVENT_COLUMNS]
