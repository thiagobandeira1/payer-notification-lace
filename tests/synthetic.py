"""Builders for small synthetic ENS files. Fake names and dates only; never real data."""
from datetime import date, timedelta

import pandas as pd

from readmit.ingest import ENS_COLUMNS

BASE = date(2025, 3, 1)


def mdy(day: int, time: str | None = None) -> str:
    """Day offset from BASE as the feed's month/day/year string."""
    d = BASE + timedelta(days=day)
    s = f"{d.month:02d}/{d.day:02d}/{d.year}"
    return f"{s} {time}" if time else s


def row(event, day, member="TEST PATIENT A", dob="01/02/1950", ptclass="Inpatient", facility="General Hospital",
        admit_day=None, discharge_day=None, disposition=None, episode="E1", dx=None, all_dx=None,
        admit=None, discharge=None, mrn="MRN-1"):
    r = dict.fromkeys(ENS_COLUMNS)
    r.update({
        "Episode Id": episode, "EventTime": mdy(day), "Dest Mrn": mrn, "MEMBER": member, "DOB": dob,
        "Member Address": "1 Fake St", "Event": event, "Source Ptclass": ptclass, "Source Facility": facility,
        "Attending Physician": "DR FAKE", "PCP Name": "FAKE PCP", "Patient Complaint": "fake complaint",
        "Primary Diagnosis Code": dx, "All Diagnosis Codes": all_dx, "Discharge Disposition": disposition,
        "Admit Timestamp": admit if admit is not None else (mdy(admit_day, "08:00") if admit_day is not None else None),
        "Discharge Timestamp": discharge if discharge is not None else (
            mdy(discharge_day, "14:00") if discharge_day is not None else None),
    })
    return r


def stay(admit_day, discharge_day, disposition="Home", admit_event="Admission", **kw):
    """Admission + Discharge rows for one hospital stay."""
    return [row(admit_event, admit_day, admit_day=admit_day, **kw),
            row("Discharge", discharge_day, admit_day=admit_day, discharge_day=discharge_day,
                disposition=disposition, **kw)]


def make_raw(*row_lists, pad=True) -> pd.DataFrame:
    """Combine rows into a raw ENS frame. `pad` adds an unrelated event a year later so data end is far away."""
    rows = [r for rl in row_lists for r in (rl if isinstance(rl, list) else [rl])]
    if pad:
        rows.append(row("Registration", 400, member="TEST PATIENT Z", ptclass="Outpatient", episode="EZ",
                        admit_day=400))
    return pd.DataFrame(rows, columns=ENS_COLUMNS, dtype="str")
