"""Load a raw payer ENS file. Never prints row-level data."""
from pathlib import Path

import pandas as pd

ENS_COLUMNS = [
    "Episode Id", "EventTime", "Dest Mrn", "MEMBER", "DOB", "Member Address", "Event", "Source Ptclass",
    "Source Facility", "Patient Complaint", "Attending Physician", "Primary Diagnosis Code",
    "Primary Diagnosis Descriptions", "Admit Timestamp", "Discharge Timestamp", "Discharge Disposition",
    "All Diagnosis Codes", "All Diagnosis Descriptions", "All Diagnosis Types", "PCP Name", "PCP Tax ID",
]


def load_ens(path: str | Path) -> pd.DataFrame:
    """Read the Aetna ENS CSV as strings and check that the expected columns are present."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    missing = [c for c in ENS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ENS file is missing columns: {missing}")
    return df
