"""Settings for building stays and labeling readmissions."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT = RAW_DIR / "ens_notifications.csv"


@dataclass(frozen=True)
class ReadmitConfig:
    # "broad": Ptclass Inpatient OR an Admission / Transfer to Inpatient event.
    # "strict": Ptclass Inpatient only.
    definition: Literal["broad", "strict"] = "broad"
    window_min_days: int = 1
    window_max_days: int = 30
    excluded_dispositions: tuple[str, ...] = ("Hospice", "Expired", "Still a patient")
    # index stays discharged within this many days of the end of the data have no full follow-up window
    followup_days: int = 30
    # stays starting in the first N days of data have a truncated look-back (flagged, not dropped)
    lookback_days: int = 182

    # ---- fixes over an earlier stay-building script (not included); ReadmitConfig.legacy() turns them all off ----
    # merge same-day and overlapping admissions (transfers) into one episode of care
    merge_transfers: bool = True
    # apply Cancel Discharge and Cancel Transfer (Cancel Admission is always applied)
    apply_all_cancels: bool = True
    # the readmission must meet the same admission definition as the index stay
    outcome_same_definition: bool = True
    # an admission counts as a readmission even when its own discharge was never recorded
    outcome_requires_discharge: bool = False
    # a stay with no patient class at all counts as an ER-only visit
    blank_class_is_er: bool = False
    # diagnosis codes from the last event that has them (at discharge) instead of the first
    dx_from_last: bool = True
    # codes recorded up to this many days after discharge still count (scoring runs the morning after discharge)
    dx_grace_days: int = 1
    # missing discharge timestamp falls back to the Discharge event's date (legacy: the stay's last event date)
    discharge_from_event: bool = True
    # read each stay's events by date (the file is not in time order); legacy reads them in file order
    chronological_order: bool = True
    # a Hospice or Expired disposition on any event of a stay wins over later values
    terminal_disposition_wins: bool = True

    @property
    def outcome_definition(self) -> str:
        return self.definition if self.outcome_same_definition else "broad"

    @classmethod
    def legacy(cls, **overrides) -> "ReadmitConfig":
        """Reproduce the earlier stay-building rules (used for comparison only)."""
        base = cls(
            merge_transfers=False,
            apply_all_cancels=False,
            outcome_same_definition=False,
            outcome_requires_discharge=True,
            blank_class_is_er=True,
            dx_from_last=False,
            discharge_from_event=False,
            chronological_order=False,
            terminal_disposition_wins=False,
        )
        return replace(base, **overrides)
