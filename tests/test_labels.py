from dataclasses import fields

import pytest

from readmit.config import ReadmitConfig
from readmit.pipeline import run_raw
from synthetic import make_raw, row, stay

DEFAULT, LEGACY = ReadmitConfig(), ReadmitConfig.legacy()
B = dict(member="TEST PATIENT B", dob="05/06/1960", episode="EB")


def index_of(*rows, cfg=DEFAULT, pad=True):
    idx = run_raw(make_raw(*rows, pad=pad), cfg).index
    return idx.sort_values(["patient_key", "admit"])


@pytest.mark.parametrize("gap,expected", [(1, True), (10, True), (30, True), (31, False)])
def test_readmission_window(gap, expected):
    idx = index_of(stay(0, 3), stay(3 + gap, 3 + gap + 2))
    first = idx.iloc[0]
    assert first["days_to_next"] == gap and first["readmit30"] == expected


def test_hospice_and_end_of_data_exclusions():
    idx = index_of(stay(0, 3, disposition="Hospice"), stay(380, 385, **B))
    assert idx["exclusion"].tolist() == ["disposition Hospice", "discharged in last 30 days"]
    assert not idx["eligible"].any()


def test_transfer_chain_is_one_episode():
    rows = [stay(0, 5, disposition="Transferred to acute care", facility="Hospital X"),
            stay(5, 9, facility="Hospital Y"),
            stay(20, 22)]
    idx = index_of(*rows)
    assert len(idx) == 2
    ep = idx.iloc[0]
    assert ep["n_stays"] == 2 and ep["n_facilities"] == 2 and ep["disposition"] == "Home"
    assert ep["los_days"] == 9 and ep["days_to_next"] == 11 and ep["readmit30"]

    legacy = index_of(*rows, cfg=LEGACY)
    assert len(legacy) == 3
    assert legacy["days_to_next"].tolist()[:2] == [0, 11]
    assert legacy["readmit30"].tolist()[:2] == [False, True]


def test_hospice_discharge_in_transfer_chain_excludes_episode():
    rows = [stay(0, 5, disposition="Discharged to Hospice", facility="Hospital X"),
            stay(5, 9, disposition=None, facility="Hospice House"), stay(20, 22)]
    idx = index_of(*rows)
    assert idx.iloc[0]["exclusion"] == "disposition Hospice" and idx.iloc[0]["n_stays"] == 2


def test_overlapping_stays_merge():
    idx = index_of(stay(0, 5, facility="Hospital X"), stay(3, 8, facility="Hospital Y"))
    assert len(idx) == 1 and idx.iloc[0]["los_days"] == 8


def test_transfer_into_stay_without_discharge_is_not_an_index():
    rows = [stay(0, 5, facility="Hospital X"), row("Admission", 5, facility="Hospital Y", admit_day=5)]
    assert len(index_of(*rows)) == 0


def test_strict_definition_applies_to_index_and_outcome():
    rows = [stay(0, 3), stay(10, 11, ptclass="Emergency")]
    strict = index_of(*rows, cfg=ReadmitConfig(definition="strict"))
    assert len(strict) == 1
    assert not strict.iloc[0]["readmit30"] and strict.iloc[0]["er_visit_30d"]

    broad = index_of(*rows)
    assert len(broad) == 2 and broad.iloc[0]["readmit30"]

    legacy_strict = index_of(*rows, cfg=ReadmitConfig.legacy(definition="strict"))
    assert len(legacy_strict) == 1 and legacy_strict.iloc[0]["readmit30"]


def test_readmission_without_recorded_discharge_counts():
    rows = [stay(0, 3), row("Admission", 12, admit_day=12)]
    assert index_of(*rows).iloc[0]["readmit30"]
    assert not index_of(*rows, cfg=LEGACY).iloc[0]["readmit30"]


@pytest.mark.parametrize("gap,expected", [(5, True), (40, False)])
def test_er_visit_after_discharge(gap, expected):
    er = row("Registration", 3 + gap, ptclass="Emergency", admit_day=3 + gap, episode="E2")
    idx = index_of(stay(0, 3), er)
    assert idx.iloc[0]["er_visit_30d"] == expected and not idx.iloc[0]["readmit30"]


def test_lookback_flag():
    idx = index_of(stay(0, 3), stay(200, 203, **B))
    assert idx.set_index("admit")["full_lookback"].sort_index().tolist() == [False, True]


def test_negative_length_of_stay_dropped():
    idx = index_of(stay(5, 3))
    assert len(idx) == 0


def test_legacy_turns_every_fix_off():
    fix_fields = ["merge_transfers", "apply_all_cancels", "outcome_same_definition", "dx_from_last",
                  "discharge_from_event", "chronological_order", "terminal_disposition_wins"]
    assert all(getattr(DEFAULT, f) for f in fix_fields) and not any(getattr(LEGACY, f) for f in fix_fields)
    assert LEGACY.outcome_requires_discharge and LEGACY.blank_class_is_er
    assert not DEFAULT.outcome_requires_discharge and not DEFAULT.blank_class_is_er
    assert {f.name for f in fields(ReadmitConfig)} >= set(fix_fields)


def test_index_output_has_no_identifiers():
    idx = index_of(stay(0, 3), stay(10, 12))
    text = idx.astype(str).to_string() + " ".join(idx.columns)
    assert "TEST PATIENT" not in text and "01/02/1950" not in text and "MEMBER" not in text
