import pandas as pd

from readmit.config import ReadmitConfig
from readmit.normalize import normalize_ens
from readmit.stays import build_stays, is_admission, is_er_only
from synthetic import BASE, make_raw, row, stay

DEFAULT, LEGACY = ReadmitConfig(), ReadmitConfig.legacy()


def build(*rows, cfg=DEFAULT):
    return build_stays(normalize_ens(make_raw(*rows, pad=False)), cfg)


def day(n):
    return pd.Timestamp(BASE) + pd.Timedelta(days=n)


def test_reused_episode_ids_split_by_admit_date():
    st = build(stay(0, 3, episode="E1"), stay(20, 22, episode="E1"))
    assert len(st) == 2


def test_same_admit_date_different_episode_ids_is_one_stay():
    st = build(row("Admission", 0, admit_day=0, episode="E1"),
               row("Discharge", 3, admit_day=0, discharge_day=3, disposition="Home", episode="E2"))
    assert len(st) == 1
    assert st["discharge"].iloc[0].normalize() == day(3)


def test_missing_admit_date_falls_back_to_episode_id():
    st = build(row("Admission", 0, admit="1/1/1900", episode="E9"),
               row("Discharge", 3, admit="1/1/1900", discharge_day=3, episode="E9"))
    assert len(st) == 1
    assert "_EPE9" in st.index[0]
    assert st["admit"].iloc[0] == day(0)  # admit falls back to the first event date


def test_missing_discharge_ts_uses_discharge_event_date():
    rows = [row("Admission", 0, admit_day=0),
            row("Discharge", 3, admit_day=0, discharge="1/1/1900"),
            row("Update", 6, admit_day=0)]
    assert build(rows)["discharge"].iloc[0] == day(3)
    assert build(rows, cfg=LEGACY)["discharge"].iloc[0] == day(6)  # legacy: last event of the stay


def test_cancel_admission_drops_stay():
    st = build(stay(0, 3), row("Cancel Admission", 1, admit_day=0))
    assert len(st) == 0


def test_cancel_discharge_then_new_discharge():
    rows = [row("Admission", 0, admit_day=0),
            row("Discharge", 2, admit_day=0, discharge_day=2),
            row("Cancel Discharge", 2, admit_day=0, discharge_day=2),
            row("Discharge", 5, admit_day=0, discharge_day=5, disposition="Home")]
    assert build(rows)["discharge"].iloc[0].normalize() == day(5)


def test_cancel_discharge_without_new_discharge_leaves_stay_open():
    rows = [row("Admission", 0, admit_day=0),
            row("Discharge", 2, admit_day=0, discharge_day=2),
            row("Cancel Discharge", 3, admit_day=0)]
    assert pd.isna(build(rows)["discharge"].iloc[0])
    assert build(rows, cfg=LEGACY)["discharge"].iloc[0].normalize() == day(2)


def test_cancel_transfer_reverses_transfer_to_inpatient():
    rows = [row("Registration", 0, ptclass="Emergency", admit_day=0),
            row("Transfer to Inpatient", 0, ptclass="Emergency", admit_day=0),
            row("Cancel Transfer", 1, ptclass="Emergency", admit_day=0),
            row("Discharge", 1, ptclass="Emergency", admit_day=0, discharge_day=1)]
    assert not build(rows)["has_admission_event"].iloc[0]
    assert build(rows, cfg=LEGACY)["has_admission_event"].iloc[0]


def test_blank_class_is_not_er_by_default():
    rows = [row("Registration", 0, ptclass=None, admit_day=0)]
    assert not is_er_only(build(rows), "broad", DEFAULT).iloc[0]
    assert is_er_only(build(rows, cfg=LEGACY), "broad", LEGACY).iloc[0]


def test_emergency_class_admission_is_broad_only():
    st = build(stay(0, 2, ptclass="Emergency"))
    assert is_admission(st, "broad").iloc[0]
    assert not is_admission(st, "strict").iloc[0]
    assert is_er_only(st, "strict", DEFAULT).iloc[0]


def test_hospice_disposition_wins_over_later_value():
    rows = [row("Admission", 0, admit_day=0),
            row("Discharge", 3, admit_day=0, discharge_day=3, disposition="Hospice - Home"),
            row("Update", 4, admit_day=0, disposition="Transferred elsewhere")]
    assert build(rows)["disposition"].iloc[0] == "Hospice"
    assert build(rows, cfg=LEGACY)["disposition"].iloc[0] == "Other"


def test_later_discharge_beats_earlier_still_a_patient():
    rows = [row("Discharge", 3, admit_day=0, discharge_day=3, disposition="Home"),
            row("Admission", 0, admit_day=0, disposition="Still a patient")]  # file order is not date order
    assert build(rows)["disposition"].iloc[0] == "Home"
    assert build(rows, cfg=LEGACY)["disposition"].iloc[0] == "Still a patient"


def test_dx_codes_from_last_event_up_to_discharge():
    rows = [row("Admission", 0, admit_day=0, dx="I10", all_dx="I10"),
            row("Discharge", 3, admit_day=0, discharge_day=3, dx="I50.9", all_dx="I10, I50.9"),
            row("Update", 9, admit_day=0, dx="Z99.9", all_dx="Z99.9")]
    st = build(rows)
    assert st["dx_codes"].iloc[0] == "I10;I509" and st["primary_dx"].iloc[0] == "I50.9"
    assert build(rows, cfg=LEGACY)["dx_codes"].iloc[0] == "I10"


def test_dx_grace_days_accepts_codes_after_discharge():
    rows = [row("Admission", 0, admit_day=0, all_dx=";"),
            row("Discharge", 3, admit_day=0, discharge_day=3),
            row("Update", 4, admit_day=0, all_dx="I50.9^N18.3")]
    assert build(rows)["dx_codes"].iloc[0] == "I509;N183"  # default: 1 day grace
    assert pd.isna(build(rows, cfg=ReadmitConfig(dx_grace_days=0))["dx_codes"].iloc[0])
    rows[-1] = row("Update", 5, admit_day=0, all_dx="I50.9^N18.3")  # 2 days after discharge
    assert pd.isna(build(rows)["dx_codes"].iloc[0])
