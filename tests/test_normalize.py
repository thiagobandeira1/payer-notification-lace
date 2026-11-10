import pandas as pd
import pytest

from readmit.normalize import disposition_group, normalize_ens, parse_codes, parse_dates, patient_key
from synthetic import make_raw, row, stay


def key(name, dob):
    return patient_key(pd.Series([name]), pd.Series([dob])).iloc[0]


def test_patient_key_ignores_case_and_outer_spaces():
    k = key("TEST PATIENT A", "01/02/1950")
    assert len(k) == 12
    assert key("  test patient a ", "01/02/1950") == k
    assert key("TEST PATIENT A", "01/03/1950") != k


def test_patient_key_missing_name_or_dob_is_nan():
    assert pd.isna(key(None, "01/02/1950"))
    assert pd.isna(key("TEST PATIENT A", None))


def test_parse_dates_month_first_and_1900_placeholder():
    d = parse_dates(pd.Series(["03/04/2025 10:30", "03/04/2025", "1/1/1900", "1/1/1900 00:00", None]))
    assert d.iloc[0] == pd.Timestamp("2025-03-04 10:30")
    assert d.iloc[1] == pd.Timestamp("2025-03-04")
    assert d.iloc[2:].isna().all()


@pytest.mark.parametrize("raw,group", [
    ("01 - Home or Self Care", "Home"), ("1", "Home"), ("HOM", "Home"),
    ("Discharged to Home Health Service", "Home Health"), ("HHS", "Home Health"),
    ("Skilled Nursing Facility", "SNF"), ("SNF", "SNF"), ("Hospice - Home", "Hospice"),
    ("Expired", "Expired"), ("Left AMA", "AMA"), ("Assisted Living", "ALF"),
    ("Still a patient", "Still a patient"), ("Psychiatric hospital", "Other"), (None, "Missing"),
])
def test_disposition_groups(raw, group):
    assert disposition_group(raw) == group


def test_parse_codes_normalizes_and_dedupes():
    assert parse_codes("I50.9, E11.65;N18.3 | I50.9") == ("I509", "E1165", "N183")
    assert parse_codes("I50.9^E11.65^") == ("I509", "E1165")
    assert parse_codes(";") == () and parse_codes(None) == ()


def test_normalize_drops_identifiers():
    ev = normalize_ens(make_raw(stay(0, 3)))
    for col in ["MEMBER", "DOB", "Member Address", "Dest Mrn", "Attending Physician", "PCP Name", "Patient Complaint"]:
        assert col not in ev.columns
    text = ev.astype(str).to_string()
    assert "TEST PATIENT" not in text and "01/02/1950" not in text and "Fake St" not in text


def test_shared_dest_mrn_does_not_merge_people():
    raw = make_raw(stay(0, 3, mrn="SHARED"), stay(0, 3, member="TEST PATIENT B", mrn="SHARED"), pad=False)
    assert normalize_ens(raw)["patient_key"].nunique() == 2


def test_age_at_event_is_whole_years_and_capped():
    from readmit.normalize import age_at
    when = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-06-01"]))
    ages = age_at(pd.Series(["01/02/1950", "01/02/1950", "01/01/1920"]), when)
    assert ages.tolist() == [74, 75, 90]


def test_rows_without_name_are_dropped():
    raw = make_raw(stay(0, 3), row("Admission", 5, member=None, admit_day=5), pad=False)
    assert len(normalize_ens(raw)) == 2
