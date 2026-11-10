"""Aggregate checks behind the cleaning rules. Prints counts only, never values tied to a patient."""
import pandas as pd

from .normalize import disposition_group, parse_dates, patient_key


def _section(title: str) -> None:
    print(f"\n---- {title} ----")


def run(raw: pd.DataFrame, events: pd.DataFrame) -> None:
    _section("Event and class counts")
    print(raw["Event"].value_counts(dropna=False).to_string())
    print(raw["Source Ptclass"].value_counts(dropna=False).to_string())

    _section("Patient key")
    print("rows without name or DOB:", len(raw) - len(events))
    dob_norm = parse_dates(raw["DOB"]).dt.strftime("%m/%d/%Y")
    print("patients with raw DOB string:", patient_key(raw["MEMBER"], raw["DOB"]).nunique(),
          "| with normalized DOB:", patient_key(raw["MEMBER"], dob_norm).nunique())

    _section("Row order")
    ev = events.sort_values("row")
    back = ev.groupby("stay_key")["event_date"].diff() < pd.Timedelta(0)
    print("rows dated before the previous row of the same stay:", int(back.sum()))

    _section("Cancels")
    for c in ["Cancel Admission", "Cancel Discharge", "Cancel Transfer"]:
        rows = ev[ev["event"] == c]
        target = {"Cancel Admission": "Admission", "Cancel Discharge": "Discharge"}.get(c, "Transfer")
        prior = later = 0
        for _, r in rows.iterrows():
            same = ev[(ev["stay_key"] == r["stay_key"]) & ~ev["event"].str.startswith("Cancel", na=False)]
            hits = same[same["event"].str.startswith(target, na=False)]
            prior += bool((hits["event_date"] <= r["event_date"]).any())
            later += bool((hits["event_date"] > r["event_date"]).any())
        print(f"{c}: {len(rows)} | target event earlier in stay: {prior} | later in stay: {later}"
              f" | cancel row has discharge ts: {int(rows['discharge_ts'].notna().sum())}"
              f" | cancel row has no admit date: {int(rows['admit_ts'].isna().sum())}")
        if c == "Cancel Transfer":
            stays = ev[ev["stay_key"].isin(rows["stay_key"])]
            print("  events in those stays:", stays["event"].value_counts().to_dict())

    _section("Rows missing an admit date")
    miss = events[events["admit_ts"].isna()]
    dated = events[events["admit_ts"].notna()]
    n_dates = dated.groupby(["patient_key", "episode_id", "facility"])["admit_ts"].agg(
        lambda s: s.dt.normalize().nunique())
    key = pd.MultiIndex.from_frame(miss[["patient_key", "episode_id", "facility"]])
    matched = n_dates.reindex(key)
    print(f"rows: {len(miss)} | same patient+episode+facility has exactly 1 admit date: {int((matched == 1).sum())}"
          f" | several dates: {int((matched > 1).sum())} | none: {int(matched.isna().sum())}")

    _section("Disposition grouping")
    disp = raw["Discharge Disposition"]
    grp = disp.map(disposition_group)
    print(pd.DataFrame({"rows": grp.value_counts(),
                        "spellings": disp.groupby(grp).nunique()}).fillna(0).astype(int).to_string())
    hh = disp[grp == "Home Health"].str.upper()
    print("Home Health rows without HOME or HHS in the value:",
          int((~hh.str.contains("HOME") & ~hh.str.contains("HHS")).sum()))

    _section("Diagnosis code lists")
    dx = raw["All Diagnosis Codes"].dropna()
    print("non-null rows:", len(dx), {d: int(dx.str.contains(d, regex=False).sum()) for d in [",", ";", "|", " ", "."]})
