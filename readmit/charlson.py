"""Charlson comorbidity from ICD-10 codes.

Conditions use the Quan et al. (2005) ICD-10 coding algorithm (Med Care 43:1130), matched by code prefix so
ICD-10-CM codes (e.g. I50.23, E11.65) map to their WHO ICD-10 parents. Weights are the original Charlson (1987)
weights. Hierarchy: severe liver disease, diabetes with complications and metastatic tumor replace their milder
counterparts. Codes that are not ICD-10 (e.g. 6-digit numeric codes in the feed) simply do not match.
"""
import re

import pandas as pd

# condition: (weight, Quan ICD-10 codes; "I425-I429" is an inclusive range of prefixes)
QUAN_ICD10 = {
    "mi": (1, "I21 I22 I252"),
    "chf": (1, "I099 I110 I130 I132 I255 I420 I425-I429 I43 I50 P290"),
    "pvd": (1, "I70 I71 I731 I738 I739 I771 I790 I792 K551 K558 K559 Z958 Z959"),
    "cevd": (1, "G45 G46 H340 I60-I69"),
    "dementia": (1, "F00-F03 F051 G30 G311"),
    "cpd": (1, "I278 I279 J40-J47 J60-J67 J684 J701 J703"),
    "rheum": (1, "M05 M06 M315 M32-M34 M351 M353 M360"),
    "pud": (1, "K25-K28"),
    "mild_liver": (1, "B18 K700-K703 K709 K713-K715 K717 K73 K74 K760 K762-K764 K768 K769 Z944"),
    "diab": (1, "E100 E101 E106 E108 E109 E110 E111 E116 E118 E119 E120 E121 E126 E128 E129 "
                "E130 E131 E136 E138 E139 E140 E141 E146 E148 E149"),
    "diab_cc": (2, "E102-E105 E107 E112-E115 E117 E122-E125 E127 E132-E135 E137 E142-E145 E147"),
    "hemiplegia": (2, "G041 G114 G801 G802 G81 G82 G830-G834 G839"),
    "renal": (2, "I120 I131 N032-N037 N052-N057 N18 N19 N250 Z490-Z492 Z940 Z992"),
    "cancer": (2, "C00-C26 C30-C34 C37-C41 C43 C45-C58 C60-C76 C81-C85 C88 C90-C97"),
    "severe_liver": (3, "I850 I859 I864 I982 K704 K711 K721 K729 K765 K766 K767"),
    "metastatic": (6, "C77-C80"),
    "hiv": (6, "B20-B22 B24"),
}
CONDITIONS = list(QUAN_ICD10)
WEIGHTS = {c: w for c, (w, _) in QUAN_ICD10.items()}
HIERARCHY = {"severe_liver": "mild_liver", "diab_cc": "diab", "metastatic": "cancer"}  # severe -> milder


def _expand(spec: str) -> set[str]:
    out = set()
    for token in spec.split():
        if "-" not in token:
            out.add(token)
            continue
        lo, hi = token.split("-")
        letter, a, b = lo[0], lo[1:], hi[1:]
        out.update(f"{letter}{n:0{len(a)}d}" for n in range(int(a), int(b) + 1))
    return out


PREFIXES = {c: _expand(spec) for c, (_, spec) in QUAN_ICD10.items()}


def conditions_for(codes) -> set[str]:
    """Charlson conditions present in an iterable of normalized codes (uppercase, no dots)."""
    found = set()
    for code in codes:
        for cond, prefixes in PREFIXES.items():
            if any(code.startswith(p) for p in prefixes):
                found.add(cond)
    for severe, mild in HIERARCHY.items():
        if severe in found:
            found.discard(mild)
    return found


def charlson_frame(dx_codes: pd.Series) -> pd.DataFrame:
    """One 0/1 column per condition plus `charlson` (weighted score) for a Series of ';'-joined code lists."""
    found = dx_codes.map(lambda s: conditions_for(s.split(";")) if isinstance(s, str) and s else set())
    out = pd.DataFrame({f"cci_{c}": found.map(lambda f, c=c: int(c in f)) for c in CONDITIONS}, index=dx_codes.index)
    out["charlson"] = sum(out[f"cci_{c}"] * w for c, w in WEIGHTS.items())
    return out


def is_icd10(code: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]\d[0-9A-Z]{1,5}", code))
