import numpy as np
import pandas as pd
import pytest

from readmit.charlson import charlson_frame, conditions_for
from readmit.config import ReadmitConfig
from readmit.evaluate import calibration_intercept, calibration_slope, cluster_bootstrap, metrics
from readmit.features import LEAKY, build_features, dx_chapter
from readmit.lace import charlson_points, compute_lace, er_points, los_points
from readmit.pipeline import run_raw
from synthetic import make_raw, row, stay

B = dict(member="TEST PATIENT B", dob="05/06/1960", episode="EB")


@pytest.mark.parametrize("codes,expected", [
    (["I509"], {"chf"}), (["I5023", "E1165"], {"chf", "diab"}), (["E1122", "E119"], {"diab_cc"}),
    (["N183"], {"renal"}), (["C787", "C349"], {"metastatic"}), (["K766", "K740"], {"severe_liver"}),
    (["I252"], {"mi"}), (["J449"], {"cpd"}), (["Z9989"], set()), (["123456"], set()),
])
def test_quan_conditions_with_hierarchy(codes, expected):
    assert conditions_for(codes) == expected


def test_charlson_weights():
    f = charlson_frame(pd.Series(["I509;E1122;C787", np.nan]))
    assert f["charlson"].tolist() == [1 + 2 + 6, 0]


def test_lace_point_tables():
    assert los_points(pd.Series([0, 1, 2, 3, 4, 6, 7, 13, 14, 40])).tolist() == [0, 1, 2, 3, 4, 4, 5, 5, 7, 7]
    assert charlson_points(pd.Series([0, 1, 3, 4, 9])).tolist() == [0, 1, 3, 5, 5]
    assert er_points(pd.Series([0, 2, 4, 7])).tolist() == [0, 2, 4, 4]


def er(day, **kw):
    return row("Registration", day, ptclass="Emergency", admit_day=day, episode=f"ER{day}", **kw)


def test_acute_and_prior_er_visits():
    # ER visit the day before admission at another hospital = acute; 3 earlier ER days in 6 months, 1 too old
    rows = [er(-200), er(-100), er(-40), er(-40, facility="Other Hospital"), er(-10),
            er(-1, facility="Other Hospital"), stay(0, 4, all_dx="I50.9")]
    res = run_raw(make_raw(*rows), ReadmitConfig())
    lace = compute_lace(res.index, res.stays).iloc[0]
    assert lace["acute"] and lace["er_visits_6m"] == 3 and lace["er_visits_30d"] == 1
    assert lace["lace"] == 4 + 3 + 1 + 3


def test_features_use_only_history_before_admission():
    rows = [stay(-30, -25), stay(0, 4), stay(10, 12), er(20)]  # later stays must not leak into the first index
    res = run_raw(make_raw(*rows), ReadmitConfig())
    idx = res.index.sort_values("admit")
    X = build_features(idx, res.episodes, res.stays)
    first = X.loc[idx.index[1]]
    assert first["prior_adm_6m"] == 1 and first["days_since_last_discharge"] == 25 and first["er_visits_6m"] == 0
    assert not LEAKY & set(X.columns)


def test_compact_features_exist_and_folds_never_share_patients():
    from readmit.features import COMPACT_FEATURES
    from readmit.model import outer_folds
    res = run_raw(make_raw(stay(0, 4), stay(10, 12), stay(0, 3, **B)), ReadmitConfig())
    assert set(COMPACT_FEATURES) <= set(build_features(res.index, res.episodes, res.stays).columns)
    groups = np.repeat(np.arange(60), 3)
    y = np.tile([0, 0, 1], 60)
    for train, test in outer_folds(y, groups, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_dx_chapter():
    assert [dx_chapter(c) for c in ["I50.9", "J44.1", "C34.90", "D50.0", None, "Z99"]] == [
        "circulatory", "respiratory", "neoplasm", "blood", "missing", "other"]


def test_metrics_and_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 4000)
    y = rng.binomial(1, p)
    m = metrics(y, p)
    assert 0.6 < m["auroc"] < 0.8
    assert abs(calibration_intercept(y, p)) < 0.15 and abs(calibration_slope(y, p) - 1) < 0.15
    assert calibration_slope(y, np.clip(p * 1.5 - 0.1, 0.01, 0.99)) < 1  # too extreme -> slope < 1


def test_cluster_bootstrap_resamples_patients():
    y = np.array([0, 1] * 50)
    groups = np.repeat(np.arange(25), 4)
    ci = cluster_bootstrap(groups, lambda pick: {"rate": y[pick].mean(), "n": len(pick)}, n_boot=200)
    assert ci["n"] == (100, 100) and ci["rate"][0] <= 0.5 <= ci["rate"][1]
