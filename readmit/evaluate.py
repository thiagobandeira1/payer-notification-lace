"""Discrimination and calibration metrics with patient-clustered bootstrap confidence intervals."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

N_BOOT = 2000


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: intercept a in logit(P(y)) = a + logit(p), slope fixed at 1 (0 is ideal)."""
    lp, a = _logit(p), 0.0
    for _ in range(50):  # Newton steps on a 1-D logistic likelihood
        q = 1 / (1 + np.exp(-(a + lp)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return a


def calibration_slope(y, p) -> float:
    """Slope b in logit(P(y)) = a + b logit(p) (1 is ideal, <1 means predictions are too extreme)."""
    lr = LogisticRegression(C=np.inf).fit(_logit(p).reshape(-1, 1), y)  # unpenalized
    return float(lr.coef_[0, 0])


def metrics(y, p, calibrated: bool = True) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
    if calibrated:  # p is a probability
        out.update(brier=brier_score_loss(y, p), cal_intercept=calibration_intercept(y, p),
                   cal_slope=calibration_slope(y, p), mean_predicted=p.mean(), observed_rate=y.mean())
    return out


def top_fraction(y, score, frac: float = 0.2) -> dict:
    """Flag the highest-risk `frac` of discharges: share of readmissions caught (sensitivity) and PPV."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    flagged = score >= np.quantile(score, 1 - frac)
    return {"flagged": float(flagged.mean()), "sensitivity": y[flagged].sum() / max(y.sum(), 1),
            "ppv": y[flagged].mean() if flagged.any() else float("nan")}


def threshold(y, score, cut: float) -> dict:
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    flagged = score >= cut
    return {"flagged": float(flagged.mean()), "sensitivity": y[flagged].sum() / max(y.sum(), 1),
            "specificity": ((~flagged) & (y == 0)).sum() / max((y == 0).sum(), 1),
            "ppv": y[flagged].mean() if flagged.any() else float("nan")}


def cluster_bootstrap(groups, stat_fn, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """95% percentile CIs of stat_fn(row_indices) -> dict, resampling patients (groups) with replacement."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {}
    for _ in range(n_boot):
        pick = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        try:
            stats = stat_fn(pick)
        except ValueError:  # resample with a single class
            continue
        for k, v in stats.items():
            draws.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in draws.items()}


def with_ci(point: dict, ci: dict) -> dict:
    return {k: {"estimate": float(v), "ci95": ci.get(k)} for k, v in point.items()}
