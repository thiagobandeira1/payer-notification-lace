"""Phase 1 steps 2-3: LACE baseline and model comparison. Prints aggregates only; row-level files go to outputs/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROJECT_ROOT, ReadmitConfig  # noqa: E402
from .evaluate import cluster_bootstrap, metrics, threshold, top_fraction, with_ci  # noqa: E402
from .features import COMPACT_FEATURES, build_features  # noqa: E402
from .lace import HIGH_RISK_LACE  # noqa: E402
from .model import cross_validate, fit_final_xgb  # noqa: E402
from .pipeline import PipelineResult  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPEAT_SEEDS = [1, 2, 3, 4]


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"] if d["ci95"] else (float("nan"), float("nan"))
    return f"{d['estimate']:.3f} ({lo:.3f}-{hi:.3f})"


def _out_dir(cfg: ReadmitConfig) -> Path:
    d = OUTPUT_DIR / cfg.definition
    d.mkdir(parents=True, exist_ok=True)
    return d


def eligible_data(res: PipelineResult):
    idx = res.index[res.index["eligible"]]
    X = build_features(idx, res.episodes, res.stays)
    return idx, X, idx["readmit30"].to_numpy(dtype=int), idx["patient_key"].to_numpy()


def lace_report(res: PipelineResult, cfg: ReadmitConfig) -> dict:
    idx, X, y, groups = eligible_data(res)
    lace = X["lace"].to_numpy(dtype=float)
    report = {"n": len(y), "patients": int(len(set(groups))), "readmissions": int(y.sum())}

    def stat(pick):
        return {**metrics(y[pick], lace[pick], calibrated=False),
                **{f"lace>={HIGH_RISK_LACE} {k}": v for k, v in threshold(y[pick], lace[pick], HIGH_RISK_LACE).items()}}

    point = stat(np.arange(len(y)))
    report["overall"] = with_ci(point, cluster_bootstrap(groups, stat))
    for name, mask in [("full 6-month look-back", idx["full_lookback"].to_numpy()),
                       ("has diagnosis codes", X["dx_missing"].to_numpy() == 0)]:
        sub_y, sub_l, sub_g = y[mask], lace[mask], groups[mask]
        ci = cluster_bootstrap(sub_g, lambda p: {"auroc": metrics(sub_y[p], sub_l[p], calibrated=False)["auroc"]})
        report[f"subgroup: {name}"] = {"n": int(mask.sum()), "readmissions": int(sub_y.sum()),
                                       **with_ci({"auroc": metrics(sub_y, sub_l, calibrated=False)["auroc"]}, ci)}
    comp = X[["lace", "los_days", "charlson", "er_visits_6m", "acute"]]
    report["components (mean)"] = comp.mean().round(2).to_dict()
    report["acute from a separate ER stay the day before"] = int((X["acute"] == 1).sum() - idx["via_er"].sum())
    report["lace by readmission (mean)"] = {"readmitted": float(lace[y == 1].mean()), "not": float(lace[y == 0].mean())}
    (_out_dir(cfg) / "lace_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def print_lace(report: dict, cfg: ReadmitConfig) -> None:
    print(f"\n==== LACE baseline: {cfg.definition} definition ====")
    print(f"index stays {report['n']} | patients {report['patients']} | readmissions {report['readmissions']}")
    for k, v in report["overall"].items():
        print(f"  {k}: {_fmt(v)}")
    for k, v in report.items():
        if k.startswith("subgroup"):
            print(f"  {k}: n={v['n']} readmits={v['readmissions']} AUROC {_fmt(v['auroc'])}")
    for k in ["components (mean)", "lace by readmission (mean)", "acute from a separate ER stay the day before"]:
        print(f"  {k}: {report[k]}")


def model_report(res: PipelineResult, cfg: ReadmitConfig) -> dict:
    idx, X, y, groups = eligible_data(res)
    feature_sets = {"compact": COMPACT_FEATURES, "full": list(X.columns)}
    cv = cross_validate(X, y, groups, feature_sets, seed=0)
    oof = cv["oof"]
    out_dir = _out_dir(cfg)
    pd.DataFrame({"patient_key": groups, "readmit30": y, **oof}, index=idx.index).to_csv(
        out_dir / "oof_predictions.csv")  # row-level: local only
    models = [m for m in oof if m != "lace"]

    def stat(pick):
        yb, s = y[pick], {}
        s.update({f"lace_score {k}": v for k, v in metrics(yb, oof["lace"][pick], calibrated=False).items()})
        for m in models:
            s.update({f"{m} {k}": v for k, v in metrics(yb, oof[m][pick]).items()})
        for m in models[1:]:
            s[f"delta auroc {m} - lace_score"] = s[f"{m} auroc"] - s["lace_score auroc"]
        return s

    point = stat(np.arange(len(y)))
    report = {"n": len(y), "patients": int(len(set(groups))), "readmissions": int(y.sum()),
              "n_features": {k: len(v) for k, v in feature_sets.items()},
              "metrics": with_ci(point, cluster_bootstrap(groups, stat)), "chosen_params": cv["chosen_params"]}
    report["top 20% flagged"] = {m: top_fraction(y, oof[m]) for m in ["lace", *models[1:]]}
    repeats = {m: [] for m in models}
    for seed in REPEAT_SEEDS:
        rep = cross_validate(X, y, groups, feature_sets, seed=seed)["oof"]
        for m in repeats:
            repeats[m].append(metrics(y, rep[m], calibrated=False)["auroc"])
    report["auroc over 4 more CV repeats (mean, sd)"] = {
        m: (float(np.mean(v)), float(np.std(v))) for m, v in repeats.items()}
    _calibration_plot(y, oof, ["lace_prob", "logistic_compact", "xgboost_compact", "xgboost_full"],
                      out_dir / "calibration.png", cfg)
    report["shap"] = {fs: _shap(X[cols], y, groups, out_dir, fs) for fs, cols in feature_sets.items()}
    (out_dir / "model_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def print_models(report: dict, cfg: ReadmitConfig) -> None:
    print(f"\n==== Models: {cfg.definition} definition ====")
    print(f"index stays {report['n']} | patients {report['patients']} | readmissions {report['readmissions']}"
          f" | features {report['n_features']}")
    rows = []
    for m in ["lace_score", *[k for k in report["auroc over 4 more CV repeats (mean, sd)"]]]:
        r = {"model": LABELS.get(m, m)}
        for k in ["auroc", "auprc", "brier", "cal_intercept", "cal_slope"]:
            key = f"{m} {k}"
            r[k] = _fmt(report["metrics"][key]) if key in report["metrics"] else ""
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    for k, v in report["metrics"].items():
        if k.startswith("delta"):
            print(f"  {k}: {_fmt(v)}")
    print("  top 20% flagged (sensitivity, PPV):",
          {m: (round(float(d["sensitivity"]), 3), round(float(d["ppv"]), 3)) for m, d in report["top 20% flagged"].items()})
    print("  AUROC over 4 more CV repeats (mean, sd):",
          {m: (round(a, 3), round(b, 3)) for m, (a, b) in report["auroc over 4 more CV repeats (mean, sd)"].items()})
    print("  hyperparameters chosen per fold:", report["chosen_params"])
    for fs, top in report["shap"].items():
        print(f"  SHAP mean |value|, XGBoost {fs}, top 10:")
        for k, v in list(top.items())[:10]:
            print(f"    {k:40s} {v:.3f}")


LABELS = {"lace_score": "LACE score", "lace_prob": "LACE (logistic on score)",
          "logistic_compact": "Logistic, compact", "xgboost_compact": "XGBoost, compact",
          "logistic_full": "Logistic, full", "xgboost_full": "XGBoost, full"}


def _calibration_plot(y, oof, models: list[str], path: Path, cfg: ReadmitConfig, bins: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.6], [0, 0.6], ls="--", c="grey", lw=1)
    for m in models:
        label = LABELS[m]
        p = oof[m]
        q = pd.qcut(p, bins, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
        ax.plot(g["p"], g["y"], marker="o", label=label)
    ax.set(xlabel="Predicted 30-day readmission risk", ylabel="Observed rate",
           title=f"Calibration, out-of-fold ({cfg.definition}, quintiles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shap(X, y, groups, out_dir: Path, tag: str) -> dict:
    import shap

    model = fit_final_xgb(X, y, groups)
    values = shap.TreeExplainer(model).shap_values(X)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    for kind, name in [("bar", f"shap_bar_{tag}.png"), ("dot", f"shap_beeswarm_{tag}.png")]:
        shap.summary_plot(values, X, plot_type=kind, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=150)
        plt.close()
    return mean_abs.head(15).round(4).to_dict()
