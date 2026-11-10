"""Command line. Prints aggregate output only.

  python -m readmit build   [--input PATH] [--definition broad|strict|both] [--legacy]
  python -m readmit compare [--input PATH]   # legacy vs each fix vs default, no files written
  python -m readmit profile [--input PATH]   # data quality checks behind the cleaning rules
  python -m readmit lace    [--definition broad|strict|both]   # LACE baseline, AUROC with bootstrap CIs
  python -m readmit model   [--definition broad|strict|both]   # LACE vs logistic vs XGBoost, calibration, SHAP
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import pipeline, profile
from .config import DEFAULT_INPUT, ReadmitConfig
from .ingest import load_ens
from .normalize import normalize_ens
from .report import data_quality, print_summary, summarize

FIXES = {
    "merge transfers": dict(merge_transfers=True),
    "apply discharge/transfer cancels": dict(apply_all_cancels=True),
    "outcome needs same definition": dict(outcome_same_definition=True),
    "readmission needs no recorded discharge": dict(outcome_requires_discharge=False),
    "blank class is not ER": dict(blank_class_is_er=False),
    "dx codes from discharge (+1 day)": dict(dx_from_last=True),
    "discharge date from Discharge event": dict(discharge_from_event=True),
    "events read in date order": dict(chronological_order=True),
    "hospice/expired disposition wins": dict(terminal_disposition_wins=True),
}


def _definitions(arg: str) -> list[str]:
    return ["broad", "strict"] if arg == "both" else [arg]


def cmd_build(args, raw, events):
    print_summary(data_quality(raw, events), "Data quality")
    for d in _definitions(args.definition):
        cfg = ReadmitConfig.legacy(definition=d) if args.legacy else ReadmitConfig(definition=d)
        res = pipeline.run(events, cfg)
        print_summary(summarize(res.stays, res.index, cfg), f"{d} definition{' (legacy)' if args.legacy else ''}")
        paths = pipeline.save(res, cfg, suffix="_legacy" if args.legacy else "")
        print("Saved:", ", ".join(p.name for p in paths), "in data/processed (row-level, keep local)")


def cmd_compare(args, raw, events):
    rows = []
    for d in ["broad", "strict"]:
        legacy = ReadmitConfig.legacy(definition=d)
        configs = {"legacy": legacy, **{f"+ {k}": replace(legacy, **v) for k, v in FIXES.items()},
                   "all fixes (default)": ReadmitConfig(definition=d),
                   "default, 0-day dx grace": ReadmitConfig(definition=d, dx_grace_days=0)}
        for name, cfg in configs.items():
            res = pipeline.run(events, cfg)
            s = summarize(res.stays, res.index, cfg)
            rows.append({"definition": d, "config": name, "index stays": s["eligible index stays"],
                         "patients": s["eligible patients"], "readmits": s["30-day readmissions"],
                         "rate": f"{s['readmission rate']:.1%}", "ER-only 30d": s["ER-only visit within 30 days"],
                         "merged": s["episodes merged from transfers"], "dx complete": f"{s['all dx codes complete']:.0%}"})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_lace(args, raw, events):
    from .experiment import lace_report, print_lace
    for d in _definitions(args.definition):
        cfg = ReadmitConfig(definition=d)
        print_lace(lace_report(pipeline.run(events, cfg), cfg), cfg)


def cmd_model(args, raw, events):
    from .experiment import model_report, print_models
    for d in _definitions(args.definition):
        cfg = ReadmitConfig(definition=d)
        print_models(model_report(pipeline.run(events, cfg), cfg), cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog="readmit")
    p.add_argument("command", choices=["build", "compare", "profile", "lace", "model"])
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--definition", choices=["broad", "strict", "both"], default="both")
    p.add_argument("--legacy", action="store_true", help="reproduce the earlier stay-building rules")
    args = p.parse_args(argv)
    if not Path(args.input).exists():
        raise SystemExit(f"Input not found: {args.input}. Put the payer file in data/raw/ or pass --input.")
    raw = load_ens(args.input)
    events = normalize_ens(raw)
    commands = {"build": cmd_build, "compare": cmd_compare, "profile": lambda a, r, e: profile.run(r, e),
                "lace": cmd_lace, "model": cmd_model}
    commands[args.command](args, raw, events)


if __name__ == "__main__":
    main()
