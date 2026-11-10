"""ingest -> normalize -> build stays -> label readmissions."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, ReadmitConfig
from .labels import build_episodes, label_readmissions
from .normalize import normalize_ens
from .stays import build_stays


@dataclass
class PipelineResult:
    stays: pd.DataFrame
    episodes: pd.DataFrame  # every admission episode (history for features)
    index: pd.DataFrame


def run(events: pd.DataFrame, cfg: ReadmitConfig) -> PipelineResult:
    stays = build_stays(events, cfg)
    eps = build_episodes(stays, cfg)
    return PipelineResult(stays, eps, label_readmissions(stays, cfg, eps))


def run_raw(raw: pd.DataFrame, cfg: ReadmitConfig) -> PipelineResult:
    return run(normalize_ens(raw), cfg)


def save(result: PipelineResult, cfg: ReadmitConfig, out_dir: Path = PROCESSED_DIR, suffix: str = "") -> list[Path]:
    """Write row-level outputs (hashed keys only, but still protected health information because of the dates: keep local)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"stays{suffix}.csv", out_dir / f"index_stays_{cfg.definition}{suffix}.csv"]
    result.stays.to_csv(paths[0])
    result.index.to_csv(paths[1])
    return paths
