"""Load and check the Experiment 3 inputs.

Three sources, all produced elsewhere:

  * the conditioned greedy run of Experiment 1 (``<run-dir>/per_sample/greedy.jsonl``),
    loaded through ``analysis_exp1.load.load_greedy`` unchanged;
  * the per-target features of Experiment 2 (``<features-dir>/exp2_features.csv``),
    which carry ``p``, ``variance``, ``emb_norm`` and ``relational_faithfulness``;
  * the rescaling sweep of ``scripts/validation_variance_rescaling.py``
    (``<rescaling-dir>/scale_<c>/per_sample/greedy.jsonl`` for every ``c``, plus
    ``rescaling_manifest.json``).

Every loader is a hard gate: duplicated or missing targets, a scale with a
different target set than the others, or a manifest that disagrees with the
directories on disk all raise rather than produce a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "analysis_exp1"))
from load import GREEDY_REQUIRED, consistency_checks, read_jsonl        # noqa: E402

INTERVENTION_FIELDS = {
    "scale", "target_emb_norm", "input_emb_norm",
    "target_base_rate", "target_variance",
    "generated_base_rate", "generated_variance", "generated_emb_norm",
    "direction_cosine", "mean_token_entropy",
}
RESCALING_REQUIRED = GREEDY_REQUIRED | INTERVENTION_FIELDS

FEATURE_REQUIRED = {"formula_id", "p", "variance", "emb_norm", "relational_faithfulness"}


def load_features(features_dir: Path) -> pd.DataFrame:
    path = Path(features_dir) / "exp2_features.csv"
    df = pd.read_csv(path)
    missing = FEATURE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["formula_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated formula_id")
    if (df["variance"] <= 0).any():
        raise ValueError(f"{path}: non-positive variance rows present")
    return df.sort_values("formula_id").reset_index(drop=True)


def read_manifest(rescaling_dir: Path) -> dict:
    path = Path(rescaling_dir) / "rescaling_manifest.json"
    return json.loads(path.read_text())


def load_rescaling(rescaling_dir: Path, *, expected_n: int | None) -> tuple[pd.DataFrame, list[dict]]:
    """Concatenate every scale's greedy records into one long frame.

    Returns ``(df, checks)``: ``df`` has one row per (formula_id, scale) with the
    Experiment 1 greedy columns and the intervention fields; ``checks`` collects the
    Experiment 1 consistency checks per scale. Fails if any scale is missing a target,
    if the scales' target sets differ, or if the manifest lists scales not on disk.
    """
    root = Path(rescaling_dir)
    manifest = read_manifest(root)
    scales = [float(s) for s in manifest["scales"]]

    frames: list[pd.DataFrame] = []
    checks: list[dict] = []
    for scale in scales:
        run_dir = root / f"scale_{scale:g}"
        path = run_dir / "per_sample" / "greedy.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"manifest lists scale {scale:g} but {path} is missing")
        df = read_jsonl(path, RESCALING_REQUIRED)
        if df["formula_id"].duplicated().any():
            raise ValueError(f"{path}: duplicated formula_id -- gather deduplication failed")
        if expected_n is not None and len(df) != expected_n:
            raise ValueError(f"{path}: {len(df)} rows, expected {expected_n} targets")
        if (df["scale"] != scale).any():
            raise ValueError(f"{path}: records carry a scale other than the directory's {scale:g}")
        checks.extend(consistency_checks(df, name=f"scale_{scale:g}/greedy"))
        frames.append(df)

    ids = [set(f["formula_id"]) for f in frames]
    if any(s != ids[0] for s in ids[1:]):
        raise ValueError("rescaling: target sets differ across scales")

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["formula_id", "scale"]).reset_index(drop=True)
    # Invalid generations carry null intervention fields; make them NaN uniformly.
    for col in ("generated_base_rate", "generated_variance", "generated_emb_norm", "direction_cosine"):
        long[col] = pd.to_numeric(long[col], errors="coerce")
    return long, checks


def summarise_checks(checks: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(checks)
