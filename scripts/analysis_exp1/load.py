"""Load and verify per-generation validation JSONLs (Experiment 1).

Reads the raw records emitted by ``validation_utils`` (``greedy.jsonl`` and
``topk_flat.jsonl``), asserts the schema, and runs completeness and
consistency checks before any statistic is computed. The checks are hard
failures by default: if a run directory contains duplicated or missing
targets (e.g. a DDP gather bug) or semantically inconsistent records, the
analysis refuses to run rather than silently producing wrong numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GREEDY_REQUIRED = {
    "formula_id", "target_formula_str", "target_depth",
    "generated_formula_str", "generated_depth",
    "is_invalid", "is_semantic_equivalent",
    "semantic_distance", "token_ids",
}

TOPK_REQUIRED = {
    "formula_id", "target_depth", "k_idx", "generated_formula_str",
    "generated_depth", "is_invalid",
    "is_semantic_equivalent", "semantic_distance", "token_ids",
}


def read_jsonl(path: Path, required: set[str]) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return pd.DataFrame(rows)


def consistency_checks(df: pd.DataFrame, *, name: str) -> list[dict]:
    """Semantic invariants that hold by construction of the scoring.

    Violations indicate a scoring bug (or, in ablation runs, exactly the kind
    of inconsistency the run is meant to surface) -- they are counted, reported,
    and treated as failures by the driver unless explicitly allowed.
    """
    checks = []

    def check(label: str, bad_mask: pd.Series) -> None:
        checks.append({"source": name, "check": label, "violations": int(bad_mask.sum())})

    check("invalid_implies_distance_1",
          df["is_invalid"] & (df["semantic_distance"] != 1.0))
    check("equivalent_iff_distance_0",
          df["is_semantic_equivalent"] != (df["semantic_distance"] == 0.0))
    check("invalid_has_null_depth",
          df["is_invalid"] & df["generated_depth"].notna())
    check("valid_has_depth",
          ~df["is_invalid"] & df["generated_depth"].isna())
    check("distance_in_unit_interval",
          (df["semantic_distance"] < 0.0) | (df["semantic_distance"] > 1.0))
    return checks


def load_greedy(run_dir: Path, *, expected_n: int | None) -> tuple[pd.DataFrame, list[dict]]:
    path = run_dir / "per_sample" / "greedy.jsonl"
    df = read_jsonl(path, GREEDY_REQUIRED)

    if df["formula_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated formula_id -- gather deduplication failed")
    if expected_n is not None and len(df) != expected_n:
        raise ValueError(f"{path}: {len(df)} rows, expected {expected_n} targets")

    return df, consistency_checks(df, name=f"{run_dir.name}/greedy")


def load_topk(run_dir: Path, *, expected_n: int | None, k: int) -> tuple[pd.DataFrame, list[dict]]:
    path = run_dir / "per_sample" / "topk_flat.jsonl"
    df = read_jsonl(path, TOPK_REQUIRED)

    if df.duplicated(subset=["formula_id", "k_idx"]).any():
        raise ValueError(f"{path}: duplicated (formula_id, k_idx) -- gather deduplication failed")
    counts = df.groupby("formula_id").size()
    if (counts != k).any():
        bad = counts[counts != k]
        raise ValueError(f"{path}: {len(bad)} targets without exactly {k} draws "
                         f"(e.g. {dict(bad.head(3))})")
    if expected_n is not None and counts.size != expected_n:
        raise ValueError(f"{path}: {counts.size} targets, expected {expected_n}")

    return df, consistency_checks(df, name=f"{run_dir.name}/topk_flat")


def read_dataset_size(dataset_dir: Path | None) -> int | None:
    """Target count from the validation dataset's metadata.json (tensors untouched)."""
    if dataset_dir is None:
        return None
    meta = json.loads((Path(dataset_dir) / "metadata.json").read_text())
    return int(meta["size"])


def infer_special_ids(token_rows: pd.Series) -> tuple[int, int]:
    """Infer (bos_id, eos_id) from stored trajectories, with hard consistency asserts.

    BOS is the unanimous first token of every stored sequence (``generate``
    always emits it). EOS is the unanimous last token over sequences that
    terminated; length-capped sequences (no EOS) are excluded by taking the
    modal last token and asserting it accounts for >95% of rows.
    """
    firsts = token_rows.str[0]
    if firsts.nunique() != 1:
        raise ValueError(f"BOS inference failed: first tokens not unanimous ({firsts.unique()[:5]})")
    bos = int(firsts.iloc[0])

    lasts = token_rows.str[-1]
    modal = lasts.mode()
    eos = int(modal.iloc[0])
    frac = float((lasts == eos).mean())
    if frac < 0.95:
        raise ValueError(f"EOS inference unreliable: modal last token {eos} covers only {frac:.1%}")
    if eos == bos:
        raise ValueError("EOS inference collided with BOS")
    return bos, eos
