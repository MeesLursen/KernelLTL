"""Load per-run validation JSONLs into long-form pandas DataFrames.

Three frames per run, concatenated across runs with a ``run`` column:

* ``df_greedy``        — one row per (run, formula_id). Columns from
  ``greedy.jsonl`` plus precomputed ``seq_entropy_mean``,
  ``seq_kl_mean``, and ``seq_log_prob_mean``.
* ``df_topk_flat``     — one row per (run, formula_id, k_idx). Columns
  from ``topk_flat.jsonl`` plus derived ``is_exact_match``,
  ``is_semantic_equivalent``, ``semantic_distance``, ``generated_depth``,
  ``generated_length_tokens``, ``seq_entropy_mean``, ``seq_kl_mean``,
  ``seq_log_prob_mean``. The flat per-token arrays are dropped by default
  to keep memory in check.
* ``df_topk_grouped``  — one row per (run, formula_id). From
  ``topk_grouped.jsonl`` plus the K-aggregates we derive here:
  ``exact_match_rate_topk``, ``semantic_equiv_rate_topk``,
  ``syntax_semantics_gap_topk``, ``generated_depth_mean_topk``,
  ``generated_length_tokens_mean_topk``, ``semantic_distance_mean_topk``,
  ``semantic_distance_variance_topk``, ``mc_se_topk``.

Loader rewrites zero-filled ``ce_base`` KL fields to NaN; asserts
schemas; intersects ``formula_id`` sets across runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from formula_utils import ParseError, str_to_formula


# ---------------------------------------------------------------------------
# JSONL → DataFrame
# ---------------------------------------------------------------------------


_GREEDY_REQUIRED_FIELDS = {
    "formula_id", "target_formula_str", "target_depth", "target_length_tokens",
    "generated_formula_str", "generated_depth", "generated_length_tokens",
    "is_invalid", "is_exact_match", "is_semantic_equivalent",
    "semantic_distance", "token_ids", "token_entropies",
    "token_log_probs", "token_kls",
}

_TOPK_FLAT_REQUIRED_FIELDS = {
    "formula_id", "target_depth", "k_idx", "generated_formula_str",
    "is_invalid", "reward", "token_ids", "token_entropies",
    "token_log_probs", "token_kls",
}

_TOPK_GROUPED_REQUIRED_FIELDS = {
    "formula_id", "target_depth", "k", "n_invalid",
    "reward_mean", "reward_variance", "self_bleu",
    "policy_entropy_target_seq_mean", "policy_entropy_target_token_mean",
    "kl_from_base_target_seq_mean", "kl_from_base_target_token_mean",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _assert_fields(rows: list[dict], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path}: file is empty")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")


def _seq_mean(arr) -> float:
    if arr is None:
        return float("nan")
    arr = list(arr)
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def _count_non_pad(token_ids, pad_id: int) -> int:
    return int(sum(1 for t in token_ids if t != pad_id))


def _safe_parse_depth(s: str) -> float:
    try:
        return float(str_to_formula(s).depth())
    except (ParseError, ValueError, IndexError, KeyError):
        return float("nan")


# ---------------------------------------------------------------------------
# Per-run loaders
# ---------------------------------------------------------------------------


def load_greedy(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = run_dir / "per_sample" / "greedy.jsonl"
    rows = _read_jsonl(path)
    _assert_fields(rows, _GREEDY_REQUIRED_FIELDS, path)

    for r in rows:
        r["seq_entropy_mean"] = _seq_mean(r["token_entropies"])
        r["seq_kl_mean"] = _seq_mean(r["token_kls"])
        r["seq_log_prob_mean"] = _seq_mean(r["token_log_probs"])

    df = pd.DataFrame(rows)
    df.insert(0, "run", run_name)
    return df


def load_topk_flat(
    run_dir: Path,
    run_name: str,
    *,
    target_str_lookup: dict[int, str],
    pad_token_id: int,
    drop_token_arrays: bool = True,
) -> pd.DataFrame:
    path = run_dir / "per_sample" / "topk_flat.jsonl"
    rows = _read_jsonl(path)
    _assert_fields(rows, _TOPK_FLAT_REQUIRED_FIELDS, path)

    for r in rows:
        r["seq_entropy_mean"] = _seq_mean(r["token_entropies"])
        r["seq_kl_mean"] = _seq_mean(r["token_kls"])
        r["seq_log_prob_mean"] = _seq_mean(r["token_log_probs"])

        # Derived per-(target, k_idx) fields
        target_str = target_str_lookup.get(int(r["formula_id"]), None)
        gen_str = r["generated_formula_str"]
        r["target_formula_str"] = target_str
        r["is_exact_match"] = (gen_str == target_str) if target_str is not None else False
        r["is_semantic_equivalent"] = float(r["reward"]) == 1.0
        r["semantic_distance"] = 1.0 - float(r["reward"])
        if r["is_invalid"]:
            r["generated_depth"] = float("nan")
            r["generated_length_tokens"] = float("nan")
        else:
            r["generated_depth"] = _safe_parse_depth(gen_str)
            r["generated_length_tokens"] = float(_count_non_pad(r["token_ids"], pad_token_id))

        if drop_token_arrays:
            for k in ("token_entropies", "token_log_probs", "token_kls"):
                r.pop(k, None)

    df = pd.DataFrame(rows)
    df.insert(0, "run", run_name)
    return df


def load_topk_grouped(run_dir: Path, run_name: str) -> pd.DataFrame:
    path = run_dir / "per_sample" / "topk_grouped.jsonl"
    rows = _read_jsonl(path)
    _assert_fields(rows, _TOPK_GROUPED_REQUIRED_FIELDS, path)

    df = pd.DataFrame(rows)
    df.insert(0, "run", run_name)
    return df


def read_trivial_ids(path) -> set[int]:
    """Read formula_ids of trivial (tautology/contradiction = all-0/all-1, std==0)
    targets. Accepts geometry_features.csv (uses ``is_trivial``) or a plain
    trivial_ids.csv (single ``formula_id`` column)."""
    df = pd.read_csv(path)
    if "is_trivial" in df.columns:
        return set(df.loc[df["is_trivial"].astype(int) == 1, "formula_id"].astype(int))
    return set(df["formula_id"].astype(int))


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_runs(
    *,
    validation_root: Path,
    runs: Iterable[str],
    pad_token_id: int,
    drop_token_arrays: bool = True,
    exclude_ids: set[int] | None = None,
    dataset_dir: Path | str | None = None,
    log: callable = lambda m: print(m, file=sys.stderr),
) -> dict[str, pd.DataFrame]:
    """Load greedy/topk_flat/topk_grouped from every run, return a dict.

    Returns:
      {
        "df_greedy":        pd.DataFrame,
        "df_topk_flat":     pd.DataFrame,
        "df_topk_grouped":  pd.DataFrame,
        "df_topk_aggregates": pd.DataFrame,
        "validation_summary": dict[run -> dict],
      }
    """
    runs = list(runs)
    greedy_frames: list[pd.DataFrame] = []
    flat_frames: list[pd.DataFrame] = []
    grouped_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}

    # Auto-discover the trivial (tautology/contradiction = all-0/all-1, std==0) set:
    # it is a property of the dataset, so we read it from <dataset_dir>/trivial_ids.csv
    # unless an explicit exclude_ids was passed. These targets are dropped from ALL frames. 
    if exclude_ids is None and dataset_dir is not None:
        tpath = Path(dataset_dir) / "trivial_ids.csv"
        if tpath.exists():
            exclude_ids = read_trivial_ids(tpath)
            log(f"[loaders] auto-discovered {len(exclude_ids)} trivial ids from {tpath}")
        else:
            log(f"[loaders] WARNING: no trivial_ids.csv at {tpath}; trivial targets NOT filtered")

    for run_name in runs:
        run_dir = Path(validation_root) / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        log(f"[loaders] loading {run_name} from {run_dir}")

        df_g = load_greedy(run_dir, run_name)
        greedy_frames.append(df_g)

        target_str_lookup = dict(zip(df_g["formula_id"], df_g["target_formula_str"]))

        df_f = load_topk_flat(
            run_dir,
            run_name,
            target_str_lookup=target_str_lookup,
            pad_token_id=pad_token_id,
            drop_token_arrays=drop_token_arrays,
        )
        flat_frames.append(df_f)

        df_grp = load_topk_grouped(run_dir, run_name)
        grouped_frames.append(df_grp)

        summary_path = run_dir / "validation_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[run_name] = json.load(f)

    df_greedy = pd.concat(greedy_frames, ignore_index=True)
    df_topk_flat = pd.concat(flat_frames, ignore_index=True)
    df_topk_grouped = pd.concat(grouped_frames, ignore_index=True)

    # Drop trivial targets (tautologies/contradictions = all-0/all-1 satvec) from
    # ALL frames before any aggregation, so they never enter the results.
    if exclude_ids:
        exclude_ids = set(int(i) for i in exclude_ids)
        n0 = len(df_greedy)
        df_greedy = df_greedy[~df_greedy["formula_id"].astype(int).isin(exclude_ids)].reset_index(drop=True)
        df_topk_flat = df_topk_flat[~df_topk_flat["formula_id"].astype(int).isin(exclude_ids)].reset_index(drop=True)
        df_topk_grouped = df_topk_grouped[~df_topk_grouped["formula_id"].astype(int).isin(exclude_ids)].reset_index(drop=True)
        log(f"[loaders] excluded {len(exclude_ids)} trivial formula_ids "
            f"({n0 - len(df_greedy)} greedy rows dropped per run-stack)")

    # Build the per-target K-aggregates frame from df_topk_flat.
    df_topk_aggregates = build_topk_aggregates(df_topk_flat)

    # Sanity check: matching formula_id sets across runs.
    id_sets = {r: set(df_greedy.loc[df_greedy["run"] == r, "formula_id"].astype(int)) for r in runs}
    common = set.intersection(*id_sets.values())
    for r, ids in id_sets.items():
        missing = ids - common
        if missing:
            log(f"[loaders] WARNING: {r} has {len(missing)} formula_ids not present in all runs")

    log(
        f"[loaders] loaded {len(df_greedy)} greedy rows, "
        f"{len(df_topk_flat)} topk_flat rows, "
        f"{len(df_topk_grouped)} topk_grouped rows, "
        f"{len(common)} formula_ids common to all runs"
    )

    return {
        "df_greedy": df_greedy,
        "df_topk_flat": df_topk_flat,
        "df_topk_grouped": df_topk_grouped,
        "df_topk_aggregates": df_topk_aggregates,
        "validation_summary": summaries,
        "common_formula_ids": sorted(common),
    }


def build_topk_aggregates(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Per (run, formula_id) aggregates over the K samples."""

    def _aggregate_block(block: pd.DataFrame) -> pd.Series:
        K = len(block)
        invalid_mask = block["is_invalid"].astype(bool)
        valid_mask = ~invalid_mask
        n_valid = int(valid_mask.sum())
        sd_arr = block["semantic_distance"].astype(float).to_numpy()
        return pd.Series({
            "k": K,
            "n_invalid": int(invalid_mask.sum()),
            "invalid_rate_topk": float(invalid_mask.mean()),
            "exact_match_rate_topk": float(block["is_exact_match"].astype(bool).mean()),
            "semantic_equiv_rate_topk": float(block["is_semantic_equivalent"].astype(bool).mean()),
            "semantic_distance_mean_topk": float(np.mean(sd_arr)),
            "semantic_distance_variance_topk": float(np.var(sd_arr, ddof=0)),
            "mc_se_topk": float(np.sqrt(np.var(sd_arr, ddof=0) / K)),
            "generated_depth_mean_topk": (
                float(block.loc[valid_mask, "generated_depth"].astype(float).mean())
                if n_valid else float("nan")
            ),
            "generated_length_tokens_mean_topk": (
                float(block.loc[valid_mask, "generated_length_tokens"].astype(float).mean())
                if n_valid else float("nan")
            ),
        })

    grouped = df_flat.groupby(["run", "formula_id", "target_depth"], sort=False).apply(_aggregate_block, include_groups=False)
    grouped = grouped.reset_index()
    grouped["syntax_semantics_gap_topk"] = (
        grouped["semantic_equiv_rate_topk"] - grouped["exact_match_rate_topk"]
    )
    return grouped
