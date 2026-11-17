"""Load and verify per-generation validation JSONLs (shared by all analyses).

Reads the raw records emitted by ``validation_utils`` (``greedy.jsonl`` and
``topk_flat.jsonl``), asserts the schema, and runs completeness and
consistency checks before any statistic is computed. The checks are hard
failures by default: if a run directory contains duplicated or missing
targets (e.g. a DDP gather bug), records that do not belong to the dataset
being read, or semantically inconsistent records, the analysis refuses to run
rather than silently producing wrong numbers.
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


def read_special_ids(tokenizer_dir: Path, token_rows: pd.Series) -> tuple[int, int]:
    """(bos_id, eos_id) from the saved tokenizer, checked against the stored trajectories.

    The ids come from the project's own tokenizer class, so the analysis uses
    exactly the mapping the model was trained and validated with. As a
    consistency check, every stored sequence must start with BOS (``generate``
    always emits it); EOS may be missing from length-capped sequences.
    """
    from tokenizer_pretrained_class import LTLTokenizer  # core module; imported only when needed

    tokenizer = LTLTokenizer.from_pretrained(str(tokenizer_dir))
    bos, eos = int(tokenizer.bos_token_id), int(tokenizer.eos_token_id)
    firsts = token_rows.str[0]
    if not (firsts == bos).all():
        bad = firsts[firsts != bos]
        raise ValueError(f"{len(bad)} stored sequences do not start with the tokenizer's BOS id {bos} "
                         f"(e.g. {bad.unique()[:5]}) -- wrong --tokenizer-dir?")
    return bos, eos


def check_alignment(df: pd.DataFrame, dataset_dir: Path, *, name: str) -> None:
    """Assert every record's target is the dataset row its ``formula_id`` points at.

    ``formula_id`` is a row index into the validation dataset, so a run scored
    against a different (e.g. rebuilt) dataset would silently misalign. Each
    record's ``target_formula_str`` must equal line ``formula_id`` of the
    dataset's ``formulas.jsonl`` (plain text; no tensors are loaded).
    """
    formulas = [line.rstrip("\n") for line in open(Path(dataset_dir) / "formulas.jsonl") if line.strip()]
    ids = df["formula_id"].to_numpy()
    if ids.min() < 0 or ids.max() >= len(formulas):
        raise ValueError(f"{name}: formula_id outside the {len(formulas)} dataset rows")
    expected = pd.Series([formulas[i] for i in ids], index=df.index)
    bad = df["target_formula_str"] != expected
    if bad.any():
        example = df.loc[bad, "formula_id"].iloc[0]
        raise ValueError(f"{name}: {int(bad.sum())} records do not match {dataset_dir}/formulas.jsonl "
                         f"(e.g. formula_id {example}) -- run scored against a different dataset?")
