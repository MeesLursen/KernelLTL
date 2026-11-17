"""Per-target metric derivations for Experiment 1.

Everything here is a pure function of the loaded records. The unit of
analysis is the target formula (thesis Sec. "Statistical conventions"): the
sampling pass's five draws are aggregated to one value per target before any
averaging or resampling, so that multiple observations of the same target are
never treated as independent evidence.

Conventions mirrored from the thesis text:
  - pass@k: unbiased estimator (Chen et al., 2021), n=5 draws, per target.
  - distinct-correct: number of distinct generated strings among a target's
    draws that are semantically equivalent to it; reported over all targets
    and over solved targets (>=1 correct draw).
  - self-BLEU: sentence-BLEU-4, uniform weights, each draw as candidate vs the
    other four as references, averaged; computed on the stored token ids with
    special tokens stripped; invalid generations are included (the score
    characterises the policy's output diversity as a whole).
  - generated depth: valid generations only (invalid rows carry null depth).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# pass@k
# --------------------------------------------------------------------------- #

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# --------------------------------------------------------------------------- #
# self-BLEU (faithful port of the deleted run-side implementation)
# --------------------------------------------------------------------------- #

def sentence_bleu(candidate: list[int], references: list[list[int]], max_n: int = 4) -> float:
    """Sentence-level BLEU with +1e-8 smoothing and closest-reference brevity penalty."""
    if not candidate or not references:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(candidate) < n:
            precisions.append(1e-8)
            continue
        cand_counts: dict[tuple, int] = defaultdict(int)
        for i in range(len(candidate) - n + 1):
            cand_counts[tuple(candidate[i:i + n])] += 1
        max_ref_counts: dict[tuple, int] = defaultdict(int)
        for ref in references:
            ref_counts: dict[tuple, int] = defaultdict(int)
            if len(ref) >= n:
                for i in range(len(ref) - n + 1):
                    ref_counts[tuple(ref[i:i + n])] += 1
            for ng, cnt in ref_counts.items():
                if cnt > max_ref_counts[ng]:
                    max_ref_counts[ng] = cnt
        clipped = sum(min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in cand_counts.items())
        total = sum(cand_counts.values())
        precisions.append((clipped + 1e-8) / (total + 1e-8))

    cand_len = len(candidate)
    closest = min((len(r) for r in references), key=lambda x: (abs(x - cand_len), x))
    bp = 1.0 if cand_len > closest else math.exp(1.0 - closest / max(cand_len, 1))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def self_bleu(token_seqs: list[list[int]]) -> float:
    if len(token_seqs) < 2:
        return float("nan")
    vals = [sentence_bleu(cand, [r for j, r in enumerate(token_seqs) if j != i])
            for i, cand in enumerate(token_seqs)]
    return float(np.mean(vals))


def strip_specials(ids: list[int], bos: int, eos: int) -> list[int]:
    out = ids[1:] if ids and ids[0] == bos else list(ids)
    if out and out[-1] == eos:
        out = out[:-1]
    return out


# --------------------------------------------------------------------------- #
# Per-target frames
# --------------------------------------------------------------------------- #

def per_target_greedy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target with the greedy pass's per-target values.

    ``wrong_valid_distance`` is the semantic distance where the generation
    parsed but was not equivalent, NaN elsewhere (conditional metrics stay
    NaN-coded so the bootstrap can condition within each resample).
    """
    out = pd.DataFrame({
        "formula_id": df["formula_id"].to_numpy(),
        "target_depth": df["target_depth"].to_numpy(),
        "equiv": df["is_semantic_equivalent"].astype(float).to_numpy(),
        "invalid": df["is_invalid"].astype(float).to_numpy(),
        "distance": df["semantic_distance"].astype(float).to_numpy(),
        "gen_depth": df["generated_depth"].astype(float).to_numpy(),  # NaN when invalid
    })
    wrong_valid = (~df["is_invalid"]) & (~df["is_semantic_equivalent"])
    out["wrong_valid_distance"] = np.where(
        wrong_valid, df["semantic_distance"].astype(float), np.nan)
    return out.set_index("formula_id").sort_index()


def per_target_topk(df: pd.DataFrame, *, k_max: int, bos: int, eos: int) -> pd.DataFrame:
    """One row per target aggregating the K sampled draws."""
    rows = []
    for fid, g in df.groupby("formula_id", sort=True):
        n = len(g)
        c = int(g["is_semantic_equivalent"].sum())
        gen_depths = g.loc[~g["is_invalid"], "generated_depth"].astype(float)
        correct_strs = g.loc[g["is_semantic_equivalent"], "generated_formula_str"]
        token_seqs = [strip_specials(ids, bos, eos) for ids in g["token_ids"]]
        row = {
            "formula_id": fid,
            "target_depth": int(g["target_depth"].iloc[0]),
            "equiv": float(g["is_semantic_equivalent"].mean()),   # per-sample rate = pass@1
            "invalid": float(g["is_invalid"].mean()),
            "distance": float(g["semantic_distance"].mean()),
            "gen_depth": float(gen_depths.mean()) if len(gen_depths) else np.nan,
            "distinct_correct": float(correct_strs.nunique()),
            "solved": float(c > 0),
            "self_bleu": self_bleu(token_seqs),
        }
        row["distinct_correct_solved"] = row["distinct_correct"] if c > 0 else np.nan
        for k in range(1, k_max + 1):
            row[f"pass_at_{k}"] = pass_at_k(n, c, k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("formula_id").sort_index()


# --------------------------------------------------------------------------- #
# Metric specifications: which columns feed which reported estimates
# --------------------------------------------------------------------------- #

# (metric name, per-target column). All estimators are (NaN-aware) means over
# targets; conditional metrics condition through their NaN coding.
GREEDY_METRICS = [
    ("semantic_equivalent_rate", "equiv"),
    ("semantic_distance", "distance"),
    ("invalid_rate", "invalid"),
    ("generated_depth_valid", "gen_depth"),
    ("wrong_valid_distance", "wrong_valid_distance"),
]

TOPK_METRICS = [
    ("semantic_equivalent_rate", "equiv"),
    ("semantic_distance", "distance"),
    ("invalid_rate", "invalid"),
    ("generated_depth_valid", "gen_depth"),
    ("distinct_correct_all", "distinct_correct"),
    ("distinct_correct_solved", "distinct_correct_solved"),
    ("solved_rate", "solved"),
    ("self_bleu", "self_bleu"),
]
