"""Conditional descriptive metrics, pass@k, and distinct-correct counts.

These complement the main validation-analysis grid by drilling into
properties of *subsets* of generations (correct, wrong-but-valid,
distinct-correct) rather than headline overall means.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    vals: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = vals[rng.integers(0, n, n)].mean()
    return (
        float(vals.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# ---------------------------------------------------------------------------
# Conditional difference / value statistics
# ---------------------------------------------------------------------------


def conditional_value_stats(
    df: pd.DataFrame,
    *,
    value_col: str,
    condition_fn: Callable[[pd.DataFrame], pd.Series],
    runs: list[str],
    by_depth: bool = False,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Mean of ``value_col`` conditional on ``condition_fn``, per run.

    Drops rows where ``value_col`` is NaN.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        sub = df[df["run"] == r]
        mask = condition_fn(sub)
        sub = sub[mask].copy()
        sub["_v"] = sub[value_col].astype(float)
        sub = sub[~sub["_v"].isna()]
        if by_depth:
            for d, dsub in sub.groupby("target_depth"):
                m, lo, hi = bootstrap_mean_ci(
                    dsub["_v"].to_numpy(), n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
                )
                rows.append({
                    "run": r, "target_depth": int(d), "n": int(len(dsub)),
                    "mean": m, "ci_low": lo, "ci_high": hi,
                })
        else:
            m, lo, hi = bootstrap_mean_ci(
                sub["_v"].to_numpy(), n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
            )
            rows.append({
                "run": r, "n": int(len(sub)),
                "mean": m, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


def conditional_diff_stats(
    df: pd.DataFrame,
    *,
    gen_col: str,
    tgt_col: str,
    condition_fn: Callable[[pd.DataFrame], pd.Series],
    runs: list[str],
    by_depth: bool = False,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Mean of ``gen_col - tgt_col`` conditional on ``condition_fn``, per run."""
    df = df.copy()
    df["_diff"] = df[gen_col].astype(float) - df[tgt_col].astype(float)
    return conditional_value_stats(
        df,
        value_col="_diff",
        condition_fn=condition_fn,
        runs=runs,
        by_depth=by_depth,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# Condition factories
# ---------------------------------------------------------------------------


def cond_correct(df: pd.DataFrame) -> pd.Series:
    return df["is_semantic_equivalent"].astype(bool)


def cond_wrong_and_valid(df: pd.DataFrame) -> pd.Series:
    return (~df["is_semantic_equivalent"].astype(bool)) & (~df["is_invalid"].astype(bool))


# ---------------------------------------------------------------------------
# pass@k' curve
# ---------------------------------------------------------------------------


def _pass_at_k_unbiased(K: int, c: int, k: int) -> float:
    """``1 - C(K-c, k) / C(K, k)`` — probability that drawing ``k`` of ``K``
    samples (``c`` of them correct) yields at least one correct."""
    if K - c < k:
        return 1.0
    if c == 0:
        return 0.0
    return 1.0 - math.comb(K - c, k) / math.comb(K, k)


def compute_pass_at_k_curve(
    df_flat: pd.DataFrame,
    *,
    runs: list[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per-run pass@k' for k' in [1..K], averaged over targets with bootstrap CIs."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        rdf = df_flat[df_flat["run"] == r]
        per_target = (
            rdf.groupby("formula_id")["is_semantic_equivalent"]
               .agg(K_count="count", c_count="sum")
               .reset_index()
        )
        per_target["K_count"] = per_target["K_count"].astype(int)
        per_target["c_count"] = per_target["c_count"].astype(int)
        if per_target.empty:
            continue
        K = int(per_target["K_count"].mode().iloc[0])
        per_target = per_target[per_target["K_count"] == K]
        K_arr = per_target["K_count"].to_numpy()
        c_arr = per_target["c_count"].to_numpy()

        for kp in range(1, K + 1):
            pak = np.array([_pass_at_k_unbiased(int(K_arr[i]), int(c_arr[i]), kp)
                            for i in range(len(per_target))])
            m, lo, hi = bootstrap_mean_ci(
                pak, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
            )
            rows.append({
                "run": r, "k_prime": kp, "K": K,
                "n_targets": int(len(per_target)),
                "mean": m, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distinct-correct counts
# ---------------------------------------------------------------------------


def compute_distinct_correct_stats(
    df_flat: pd.DataFrame,
    *,
    runs: list[str],
    conditional_on_any_correct: bool = False,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Avg distinct correct generations per target, per run.

    If ``conditional_on_any_correct`` is True, restricts to targets with >= 1
    semantically-equivalent sample.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        rdf = df_flat[df_flat["run"] == r]
        all_target_ids = rdf["formula_id"].unique()
        correct = rdf[rdf["is_semantic_equivalent"].astype(bool)]
        distinct = (
            correct.groupby("formula_id")["generated_formula_str"]
                   .nunique()
                   .to_dict()
        )
        vals = np.array([distinct.get(fid, 0) for fid in all_target_ids], dtype=float)
        if conditional_on_any_correct:
            vals = vals[vals > 0]
        m, lo, hi = bootstrap_mean_ci(
            vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
        )
        rows.append({
            "run": r,
            "n_targets": int(len(vals)),
            "mean": m, "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows)
