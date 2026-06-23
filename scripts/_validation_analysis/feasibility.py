"""RQ1 feasibility floor (G1b): conditioned model vs embedding-ablated baselines.

Given greedy correctness for the real conditioned run and one or more embedding-ablation
runs (zero / mean / shuffle, produced by ``validate_model.py --embedding-ablation``), this
quantifies how far correctness falls when the conditioning signal is destroyed or corrupted.
The drop is the floor above which "the model conditions on the embedding" (RQ1's premise H0)
is a meaningful claim.

Two tables:
  * ``feasibility_floor_descriptive`` — per run: greedy semantic-equivalence rate, mean
    semantic distance, invalid rate, each with a bootstrap CI;
  * ``feasibility_floor_drop`` — per ablation: the PAIRED drop vs the conditioned run
    (conditioned − ablated) on the common targets, with a bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci

_METRICS = [("semantic_equiv_rate", "correct"),
            ("mean_semantic_distance", "semantic_distance"),
            ("invalid_rate", "is_invalid")]


def feasibility_floor_descriptive(
    corr: pd.DataFrame,
    *,
    runs: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per-run greedy correctness descriptives with bootstrap CIs.

    ``corr`` long-form with ``run``, ``formula_id``, ``correct`` (0/1), ``is_invalid`` (0/1),
    ``semantic_distance``.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        rdf = corr[corr["run"] == r]
        if rdf.empty:
            continue
        row = {"run": r, "n": int(len(rdf))}
        for name, col in _METRICS:
            if col not in rdf.columns:
                continue
            m, lo, hi = bootstrap_mean_ci(rdf[col].astype(float).dropna().to_numpy(),
                                          n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            row[name] = m
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def feasibility_floor_drop(
    corr: pd.DataFrame,
    *,
    conditioned_run: str,
    ablation_runs: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Paired drop (conditioned − ablation) on the common targets, with bootstrap CIs.

    A large positive ``semantic_equiv_rate`` drop = the model relies on the conditioning
    signal (the floor); a near-zero drop would mean the embedding is being ignored.
    """
    rng = np.random.default_rng(rng_seed)
    cond = corr[corr["run"] == conditioned_run]
    rows = []
    for abl in ablation_runs:
        a = corr[corr["run"] == abl]
        if cond.empty or a.empty:
            continue
        for name, col in _METRICS:
            if col not in corr.columns:
                continue
            merged = (cond[["formula_id", col]].rename(columns={col: "cond"})
                      .merge(a[["formula_id", col]].rename(columns={col: "abl"}), on="formula_id"))
            if merged.empty:
                continue
            diffs = (merged["cond"].astype(float) - merged["abl"].astype(float)).to_numpy()
            m, lo, hi = bootstrap_mean_ci(diffs, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            rows.append({"ablation": abl, "metric": name, "n_pairs": int(len(diffs)),
                         "conditioned_mean": float(merged["cond"].astype(float).mean()),
                         "ablated_mean": float(merged["abl"].astype(float).mean()),
                         "drop_conditioned_minus_ablated": m, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
