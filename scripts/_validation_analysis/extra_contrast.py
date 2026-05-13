"""Contrast / paired-comparison analyses across runs.

Three studies:

* **Conditional paired-diffs**: per-target paired differences between a variant
  and the reference run, for the conditional descriptive metrics
  (depth/length gap | correct or | wrong & valid; semantic distance | wrong &
  valid). Wilcoxon signed-rank p-values complement bootstrap-CI mean differences.
* **Per-target correctness agreement**: pairwise Cohen's κ + McNemar's test on
  matched-target correctness across runs.
* **Output similarity**: pairwise BLEU (sentence-level, BLEU-4 with epsilon
  smoothing) and exact-match rate between greedy generations of every run pair.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci


# ===========================================================================
# Per-target conditional aggregation + paired-diff
# ===========================================================================


def compute_per_target_conditional_value(
    df: pd.DataFrame,
    *,
    value_col: str | None = None,
    diff_cols: tuple[str, str] | None = None,
    condition_fn: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """Per (run, formula_id) mean of a metric over rows where condition_fn holds.

    If ``diff_cols=(gen, tgt)`` provided, computes ``gen − tgt``; else uses
    ``value_col`` directly.

    Drops rows where the metric is NaN before aggregation. Targets where no
    row meets ``condition_fn`` are simply absent from the output.
    """
    work = df.copy()
    if diff_cols is not None:
        gen_col, tgt_col = diff_cols
        work["_val"] = work[gen_col].astype(float) - work[tgt_col].astype(float)
    else:
        if value_col is None:
            raise ValueError("must supply value_col or diff_cols")
        work["_val"] = work[value_col].astype(float)
    mask = condition_fn(work)
    work = work[mask & work["_val"].notna()]
    if work.empty:
        return pd.DataFrame(columns=["run", "formula_id", "target_depth", "value", "n_samples"])
    agg = work.groupby(["run", "formula_id", "target_depth"])["_val"].agg(["mean", "count"]).reset_index()
    return agg.rename(columns={"mean": "value", "count": "n_samples"})


def compute_paired_diff_summary(
    per_target_df: pd.DataFrame,
    *,
    reference_run: str,
    variants: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary, per_target_diffs).

    ``summary`` columns: variant, n_pairs, mean_diff, ci_low, ci_high, wilcoxon_p.
    ``per_target_diffs`` columns: variant, formula_id, target_depth, ref, var, diff.
    """
    rng = np.random.default_rng(rng_seed)
    ref = per_target_df[per_target_df["run"] == reference_run][
        ["formula_id", "target_depth", "value"]
    ].rename(columns={"value": "ref"})
    summary_rows = []
    per_target_rows = []
    for v in variants:
        if v == reference_run:
            continue
        var = per_target_df[per_target_df["run"] == v][
            ["formula_id", "value"]
        ].rename(columns={"value": "var"})
        merged = ref.merge(var, on="formula_id", how="inner").dropna()
        if merged.empty:
            summary_rows.append({"variant": v, "n_pairs": 0, "mean_diff": float("nan"),
                                 "ci_low": float("nan"), "ci_high": float("nan"),
                                 "wilcoxon_p": float("nan")})
            continue
        diffs = (merged["var"].astype(float) - merged["ref"].astype(float)).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(diffs, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
        if np.any(diffs != 0):
            try:
                w_p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                w_p = float("nan")
        else:
            w_p = float("nan")
        summary_rows.append({
            "variant": v, "n_pairs": int(len(diffs)),
            "mean_diff": mean, "ci_low": lo, "ci_high": hi,
            "wilcoxon_p": w_p,
        })
        merged_out = merged.assign(variant=v, diff=diffs)
        per_target_rows.append(merged_out[["variant", "formula_id", "target_depth", "ref", "var", "diff"]])
    summary = pd.DataFrame(summary_rows)
    per_target = pd.concat(per_target_rows, ignore_index=True) if per_target_rows else pd.DataFrame()
    return summary, per_target


def compute_paired_diff_by_depth(
    per_target_df: pd.DataFrame,
    *,
    reference_run: str,
    variants: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Stratify paired diffs by target_depth."""
    rows = []
    rng = np.random.default_rng(rng_seed)
    depths = sorted(per_target_df["target_depth"].dropna().unique().astype(int).tolist())
    for d in depths:
        sub = per_target_df[per_target_df["target_depth"].astype(int) == d]
        if sub.empty:
            continue
        summary, _ = compute_paired_diff_summary(
            sub, reference_run=reference_run, variants=variants,
            n_bootstrap=n_bootstrap, alpha=alpha, rng_seed=int(rng.integers(0, 2**31 - 1)),
        )
        summary["target_depth"] = int(d)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ===========================================================================
# Pairwise correctness agreement (Cohen's κ + McNemar)
# ===========================================================================


def cohen_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's κ for two binary vectors over the same N items."""
    n = len(y1)
    if n == 0:
        return float("nan")
    p_obs = float(np.mean(y1 == y2))
    p1 = float(np.mean(y1 == 1))
    p2 = float(np.mean(y2 == 1))
    p_chance = p1 * p2 + (1 - p1) * (1 - p2)
    if p_chance >= 1.0:
        return 1.0
    return (p_obs - p_chance) / (1.0 - p_chance)


def _mcnemar_p_and_effect(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float]:
    """Return (p-value, signed effect (b−c)/N).

    Effect is ``+`` when y1 is more often correct than y2.
    """
    n = len(y1)
    if n == 0:
        return float("nan"), float("nan")
    a = int(((y1 == 1) & (y2 == 1)).sum())
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    d = int(((y1 == 0) & (y2 == 0)).sum())
    exact = (b + c) <= 25
    try:
        res = mcnemar([[a, b], [c, d]], exact=exact, correction=True)
        p = float(res.pvalue)
    except Exception:
        p = float("nan")
    effect = (b - c) / n
    return p, effect


def correctness_long_greedy(df_greedy: pd.DataFrame) -> pd.DataFrame:
    return df_greedy[["run", "formula_id"]].assign(
        correct=df_greedy["is_semantic_equivalent"].astype(int)
    )


def correctness_long_topk_any(df_topk_flat: pd.DataFrame) -> pd.DataFrame:
    return (
        df_topk_flat.groupby(["run", "formula_id"])["is_semantic_equivalent"]
                    .any().astype(int).reset_index(name="correct")
    )


def compute_pairwise_agreement(
    corr_df: pd.DataFrame,
    *,
    runs: list[str],
) -> pd.DataFrame:
    """Pairwise Cohen's κ + McNemar for binary correctness across runs.

    ``corr_df`` must be long-form with columns ``run``, ``formula_id``, ``correct``.

    Returns rows for *every ordered* pair (so an NxN matrix can be reconstructed
    by pivoting). McNemar effect is signed by the first run of each pair.
    """
    pivot = corr_df.pivot_table(index="formula_id", columns="run", values="correct")
    pivot = pivot.dropna(how="any")
    rows = []
    for a in runs:
        for b in runs:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            y1 = pivot[a].astype(int).to_numpy()
            y2 = pivot[b].astype(int).to_numpy()
            if a == b:
                rows.append({
                    "run_a": a, "run_b": b, "n_pairs": int(len(y1)),
                    "kappa": 1.0, "mcnemar_p": float("nan"), "mcnemar_effect": 0.0,
                    "acc_a": float(y1.mean()) if len(y1) else float("nan"),
                    "acc_b": float(y2.mean()) if len(y2) else float("nan"),
                })
                continue
            kappa = cohen_kappa(y1, y2)
            p, eff = _mcnemar_p_and_effect(y1, y2)
            rows.append({
                "run_a": a, "run_b": b, "n_pairs": int(len(y1)),
                "kappa": kappa, "mcnemar_p": p, "mcnemar_effect": eff,
                "acc_a": float(y1.mean()), "acc_b": float(y2.mean()),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Pairwise output similarity (BLEU + exact match) on greedy generations
# ===========================================================================


def _tokenize_formula(s: str) -> list[str]:
    """Split on whitespace after isolating parentheses as separate tokens."""
    if not isinstance(s, str):
        return []
    return s.replace("(", " ( ").replace(")", " ) ").split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_4(ref_tokens: list[str], cand_tokens: list[str], smooth: float = 1e-9) -> float:
    """Sentence-level BLEU-4 with epsilon smoothing on each precision."""
    if not cand_tokens:
        return 0.0
    weights = (0.25, 0.25, 0.25, 0.25)
    precisions: list[float] = []
    for n in range(1, 5):
        ref_ng = _ngrams(ref_tokens, n)
        cand_ng = _ngrams(cand_tokens, n)
        if sum(cand_ng.values()) == 0:
            precisions.append(smooth)
            continue
        matches = sum((cand_ng & ref_ng).values())
        total = sum(cand_ng.values())
        p = (matches + smooth) / (total + smooth)
        precisions.append(max(p, smooth))
    ref_len, cand_len = len(ref_tokens), len(cand_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    log_p = sum(w * math.log(p) for w, p in zip(weights, precisions))
    return float(bp * math.exp(log_p))


def compute_pairwise_output_similarity(
    df_greedy: pd.DataFrame,
    *,
    runs: list[str],
) -> pd.DataFrame:
    """Mean BLEU (sentence-BLEU-4, epsilon-smoothed) and exact-match rate
    between every ordered pair of runs, computed on shared formula_ids using
    each run's greedy generation."""
    pivot_gen = df_greedy.pivot_table(
        index="formula_id", columns="run",
        values="generated_formula_str", aggfunc="first",
    )
    pivot_gen = pivot_gen.dropna(how="any")

    # Pre-tokenize once per (formula_id, run) to avoid repeated work.
    tok_cache: dict[tuple[int, str], list[str]] = {}
    for run in runs:
        if run not in pivot_gen.columns:
            continue
        for fid, s in pivot_gen[run].items():
            tok_cache[(int(fid), run)] = _tokenize_formula(s)

    rows = []
    fids = pivot_gen.index.tolist()
    for a in runs:
        for b in runs:
            if a not in pivot_gen.columns or b not in pivot_gen.columns:
                continue
            if a == b:
                rows.append({"run_a": a, "run_b": b, "n_pairs": int(len(fids)),
                             "mean_bleu": 1.0, "exact_match_rate": 1.0})
                continue
            bleus = np.empty(len(fids))
            exacts = np.empty(len(fids))
            for i, fid in enumerate(fids):
                ta = tok_cache.get((int(fid), a), [])
                tb = tok_cache.get((int(fid), b), [])
                # Symmetric: average BLEU(a→b) + BLEU(b→a) / 2
                bleus[i] = 0.5 * (sentence_bleu_4(ta, tb) + sentence_bleu_4(tb, ta))
                exacts[i] = float(pivot_gen.loc[fid, a] == pivot_gen.loc[fid, b])
            rows.append({
                "run_a": a, "run_b": b, "n_pairs": int(len(fids)),
                "mean_bleu": float(bleus.mean()),
                "exact_match_rate": float(exacts.mean()),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# BH-FDR helper
# ===========================================================================


def apply_bh_fdr(
    p_values: Iterable[float],
    *,
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Apply Benjamini-Hochberg FDR to a family of p-values.

    NaN inputs are treated as 1.0 (no rejection) so the array length is
    preserved; the corresponding adjusted p is set back to NaN on return.
    """
    arr = np.array([1.0 if (p is None or np.isnan(p)) else float(p) for p in p_values])
    mask_nan = np.array([(p is None or np.isnan(p)) for p in p_values])
    if len(arr) == 0:
        return [], []
    reject, adj, _, _ = multipletests(arr, alpha=alpha, method="fdr_bh")
    adj = adj.astype(float)
    adj[mask_nan] = float("nan")
    reject = reject.astype(bool).tolist()
    # Force reject=False where p is NaN
    reject = [False if m else r for m, r in zip(mask_nan, reject)]
    return adj.tolist(), reject
