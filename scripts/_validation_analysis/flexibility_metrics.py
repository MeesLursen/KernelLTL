"""RQ2 flexibility + graceful-degradation contrasts (implements I4 / I5 / I6).

These complement the main grid (``stats.py``) and ``extra_metrics`` / ``extra_contrast``
with the RQ2-specific measures worked out in the Experiment Design chapter
(``sec:rq2_aggregate``, ``sec:rq2_mechanism_syntax``):

* **I4 — correct-only equivalence-class SPREAD.** A size-UNBIASED *pairwise-average* of the
  normalised tree-edit distance (``formula_utils.tree_edit_distance``) among the correct
  generations of a target. We deliberately do **not** reuse the multi-reference Self-BLEU
  (``validation_utils._sentence_bleu``): once the sample is filtered to the correct subset its
  size becomes variable per target per model, and the multi-reference form is monotone
  non-decreasing in set size (more correct forms -> higher Self-BLEU -> reads as *less*
  diverse), a directional bias entangled with the count it should be orthogonal to. The
  pairwise-average is a U-statistic: its expectation is independent of the number of correct
  forms; size only affects variance. We aggregate TARGET-WEIGHTED (each target's per-target mean
  counts equally) with bootstrap-by-target. Tree-edit distance (not BLEU-4) is used because it has good dynamic
  range on short LTL ASTs and treats a commutative swap as a couple of relabels, not a maximal
  change.
* **I5 — distinct-correct as a PAIRED contrast.** The count (# distinct correct strings),
  in the layout ``extra_contrast.compute_paired_diff_summary`` consumes, so flexibility
  sits in the same dual-reference / effect-size frame as correctness.
* **I6 — wrong-but-valid set overlap (Jaccard) + a paired graceful-degradation contrast.**
  RL's wrong-but-valid errors are semantically closer to the target than CE's; the marginal
  ``cond_semdist_wrong_valid`` descriptive conditions on an outcome that differs by model,
  so we (a) report the Jaccard overlap of the ``{wrong & valid}`` sets and (b) run the paired
  ``semantic_distance | both-wrong-valid`` contrast on the COMMON error pool.

Usage (I5, I6 reuse the existing paired engine)::

    from scripts._validation_analysis.extra_contrast import compute_paired_diff_summary
    dc = distinct_correct_per_target(df_topk_flat, runs=runs)
    summary, _ = compute_paired_diff_summary(dc, reference_run="ce_base", variants=variants)

    wv = wrong_valid_distance_per_target(df_greedy, runs=runs)          # greedy
    gd_summary, _ = compute_paired_diff_summary(wv, reference_run="ce_base", variants=variants)
    overlap = wrong_valid_overlap(df_greedy, runs=runs)                 # Jaccard hygiene
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from formula_utils import ParseError, str_to_formula, tree_edit_distance
from scripts._validation_analysis.extra_contrast import compute_paired_diff_summary


# ===========================================================================
# I4 — correct-only equivalence-class spread (size-unbiased pairwise distance)
# ===========================================================================


def correct_only_spread_per_target(
    df_topk_flat: pd.DataFrame,
    *,
    min_correct: int = 2,
) -> pd.DataFrame:
    """Per (run, formula_id) with >= ``min_correct`` correct generations: the U-statistic
    mean pairwise normalised tree-edit distance among the correct (``is_semantic_equivalent``)
    generations.

    Repeats are KEPT (a model that emits one form K times has spread 0; identical strings parse
    to identical ASTs -> distance 0). Each correct string is parsed once. Also returns
    ``sum_dist`` and ``n_pairs`` so the pooled bootstrap can aggregate without recomputing.

    Columns: run, formula_id, target_depth, n_correct, n_pairs, sum_dist, value(=mean_dist).
    """
    correct = df_topk_flat[df_topk_flat["is_semantic_equivalent"].astype(bool)]
    rows: list[dict] = []
    for (run, fid), block in correct.groupby(["run", "formula_id"], sort=False):
        asts = []
        for s in block["generated_formula_str"].astype(str):
            try:
                asts.append(str_to_formula(s))
            except ParseError:
                pass  # correct => valid => parseable; skip defensively
        if len(asts) < min_correct:
            continue
        dists = [tree_edit_distance(a, b) for a, b in itertools.combinations(asts, 2)]
        rows.append({
            "run": run,
            "formula_id": int(fid),
            "target_depth": int(block["target_depth"].iloc[0]),
            "n_correct": len(asts),
            "n_pairs": len(dists),
            "sum_dist": float(np.sum(dists)),
            "value": float(np.mean(dists)),
        })
    return pd.DataFrame(rows)


def correct_only_spread_contrast(
    per_target: pd.DataFrame,
    *,
    runs: list[str],
    reference_run: str,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TARGET-WEIGHTED spread per run + paired (variant - reference) difference.

    Each target counts EQUALLY: its per-target mean tree-edit distance (``value``) is one
    observation. The per-run descriptive is the mean of ``value`` over that run's own
    >= ``min_correct`` targets; the paired difference is computed on the COMMON
    >= ``min_correct`` targets only, via the shared paired engine
    (:func:`extra_contrast.compute_paired_diff_summary`: bootstrap CI + Wilcoxon), so spread
    sits in the same frame as I5/I6. Bootstrap is by target throughout.

    Input ``per_target`` = output of :func:`correct_only_spread_per_target`.
    Returns (descriptive, paired): descriptive[run, n_targets, mean_spread, ci_low, ci_high];
    paired[variant, n_pairs, mean_diff, ci_low, ci_high, wilcoxon_p].
    """
    rng = np.random.default_rng(rng_seed)
    desc_rows: list[dict] = []
    for r in runs:
        vals = per_target.loc[per_target["run"] == r, "value"].to_numpy()
        if len(vals) == 0:
            continue
        idx = rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))
        boot = vals[idx].mean(axis=1)
        desc_rows.append({
            "run": r,
            "n_targets": int(len(vals)),
            "mean_spread": float(vals.mean()),
            "ci_low": float(np.quantile(boot, alpha / 2)),
            "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        })

    variants = [r for r in runs if r != reference_run]
    paired, _ = compute_paired_diff_summary(
        per_target[["run", "formula_id", "target_depth", "value"]],
        reference_run=reference_run,
        variants=variants,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng_seed=rng_seed,
    )
    return pd.DataFrame(desc_rows), paired


# ===========================================================================
# I5 — distinct-correct as a paired contrast
# ===========================================================================


def distinct_correct_per_target(
    df_topk_flat: pd.DataFrame,
    *,
    runs: list[str],
    conditional_on_any_correct: bool = True,
) -> pd.DataFrame:
    """Per (run, formula_id): # DISTINCT correct (``is_semantic_equivalent``) generated strings,
    in the (run, formula_id, target_depth, value) layout ``compute_paired_diff_summary`` expects.

    ``conditional_on_any_correct`` -> keep only targets with >= 1 correct (the contrast then reads
    "given the model can solve it, how many distinct ways"); else 0-correct targets get value 0.
    """
    correct = df_topk_flat[df_topk_flat["is_semantic_equivalent"].astype(bool)]
    distinct = (
        correct.groupby(["run", "formula_id", "target_depth"])["generated_formula_str"]
        .nunique()
        .reset_index(name="value")
    )
    if conditional_on_any_correct:
        out = distinct[distinct["run"].isin(runs)].copy()
    else:
        base = (
            df_topk_flat[df_topk_flat["run"].isin(runs)][["run", "formula_id", "target_depth"]]
            .drop_duplicates()
        )
        out = base.merge(distinct, on=["run", "formula_id", "target_depth"], how="left")
        out["value"] = out["value"].fillna(0.0)
    out["value"] = out["value"].astype(float)
    return out[["run", "formula_id", "target_depth", "value"]]


# ===========================================================================
# I6 — wrong-but-valid set overlap (Jaccard) + paired graceful-degradation
# ===========================================================================


def _wrong_valid_mask(df: pd.DataFrame) -> pd.Series:
    return (~df["is_semantic_equivalent"].astype(bool)) & (~df["is_invalid"].astype(bool))


def wrong_valid_overlap(df_greedy: pd.DataFrame, *, runs: list[str]) -> pd.DataFrame:
    """Pairwise Jaccard of each run pair's greedy ``{wrong & valid}`` formula_id sets.

    Returns every ordered pair (pivot to an NxN matrix), with set sizes for hygiene -- the
    Jaccard tells the reader how much of each model's error pool is shared before reading the
    paired ``semantic_distance | both-wrong-valid`` contrast.
    """
    wv = df_greedy[_wrong_valid_mask(df_greedy)]
    sets = {r: set(wv.loc[wv["run"] == r, "formula_id"].astype(int)) for r in runs}
    rows = []
    for a in runs:
        for b in runs:
            sa, sb = sets.get(a, set()), sets.get(b, set())
            inter, union = len(sa & sb), len(sa | sb)
            rows.append({
                "run_a": a, "run_b": b,
                "jaccard": (inter / union) if union else float("nan"),
                "n_a": len(sa), "n_b": len(sb), "n_inter": inter, "n_union": union,
            })
    return pd.DataFrame(rows)


def wrong_valid_distance_per_target(df_greedy: pd.DataFrame, *, runs: list[str]) -> pd.DataFrame:
    """Per (run, formula_id) greedy ``semantic_distance`` for the WRONG-AND-VALID targets only,
    in ``compute_paired_diff_summary``'s layout. That function's inner merge then restricts each
    paired contrast to targets BOTH models get wrong-and-valid (the common error pool), so the
    graceful-degradation comparison is paired-on-common rather than the marginal descriptive.
    """
    wv = df_greedy[_wrong_valid_mask(df_greedy) & df_greedy["run"].isin(runs)].copy()
    wv["value"] = wv["semantic_distance"].astype(float)
    return wv[["run", "formula_id", "target_depth", "value"]]
