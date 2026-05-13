"""Target-side operator analysis: KL, decomposition, log-odds, logistic regression.

For each target formula we extract per-operator counts (and binary presence
indicators) by parsing the target via ``str_to_formula``. We then condition on
the model's correctness (``is_semantic_equivalent``) and compare the operator
distribution between the correct and wrong subsets.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from formula_class import (
    Atom, Not, And, Or, Implies, Next, Eventually, Globally, Until,
)
from formula_utils import ParseError, str_to_formula
from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci


OPERATORS: list[str] = ["NOT", "AND", "OR", "IMPLIES", "X", "F", "G", "U"]
COVARIATES: list[str] = ["target_depth", "target_length_tokens"]


# ---------------------------------------------------------------------------
# Operator extraction from formula strings
# ---------------------------------------------------------------------------


def extract_operator_counts(formula_str: str) -> dict[str, int]:
    """Parse ``formula_str`` and count each operator occurrence in the AST."""
    counts: dict[str, int] = {op: 0 for op in OPERATORS}
    try:
        f = str_to_formula(formula_str)
    except (ParseError, ValueError, IndexError, KeyError):
        return counts
    _count_recursive(f, counts)
    return counts


def _count_recursive(node, counts: dict[str, int]) -> None:
    if isinstance(node, Atom):
        return
    if isinstance(node, Not):
        counts["NOT"] += 1
        _count_recursive(node.child, counts)
    elif isinstance(node, And):
        counts["AND"] += 1
        _count_recursive(node.left, counts)
        _count_recursive(node.right, counts)
    elif isinstance(node, Or):
        counts["OR"] += 1
        _count_recursive(node.left, counts)
        _count_recursive(node.right, counts)
    elif isinstance(node, Implies):
        counts["IMPLIES"] += 1
        _count_recursive(node.left, counts)
        _count_recursive(node.right, counts)
    elif isinstance(node, Next):
        counts["X"] += 1
        _count_recursive(node.child, counts)
    elif isinstance(node, Eventually):
        counts["F"] += 1
        _count_recursive(node.child, counts)
    elif isinstance(node, Globally):
        counts["G"] += 1
        _count_recursive(node.child, counts)
    elif isinstance(node, Until):
        counts["U"] += 1
        _count_recursive(node.left, counts)
        _count_recursive(node.right, counts)


# ---------------------------------------------------------------------------
# Per-(run, formula_id) operator frame
# ---------------------------------------------------------------------------


def build_target_operator_frame(df_greedy: pd.DataFrame) -> pd.DataFrame:
    """Add per-operator count_<OP> and has_<OP> columns derived from
    ``target_formula_str``. Adds binary ``correct`` from ``is_semantic_equivalent``.

    Operator counts depend only on the target string, so they're the same
    across runs for a given ``formula_id``. We parse once per unique target.
    """
    cols = ["run", "formula_id", "target_formula_str", "target_depth",
            "target_length_tokens", "is_semantic_equivalent"]
    df = df_greedy[cols].copy()

    # Parse unique targets only.
    unique_targets = df["target_formula_str"].drop_duplicates()
    parsed = {s: extract_operator_counts(s) for s in unique_targets}

    for op in OPERATORS:
        df[f"count_{op}"] = df["target_formula_str"].map(lambda s: parsed[s][op])
        df[f"has_{op}"] = (df[f"count_{op}"] > 0).astype(int)

    df["correct"] = df["is_semantic_equivalent"].astype(int)
    return df


def build_target_operator_frame_topk(
    df_topk_flat: pd.DataFrame,
    df_greedy: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(run, formula_id) frame for top-K analyses: ``correct`` is 1 iff
    any of the K samples was semantically equivalent."""
    cols = ["run", "formula_id"]
    any_correct = (
        df_topk_flat.groupby(cols)["is_semantic_equivalent"]
                     .any().astype(int).reset_index(name="correct")
    )
    # Borrow target_formula_str + covariates from greedy (one row per (run, fid))
    targets = df_greedy[["run", "formula_id", "target_formula_str",
                         "target_depth", "target_length_tokens"]]
    merged = targets.merge(any_correct, on=cols, how="inner")

    unique_targets = merged["target_formula_str"].drop_duplicates()
    parsed = {s: extract_operator_counts(s) for s in unique_targets}

    for op in OPERATORS:
        merged[f"count_{op}"] = merged["target_formula_str"].map(lambda s: parsed[s][op])
        merged[f"has_{op}"] = (merged[f"count_{op}"] > 0).astype(int)
    return merged


# ---------------------------------------------------------------------------
# KL(P_op | correct ‖ P_op | wrong)
# ---------------------------------------------------------------------------


def compute_kl_per_run(df_op: pd.DataFrame, *, runs: list[str]) -> pd.DataFrame:
    """KL on per-token operator frequencies, between the correct and wrong subsets."""
    rows = []
    for r in runs:
        rdf = df_op[df_op["run"] == r]
        correct = rdf[rdf["correct"] == 1]
        wrong = rdf[rdf["correct"] == 0]
        if correct.empty or wrong.empty:
            continue
        c_counts = np.array([correct[f"count_{op}"].sum() for op in OPERATORS], float)
        w_counts = np.array([wrong[f"count_{op}"].sum() for op in OPERATORS], float)
        if c_counts.sum() == 0 or w_counts.sum() == 0:
            continue
        eps = 1e-10
        p_c = (c_counts + eps) / (c_counts + eps).sum()
        p_w = (w_counts + eps) / (w_counts + eps).sum()
        kl = float(np.sum(p_c * np.log(p_c / p_w)))
        per_op_contrib = (p_c * np.log(p_c / p_w)).tolist()
        rows.append({
            "run": r,
            "kl_correct_to_wrong": kl,
            "n_correct": int(len(correct)),
            "n_wrong": int(len(wrong)),
            **{f"contrib_{op}": per_op_contrib[i] for i, op in enumerate(OPERATORS)},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decomposition: P(op | correct), P(op | wrong), base rate
# ---------------------------------------------------------------------------


def compute_op_decomposition(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per-(run, operator) presence probabilities with bootstrap percentile CIs.

    Returns columns:
      run, op, p_op_given_correct, p_op_given_correct_ci_low/high,
      p_op_given_wrong, p_op_given_wrong_ci_low/high,
      p_op_base, n_correct, n_wrong.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        rdf = df_op[df_op["run"] == r]
        correct = rdf[rdf["correct"] == 1]
        wrong = rdf[rdf["correct"] == 0]
        for op in OPERATORS:
            c_vals = correct[f"has_{op}"].to_numpy(dtype=float)
            w_vals = wrong[f"has_{op}"].to_numpy(dtype=float)
            if len(c_vals):
                p_c, c_lo, c_hi = bootstrap_mean_ci(
                    c_vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
                )
            else:
                p_c, c_lo, c_hi = float("nan"), float("nan"), float("nan")
            if len(w_vals):
                p_w, w_lo, w_hi = bootstrap_mean_ci(
                    w_vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
                )
            else:
                p_w, w_lo, w_hi = float("nan"), float("nan"), float("nan")
            rows.append({
                "run": r, "op": op,
                "p_op_given_correct": p_c,
                "p_op_given_correct_ci_low": c_lo,
                "p_op_given_correct_ci_high": c_hi,
                "p_op_given_wrong": p_w,
                "p_op_given_wrong_ci_low": w_lo,
                "p_op_given_wrong_ci_high": w_hi,
                "p_op_base": float(rdf[f"has_{op}"].mean()) if len(rdf) else float("nan"),
                "n_correct": int(len(correct)),
                "n_wrong":   int(len(wrong)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Log-odds-ratios (marginal, per operator)
# ---------------------------------------------------------------------------


def _log_odds_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """log( (P(x=1|y=1)/P(x=0|y=1)) / (P(x=1|y=0)/P(x=0|y=0)) )

    Uses Haldane-Anscombe 0.5 continuity correction to avoid log(0).
    """
    a = float(((x == 1) & (y == 1)).sum()) + 0.5
    b = float(((x == 0) & (y == 1)).sum()) + 0.5
    c = float(((x == 1) & (y == 0)).sum()) + 0.5
    d = float(((x == 0) & (y == 0)).sum()) + 0.5
    return math.log((a * d) / (b * c))


def compute_log_odds_ratios(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per-(run, operator) marginal log-odds-ratio with bootstrap percentile CI."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in runs:
        rdf = df_op[df_op["run"] == r].reset_index(drop=True)
        n = len(rdf)
        if n == 0:
            continue
        correct = rdf["correct"].to_numpy()
        for op in OPERATORS:
            has = rdf[f"has_{op}"].to_numpy()
            est = _log_odds_ratio(has, correct)
            samples = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                samples[i] = _log_odds_ratio(has[idx], correct[idx])
            lo = float(np.quantile(samples, alpha / 2))
            hi = float(np.quantile(samples, 1 - alpha / 2))
            rows.append({
                "run": r, "op": op,
                "log_odds_ratio": est, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Logistic regression: correct ~ has_OP_1 + ... + has_OP_n + target_depth
# ---------------------------------------------------------------------------


def compute_logistic_regression(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    alpha: float = 0.05,
    use_regularized: bool = False,
) -> pd.DataFrame:
    """Per-run logistic regression with operator-presence indicators and
    target-side covariates (``target_depth`` and ``target_length_tokens``).

    Returns per-(run, predictor) coefficient (log-odds) with CI. If standard fit
    fails to converge, falls back to L2-regularized estimation (and CIs are
    unavailable in that case — CI fields will be NaN).
    """
    rows = []
    predictors = [f"has_{op}" for op in OPERATORS] + list(COVARIATES)
    for r in runs:
        rdf = df_op[df_op["run"] == r].copy()
        if rdf["correct"].nunique() < 2 or rdf.empty:
            continue
        X = rdf[predictors].astype(float)
        # Drop predictors with zero variance (constant columns destabilise Logit)
        keep = [c for c in predictors if X[c].nunique() > 1]
        X = sm.add_constant(X[keep], has_constant="add")
        y = rdf["correct"].astype(int)
        try:
            model = sm.Logit(y, X).fit(disp=False, maxiter=200)
            params = model.params
            conf = model.conf_int(alpha=alpha)
            pvals = model.pvalues
            for c in keep:
                rows.append({
                    "run": r, "predictor": c, "op": c.replace("has_", ""),
                    "coef": float(params[c]),
                    "ci_low": float(conf.loc[c, 0]),
                    "ci_high": float(conf.loc[c, 1]),
                    "p_value": float(pvals[c]),
                    "converged": True,
                })
        except Exception:
            if not use_regularized:
                continue
            try:
                model = sm.Logit(y, X).fit_regularized(disp=False, maxiter=200, alpha=1.0)
                params = model.params
                for c in keep:
                    rows.append({
                        "run": r, "predictor": c, "op": c.replace("has_", ""),
                        "coef": float(params[c]),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "p_value": float("nan"),
                        "converged": False,
                    })
            except Exception:
                continue
    return pd.DataFrame(rows)
