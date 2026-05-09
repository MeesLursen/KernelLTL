"""Paired statistical tests for the validation analysis.

Per (metric, variant) pair vs the reference run we compute:

* **Wilcoxon signed-rank** (continuous + rate metrics)
* **Paired permutation test on mean diff** (always run alongside Wilcoxon
  for continuous metrics; ``tests_agree`` boolean flags disagreements at
  α = 0.05 of the raw p-values)
* **McNemar exact / asymptotic** (binary metrics)
* **Effect size**:
  - matched-pairs rank-biserial correlation r for continuous
  - Δp = p_var − p_ref for binary
* **Percentile bootstrap CI** on the paired mean difference (continuous)
  or Δp (binary), 10k resamples.

Plus BH-FDR multiplicity correction across the whole grid of cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def matched_pairs_rank_biserial(diffs: np.ndarray) -> float:
    """``r = (W+ − W−) / (W+ + W−)`` where W± are the signed-rank sums.

    Zero differences are dropped (consistent with Wilcoxon's
    ``zero_method='wilcox'``).
    """
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return float("nan")
    abs_ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
    w_plus = float(abs_ranks[nonzero > 0].sum())
    w_minus = float(abs_ranks[nonzero < 0].sum())
    total = w_plus + w_minus
    if total == 0:
        return float("nan")
    return (w_plus - w_minus) / total


def diff_proportions(ref: np.ndarray, var: np.ndarray) -> float:
    return float(var.mean() - ref.mean())


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    metric: str
    metric_class: str
    variant: str
    target_depth: int | None
    n_pairs: int
    wilcoxon_stat: float
    wilcoxon_p: float
    perm_p: float
    tests_agree: bool
    mcnemar_stat: float
    mcnemar_p: float
    p_used: float
    effect_size: float
    effect_size_name: str
    ci_low: float
    ci_high: float
    mean_ref: float
    mean_var: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _paired_permutation_p(
    diffs: np.ndarray, *, n_resamples: int, rng: np.random.Generator
) -> float:
    """Two-sided p for ``H_0: mean(diffs) == 0`` via sign permutations."""
    if len(diffs) == 0:
        return float("nan")
    obs = float(diffs.mean())
    n = len(diffs)
    abs_diffs = np.abs(diffs)
    # Vectorised: draw signs, multiply, mean.
    extreme = 0
    block = 1024
    remaining = n_resamples
    while remaining > 0:
        m = min(block, remaining)
        signs = rng.integers(0, 2, size=(m, n)) * 2 - 1
        means = (signs * abs_diffs).mean(axis=1)
        extreme += int(np.sum(np.abs(means) >= abs(obs) - 1e-15))
        remaining -= m
    return (extreme + 1) / (n_resamples + 1)


def _bootstrap_ci(
    diffs: np.ndarray, *, n_resamples: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    n = len(diffs)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = diffs[idx].mean()
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)


def _mcnemar_test(ref: np.ndarray, var: np.ndarray) -> tuple[float, float]:
    """Return (statistic, p-value) for paired binary vectors of {0,1}."""
    if len(ref) == 0:
        return (float("nan"), float("nan"))
    n11 = int(((ref == 1) & (var == 1)).sum())
    n10 = int(((ref == 1) & (var == 0)).sum())
    n01 = int(((ref == 0) & (var == 1)).sum())
    n00 = int(((ref == 0) & (var == 0)).sum())
    table = [[n11, n10], [n01, n00]]
    use_exact = (n10 + n01) <= 25
    res = mcnemar(table, exact=use_exact, correction=True)
    return (float(res.statistic) if res.statistic is not None else float("nan"),
            float(res.pvalue))


def run_continuous(
    metric: str,
    metric_class: str,
    variant: str,
    ref: np.ndarray,
    var: np.ndarray,
    *,
    target_depth: int | None,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> CellResult:
    diffs = var - ref
    n_pairs = int(np.sum(~np.isnan(diffs)))
    diffs = diffs[~np.isnan(diffs)]

    if n_pairs >= 1 and np.any(diffs != 0):
        try:
            w_res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_res.statistic)
            w_p = float(w_res.pvalue)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")

    perm_p = (
        _paired_permutation_p(diffs, n_resamples=n_resamples, rng=rng)
        if n_pairs >= 1 else float("nan")
    )
    tests_agree = (
        not np.isnan(w_p)
        and not np.isnan(perm_p)
        and ((w_p < alpha) == (perm_p < alpha))
    )

    eff = matched_pairs_rank_biserial(diffs) if n_pairs >= 1 else float("nan")

    if n_pairs >= 30:
        ci_low, ci_high = _bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, rng=rng)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return CellResult(
        metric=metric,
        metric_class=metric_class,
        variant=variant,
        target_depth=target_depth,
        n_pairs=n_pairs,
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p,
        perm_p=perm_p,
        tests_agree=tests_agree,
        mcnemar_stat=float("nan"),
        mcnemar_p=float("nan"),
        p_used=w_p,
        effect_size=eff,
        effect_size_name="rank_biserial_r",
        ci_low=ci_low,
        ci_high=ci_high,
        mean_ref=float(np.nanmean(ref)) if n_pairs else float("nan"),
        mean_var=float(np.nanmean(var)) if n_pairs else float("nan"),
    )


def run_binary(
    metric: str,
    metric_class: str,
    variant: str,
    ref: np.ndarray,
    var: np.ndarray,
    *,
    target_depth: int | None,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> CellResult:
    mask = ~(np.isnan(ref) | np.isnan(var))
    ref = ref[mask].astype(int)
    var = var[mask].astype(int)
    n_pairs = int(len(ref))

    mc_stat, mc_p = _mcnemar_test(ref, var) if n_pairs > 0 else (float("nan"), float("nan"))
    eff = diff_proportions(ref, var) if n_pairs > 0 else float("nan")

    if n_pairs >= 30:
        diffs = var.astype(float) - ref.astype(float)
        ci_low, ci_high = _bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, rng=rng)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return CellResult(
        metric=metric,
        metric_class=metric_class,
        variant=variant,
        target_depth=target_depth,
        n_pairs=n_pairs,
        wilcoxon_stat=float("nan"),
        wilcoxon_p=float("nan"),
        perm_p=float("nan"),
        tests_agree=True,
        mcnemar_stat=mc_stat,
        mcnemar_p=mc_p,
        p_used=mc_p,
        effect_size=eff,
        effect_size_name="diff_proportions",
        ci_low=ci_low,
        ci_high=ci_high,
        mean_ref=float(np.mean(ref)) if n_pairs else float("nan"),
        mean_var=float(np.mean(var)) if n_pairs else float("nan"),
    )


# ---------------------------------------------------------------------------
# Driver across the (metric, variant[, depth]) grid
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    name: str
    metric_class: str  # 'continuous_bounded' | 'continuous_unbounded' | 'binary' | 'rate'
    column: str        # column name in the source frame
    source: str        # 'greedy' | 'topk_flat' | 'topk_grouped' | 'topk_aggregates'


def _get_paired(
    df: pd.DataFrame,
    metric_col: str,
    ref_run: str,
    var_run: str,
) -> tuple[np.ndarray, np.ndarray]:
    ref_df = df[df["run"] == ref_run][["formula_id", metric_col]].rename(
        columns={metric_col: "ref"}
    )
    var_df = df[df["run"] == var_run][["formula_id", metric_col]].rename(
        columns={metric_col: "var"}
    )
    merged = ref_df.merge(var_df, on="formula_id", how="inner")
    return (
        merged["ref"].astype(float).to_numpy(),
        merged["var"].astype(float).to_numpy(),
    )


def run_pairwise_grid(
    *,
    sources: dict[str, pd.DataFrame],
    specs: list[MetricSpec],
    variants: Iterable[str],
    reference_run: str,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
    by_depth: bool = False,
) -> pd.DataFrame:
    out: list[CellResult] = []
    for spec in specs:
        df = sources[spec.source]
        depths_iter: list[int | None] = (
            sorted(df["target_depth"].dropna().unique().astype(int).tolist())
            if by_depth else [None]
        )
        for d in depths_iter:
            sub = df if d is None else df[df["target_depth"].astype(int) == int(d)]
            for variant in variants:
                if variant == reference_run:
                    continue
                if spec.column not in sub.columns:
                    continue
                ref, var = _get_paired(sub, spec.column, reference_run, variant)
                if spec.metric_class == "binary":
                    cell = run_binary(
                        spec.name, spec.metric_class, variant, ref, var,
                        target_depth=d, n_resamples=n_resamples, alpha=alpha, rng=rng,
                    )
                else:
                    cell = run_continuous(
                        spec.name, spec.metric_class, variant, ref, var,
                        target_depth=d, n_resamples=n_resamples, alpha=alpha, rng=rng,
                    )
                out.append(cell)

    df_out = pd.DataFrame([c.to_dict() for c in out])
    if df_out.empty:
        return df_out
    pvals = df_out["p_used"].fillna(1.0).to_numpy()
    df_out["p_adj_bh"] = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
    return df_out


# ---------------------------------------------------------------------------
# Diagnostics on paired differences (skewness, tie rate)
# ---------------------------------------------------------------------------


def paired_diff_diagnostics(
    sources: dict[str, pd.DataFrame],
    specs: list[MetricSpec],
    variants: Iterable[str],
    reference_run: str,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        df = sources[spec.source]
        for variant in variants:
            if variant == reference_run:
                continue
            if spec.column not in df.columns:
                continue
            ref, var = _get_paired(df, spec.column, reference_run, variant)
            diffs = var - ref
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) < 2:
                continue
            mean = diffs.mean()
            std = diffs.std(ddof=0) or 1e-12
            skew = float(np.mean(((diffs - mean) / std) ** 3))
            tie_rate = float(np.mean(diffs == 0))
            rows.append({
                "metric": spec.name,
                "metric_class": spec.metric_class,
                "variant": variant,
                "n": int(len(diffs)),
                "skewness": skew,
                "tie_rate": tie_rate,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def descriptives_by_run(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    extra_cols: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    extra_cols = extra_cols or {}
    for run, sub in df.groupby("run"):
        for col in columns:
            if col not in sub.columns:
                continue
            vals = sub[col].astype(float)
            n = int(vals.notna().sum())
            row = {
                "run": run,
                "metric": col,
                "n": n,
                "mean": float(vals.mean(skipna=True)),
                "median": float(vals.median(skipna=True)),
                "std": float(vals.std(skipna=True)),
            }
            for extra_name, extra_columns in extra_cols.items():
                if col in extra_columns and "mc_se_topk" in sub.columns:
                    row[extra_name] = float(sub["mc_se_topk"].mean(skipna=True))
            rows.append(row)
    return pd.DataFrame(rows)
