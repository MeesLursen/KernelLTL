"""Geometry x operator bridge (G2) and RL-regression-set characterization (G4).

These two analyses sit at the seam between the two failure lenses of RQ2 (geometric:
embedding norm / variance / orthogonality; symbolic: which operators) and the
intervention question of RQ3 (where RL degraded relative to the base).

G2 — geometry x operator bridge (model-independent, per target).
    Tests whether the *under-served cell* the geometry analysis isolates (high-variance,
    low ``norm_resid`` = anchor-orthogonal) over-represents specific operators, i.e. whether
    the two RQ2 lenses point at the same targets. Built once from the target side:
      * ``operator_geometry_contrast`` — for each operator, mean geometry feature with vs
        without the operator (+ bootstrap CI on the difference);
      * ``operator_orthogonality_regression`` — OLS ``geom ~ has_OP_1..8 + depth`` (HC1),
        so each operator's *adjusted* association with orthogonality / informativeness is
        read off one fit (BH-FDR within each response family). A negative ``norm_resid``
        coefficient = the operator's targets are systematically more anchor-orthogonal.

G4 — RL-regression set (cross-run, vs the reference).
    Beyond the net McNemar (b - c), this *characterizes the targets a finetune broke*:
      * ``compute_correctness_flips`` — per (variant, target) flip category against the
        reference: both_correct / both_wrong / regression (ref right -> variant wrong) /
        recovery (ref wrong -> variant right);
      * ``profile_flip_geometry`` — geometry (variance / emb_norm / norm_resid / depth) of
        each flip set, so "where RL degraded" gets a geometric profile;
      * ``flip_operator_logodds`` — among the ref-correct targets, the log-odds that an
        operator's presence predicts the variant breaking it (regression vs both_correct).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci
from scripts._validation_analysis.operator_analysis import (
    OPERATORS, _log_odds_ratio, extract_operator_counts,
)

GEOM_COLS = ["variance", "emb_norm", "norm_resid"]
FLIP_CATEGORIES = ["both_correct", "regression", "recovery", "both_wrong"]


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd else 1.0)


def _fit_logit(formula: str, df: pd.DataFrame):
    """Logit with HC1 robust SEs; falls back to default cov if the robust fit fails."""
    try:
        return smf.logit(formula, df).fit(disp=False, maxiter=200, cov_type="HC1")
    except Exception:
        return smf.logit(formula, df).fit(disp=False, maxiter=200)


# ===========================================================================
# G2 — geometry x operator bridge
# ===========================================================================


def build_geometry_operator_frame(
    features: pd.DataFrame,
    target_meta: pd.DataFrame,
    *,
    n_var_bins: int = 50,
) -> pd.DataFrame:
    """Per-target frame joining geometry features to operator-presence indicators.

    ``features`` = geometry_features.csv (needs ``formula_id``, ``variance``, ``emb_norm``,
    ``is_trivial``). ``target_meta`` = one row per ``formula_id`` with ``target_formula_str``
    and ``target_depth`` (operators and geometry are both model-independent, so this frame is
    built once). Trivial (std==0) targets are dropped and ``norm_resid`` is the same
    Frisch-Waugh-Lovell orthogonality residual used by the correctness geometry analysis.
    """
    feat = features[features.get("is_trivial", 0) == 0].copy()
    feat["_vbin"] = pd.qcut(feat["variance"], min(n_var_bins, feat["variance"].nunique()),
                            labels=False, duplicates="drop")
    feat["norm_resid"] = feat["emb_norm"] - feat.groupby("_vbin")["emb_norm"].transform("mean")

    meta = target_meta.drop_duplicates("formula_id").copy()
    op_rows = []
    for fid, s in zip(meta["formula_id"].astype(int), meta["target_formula_str"].astype(str)):
        counts = extract_operator_counts(s)
        op_rows.append({"formula_id": fid, **{f"has_{op}": int(counts[op] > 0) for op in OPERATORS}})
    op_df = pd.DataFrame(op_rows)

    merged = (feat[["formula_id"] + GEOM_COLS]
              .merge(meta[["formula_id", "target_depth"]], on="formula_id", how="inner")
              .merge(op_df, on="formula_id", how="inner"))
    return merged


def operator_geometry_contrast(
    frame: pd.DataFrame,
    *,
    geom_cols: list[str] | None = None,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per (operator, geometry feature): mean with vs without the operator + bootstrap CI on
    the difference. Descriptive companion to the adjusted regression below."""
    geom_cols = geom_cols or GEOM_COLS
    rng = np.random.default_rng(rng_seed)
    rows = []
    for op in OPERATORS:
        col = f"has_{op}"
        if col not in frame.columns or frame[col].nunique() < 2:
            continue
        has = frame[frame[col] == 1]
        without = frame[frame[col] == 0]
        for g in geom_cols:
            vh = has[g].astype(float).dropna().to_numpy()
            vw = without[g].astype(float).dropna().to_numpy()
            mh, _, _ = bootstrap_mean_ci(vh, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            mw, _, _ = bootstrap_mean_ci(vw, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            # bootstrap CI on the (has - without) mean difference (independent resamples)
            if len(vh) and len(vw):
                diffs = np.empty(n_bootstrap)
                for i in range(n_bootstrap):
                    diffs[i] = vh[rng.integers(0, len(vh), len(vh))].mean() - \
                               vw[rng.integers(0, len(vw), len(vw))].mean()
                d_lo = float(np.quantile(diffs, alpha / 2))
                d_hi = float(np.quantile(diffs, 1 - alpha / 2))
            else:
                d_lo = d_hi = float("nan")
            rows.append({
                "op": op, "geom": g,
                "mean_has": mh, "mean_without": mw, "diff": mh - mw,
                "ci_low": d_lo, "ci_high": d_hi,
                "n_has": int(len(vh)), "n_without": int(len(vw)),
            })
    return pd.DataFrame(rows)


def operator_orthogonality_regression(
    frame: pd.DataFrame,
    *,
    responses: tuple[str, ...] = ("norm_resid", "variance"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """OLS ``z(geom) ~ has_OP_1..8 + z(target_depth)`` with HC1 SEs, one fit per response.

    The response is z-scored so coefficients are in response-SD units; ``target_depth`` is a
    z-scored covariate. Operators with no presence variance are dropped. Returns per
    (response, operator) coef + CI + p, with BH-FDR over the operator family within each
    response. For ``norm_resid`` a negative coefficient means the operator's targets are more
    anchor-orthogonal (smaller norm than their variance permits); for ``variance`` a positive
    coefficient means they are more informative.
    """
    df = frame.copy()
    df["z_target_depth"] = _z(df["target_depth"].astype(float))
    present_ops = [op for op in OPERATORS if f"has_{op}" in df.columns and df[f"has_{op}"].nunique() > 1]
    rows = []
    for resp in responses:
        if resp not in df.columns:
            continue
        df["_y"] = _z(df[resp].astype(float))
        formula = "_y ~ " + " + ".join([f"has_{op}" for op in present_ops] + ["z_target_depth"])
        try:
            res = smf.ols(formula, df).fit(cov_type="HC1")
        except Exception:
            continue
        sub = []
        for op in present_ops:
            term = f"has_{op}"
            if term not in res.params:
                continue
            ci = res.conf_int(alpha=alpha).loc[term]
            sub.append({"response": resp, "op": op, "coef": float(res.params[term]),
                        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                        "p_value": float(res.pvalues[term])})
        sdf = pd.DataFrame(sub)
        if not sdf.empty:
            sdf["p_value_adj_bh"] = multipletests(sdf["p_value"].fillna(1.0).to_numpy(),
                                                  alpha=alpha, method="fdr_bh")[1]
            sdf["reject_bh"] = sdf["p_value_adj_bh"] < alpha
        rows.append(sdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def operator_adjusted_correctness(
    df: pd.DataFrame,
    geom_op_frame: pd.DataFrame,
    *,
    runs: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Does the geometry -> correctness effect survive operator adjustment? (G2 bridge -> RQ2)

    Per run, fit two depth-adjusted logits and compare the ``z_variance`` / ``z_norm_resid``
    coefficients with vs. without the operator-presence main effects (additive, continuous
    geometry -- the form chosen so each effect stays a single interpretable slope):
        base:     correct ~ z_variance + z_norm_resid + C(target_depth)
        adjusted: correct ~ z_variance + z_norm_resid + C(target_depth) + has_OP_1..k
    If a geometry coefficient ATTENUATES toward 0 when operators enter, that failure axis is
    (partly) operator structure in disguise; if it is stable, the axis is structure-independent.
    The ``base`` fit reproduces :func:`geometry_analysis.q2_residual`; reading the SAME two
    coefficients before/after operators keeps the attenuation a clean nested comparison.
    Primary read is CE-base (the as-designed model). Robust HC1 SEs.

    ``df`` = :func:`geometry_analysis.build_frame` output (needs ``run``, ``formula_id``,
    ``correct``, ``z_variance``, ``z_norm_resid``, ``target_depth``). ``geom_op_frame`` =
    :func:`build_geometry_operator_frame` output (``formula_id`` + ``has_*`` indicators).
    Returns per (run, predictor): base/adjusted coef (+CIs, p), attenuation = 1 - adj/base,
    and the number of operator terms entered.
    """
    has_cols = [f"has_{op}" for op in OPERATORS if f"has_{op}" in geom_op_frame.columns]
    left = df.copy()
    right = geom_op_frame[["formula_id"] + has_cols].copy()
    left["formula_id"] = left["formula_id"].astype(int)   # align join dtype (built via 2 paths)
    right["formula_id"] = right["formula_id"].astype(int)
    merged = left.merge(right, on="formula_id", how="inner")
    preds = [("z_variance", "variance"), ("z_norm_resid", "norm_resid")]
    rows = []
    for r in runs:
        rdf = merged[merged["run"] == r]
        if rdf["correct"].nunique() < 2:
            continue
        present_ops = [c for c in has_cols if rdf[c].nunique() > 1]
        base = _fit_logit("correct ~ z_variance + z_norm_resid + C(target_depth)", rdf)
        adj_formula = ("correct ~ z_variance + z_norm_resid + C(target_depth)"
                       + "".join(f" + {c}" for c in present_ops))
        adj = _fit_logit(adj_formula, rdf)
        ci0_all, ci1_all = base.conf_int(alpha=alpha), adj.conf_int(alpha=alpha)
        for pred, label in preds:
            b0, b1 = float(base.params[pred]), float(adj.params[pred])
            ci0, ci1 = ci0_all.loc[pred], ci1_all.loc[pred]
            rows.append({
                "run": r, "predictor": label,
                "coef_base": b0, "base_ci_low": float(ci0[0]), "base_ci_high": float(ci0[1]),
                "p_base": float(base.pvalues[pred]),
                "coef_adjusted": b1, "adj_ci_low": float(ci1[0]), "adj_ci_high": float(ci1[1]),
                "p_adjusted": float(adj.pvalues[pred]),
                "attenuation": (1.0 - b1 / b0) if b0 != 0 else float("nan"),
                "n_operator_terms": len(present_ops),
            })
    return pd.DataFrame(rows)


def geomop_adjusted_coefficients(
    df: pd.DataFrame,
    geom_op_frame: pd.DataFrame,
    *,
    runs: list[str],
    outcome: str = "correct",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Full coefficient vector of the joint geometry+operator model, per run, for the unified
    RQ2 diagnostic / RQ3 cross-model forest. Sibling of :func:`operator_adjusted_correctness`:
    SAME RHS (``z_variance + z_norm_resid + C(target_depth) + has_OP_1..k``) but returns EVERY
    plottable coefficient (the two geometry terms + each operator), not just the attenuation
    pair. ``kind`` tags each row (geometry|operator) so the plotter can group/divide them.

    ONE outcome per call (the driver calls it once per outcome):
      outcome="correct"            -> logistic (coef = log-odds), HC1.
      outcome="semantic_distance"  -> OLS (coef = distance units), HC1. NOTE: the distance
        outcome is ~60% exact zeros (correct gens) and its variance-dependence is partly a
        Hamming-metric property, not a model effect -- a DESCRIPTIVE error-severity companion;
        read the operator / norm_resid rows, treat the variance row as caveated.

    Continuous geometry coefs are per +1 SD, operator coefs per presence -- comparable in sign
    and significance, NOT in magnitude across the two kinds. Primary read is CE-base.
    """
    has_cols = [f"has_{op}" for op in OPERATORS if f"has_{op}" in geom_op_frame.columns]
    left = df.copy()
    right = geom_op_frame[["formula_id"] + has_cols].copy()
    left["formula_id"] = left["formula_id"].astype(int)   # align join dtype (built via 2 paths)
    right["formula_id"] = right["formula_id"].astype(int)
    merged = left.merge(right, on="formula_id", how="inner")
    # (patsy term in the joint model, display label, kind); operators appended dynamically below
    geom_terms = [("z_variance", "variance", "geometry"),
                  ("z_norm_resid", "norm_resid", "geometry")]
    is_binary = outcome == "correct"
    rows = []
    for r in runs:
        rdf = merged[merged["run"] == r]
        if outcome not in rdf or rdf[outcome].nunique() < 2:
            continue
        present_ops = [c for c in has_cols if rdf[c].nunique() > 1]
        rhs = ("z_variance + z_norm_resid + C(target_depth)"
               + "".join(f" + {c}" for c in present_ops))
        formula = f"{outcome} ~ {rhs}"
        try:
            res = (smf.logit(formula, rdf).fit(disp=False, maxiter=200, cov_type="HC1")
                   if is_binary else smf.ols(formula, rdf).fit(cov_type="HC1"))
        except Exception:
            continue
        ci = res.conf_int(alpha=alpha)
        terms = list(geom_terms) + [(c, c.replace("has_", ""), "operator") for c in present_ops]
        for term, label, kind in terms:
            if term not in res.params:
                continue
            rows.append({"run": r, "outcome": outcome, "term": label, "kind": kind,
                         "coef": float(res.params[term]),
                         "ci_low": float(ci.loc[term][0]), "ci_high": float(ci.loc[term][1]),
                         "p_value": float(res.pvalues[term])})
    return pd.DataFrame(rows)


# ===========================================================================
# G4 — RL-regression set characterization
# ===========================================================================


def compute_correctness_flips(
    corr: pd.DataFrame,
    *,
    reference_run: str,
    variants: list[str],
) -> pd.DataFrame:
    """Per (variant, formula_id) flip category vs the reference (greedy correctness).

    ``corr`` is long-form with ``run``, ``formula_id``, ``correct`` (0/1), ``target_depth``.
    Category: both_correct / regression (ref right, variant wrong) / recovery (ref wrong,
    variant right) / both_wrong. Restricted to the common targets per (reference, variant).
    """
    ref = corr[corr["run"] == reference_run][["formula_id", "correct"]].rename(
        columns={"correct": "ref_correct"})
    out = []
    for v in variants:
        var = corr[corr["run"] == v][["formula_id", "correct", "target_depth"]].rename(
            columns={"correct": "var_correct"})
        m = ref.merge(var, on="formula_id", how="inner")
        if m.empty:
            continue
        rc = m["ref_correct"].astype(int).to_numpy()
        vc = m["var_correct"].astype(int).to_numpy()
        cat = np.where((rc == 1) & (vc == 1), "both_correct",
              np.where((rc == 1) & (vc == 0), "regression",
              np.where((rc == 0) & (vc == 1), "recovery", "both_wrong")))
        m = m.assign(variant=v, category=cat)
        out.append(m[["variant", "formula_id", "target_depth", "ref_correct", "var_correct", "category"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def flip_counts(flips: pd.DataFrame, *, variants: list[str]) -> pd.DataFrame:
    """Per-variant flip-category counts and the net flip balance (recovery - regression)."""
    rows = []
    for v in variants:
        sub = flips[flips["variant"] == v]
        if sub.empty:
            continue
        counts = sub["category"].value_counts().to_dict()
        n_reg = int(counts.get("regression", 0))
        n_rec = int(counts.get("recovery", 0))
        n = int(len(sub))
        rows.append({
            "variant": v, "n_common": n,
            **{c: int(counts.get(c, 0)) for c in FLIP_CATEGORIES},
            "net_flip": n_rec - n_reg,
            "regression_rate": n_reg / n if n else float("nan"),
            "recovery_rate": n_rec / n if n else float("nan"),
        })
    return pd.DataFrame(rows)


def profile_flip_geometry(
    flips: pd.DataFrame,
    geom_op_frame: pd.DataFrame,
    *,
    variants: list[str],
    geom_cols: list[str] | None = None,
    categories: tuple[str, ...] = ("both_correct", "regression", "recovery"),
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Per (variant, category, geometry feature): mean + bootstrap CI over that flip set.

    Joins the flip categories to the per-target geometry frame. ``both_wrong`` is omitted by
    default (it is the residual). Trivial targets are already absent from ``geom_op_frame``.
    """
    geom_cols = geom_cols or GEOM_COLS
    rng = np.random.default_rng(rng_seed)
    merged = flips.merge(geom_op_frame[["formula_id"] + geom_cols], on="formula_id", how="inner")
    rows = []
    for v in variants:
        for cat in categories:
            sub = merged[(merged["variant"] == v) & (merged["category"] == cat)]
            for g in geom_cols:
                vals = sub[g].astype(float).dropna().to_numpy()
                m, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
                rows.append({"variant": v, "category": cat, "geom": g,
                             "mean": m, "ci_low": lo, "ci_high": hi, "n": int(len(vals))})
    return pd.DataFrame(rows)


def flip_operator_logodds(
    flips: pd.DataFrame,
    geom_op_frame: pd.DataFrame,
    *,
    variants: list[str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Among the ref-correct targets, per (variant, operator) log-odds that the operator's
    presence predicts the variant breaking it (regression vs both_correct).

    Positive log-odds = the operator is over-represented in the targets the variant regressed
    on (relative to those it kept correct) -- the symbolic signature of where RL degraded.
    BH-FDR over the operator family within each variant.
    """
    rng = np.random.default_rng(rng_seed)
    has_cols = [f"has_{op}" for op in OPERATORS if f"has_{op}" in geom_op_frame.columns]
    merged = flips.merge(geom_op_frame[["formula_id"] + has_cols], on="formula_id", how="inner")
    rows = []
    for v in variants:
        sub = merged[(merged["variant"] == v) & (merged["category"].isin(["both_correct", "regression"]))]
        if sub.empty or sub["category"].nunique() < 2:
            continue
        y = (sub["category"] == "regression").astype(int).to_numpy()
        var_rows = []
        for op in OPERATORS:
            col = f"has_{op}"
            if col not in sub.columns or sub[col].nunique() < 2:
                continue
            x = sub[col].astype(int).to_numpy()
            est = _log_odds_ratio(x, y)
            samples = np.empty(n_bootstrap)
            n = len(y)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                samples[i] = _log_odds_ratio(x[idx], y[idx])
            var_rows.append({"variant": v, "op": op, "log_odds_ratio": est,
                             "ci_low": float(np.quantile(samples, alpha / 2)),
                             "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
                             "n_regression": int(y.sum()), "n_both_correct": int((y == 0).sum())})
        vdf = pd.DataFrame(var_rows)
        rows.append(vdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
