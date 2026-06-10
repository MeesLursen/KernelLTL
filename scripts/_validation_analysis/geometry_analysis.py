"""Embedding-geometry vs. correctness analysis (norm / variance / orthogonality).

Tests the kernel/architecture claim chain (Ch4 cross-attention bound
||V_sem|| <= ||W^V|| ||emb(phi)||; Ch5 two causes of small magnitude: low variance
vs. anchor orthogonality), on the NON-trivial validation targets (tautologies /
contradictions, std==0, are dropped — they share the zero embedding by construction).

Outcome: binary ``correct`` (= is_semantic_equivalent). Continuous ``semantic_distance``
is only used for flagged descriptive curves (its variance-dependence is a Hamming-metric
property, not a model effect — see the trivial/⊤-⊥ discussion).

Predictors (z-scored on the non-trivial set):
  emb_norm    conditioning magnitude
  variance    informativeness (= p(1-p))
  norm_resid  orthogonality = emb_norm - E[emb_norm | variance]  (flexible binned control;
              decorrelated from variance, so it isolates anchor-coverage)
  target_depth covariate (categorical)

Studies (mirror the operator analyses + the same BH-FDR family on the cross-model test):
  Q1  marginal magnitude effect            : correct ~ emb_norm
  Q2  primary (stratified)                 : norm slope within variance strata
  Q2  summary (FWL residual)               : correct ~ variance + norm_resid + C(depth)
  bonus cross-model residual interaction   : model x (variance + norm_resid), cluster-robust,
                                             BH-FDR over the interaction family, + AME.
Robust (HC1 / cluster) SEs throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm as _normdist
from statsmodels.stats.multitest import multipletests

from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci

GEOMETRY_COLS = ["emb_norm", "variance", "norm_resid"]


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd else 1.0)


def build_frame(features: pd.DataFrame, correctness: pd.DataFrame, *, n_var_bins: int = 50) -> pd.DataFrame:
    """Join geometry features to per-(run, formula_id) correctness, DROP trivial
    (std==0) targets, compute the orthogonality residual and z-scored predictors.

    ``correctness`` columns: run, formula_id, correct, semantic_distance, target_depth.
    norm_resid = emb_norm - E[emb_norm | variance] (binned, on the non-trivial set).
    """
    feat = features[features.get("is_trivial", 0) == 0].copy()
    # orthogonality residual on the non-trivial set
    feat["_vbin"] = pd.qcut(feat["variance"], min(n_var_bins, feat["variance"].nunique()),
                            labels=False, duplicates="drop")
    feat["norm_resid"] = feat["emb_norm"] - feat.groupby("_vbin")["emb_norm"].transform("mean")
    # z-scored predictors on the (run-invariant) target distribution
    for c in GEOMETRY_COLS:
        feat[f"z_{c}"] = _z(feat[c])

    keep = ["formula_id"] + GEOMETRY_COLS + [f"z_{c}" for c in GEOMETRY_COLS]
    merged = correctness.merge(feat[keep], on="formula_id", how="inner")
    merged["z_target_depth"] = _z(merged["target_depth"].astype(float))
    return merged


def _fit_logit(formula: str, df: pd.DataFrame):
    """Logit with HC1 robust SEs; falls back to default cov if robust fit fails."""
    try:
        return smf.logit(formula, df).fit(disp=False, maxiter=200, cov_type="HC1")
    except Exception:
        return smf.logit(formula, df).fit(disp=False, maxiter=200)


# ---------------------------------------------------------------------------
# Q1 — marginal magnitude effect
# ---------------------------------------------------------------------------


def q1_marginal(df: pd.DataFrame, *, runs: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """correct ~ z_emb_norm, per run (the basic 'does magnitude predict quality')."""
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        if rdf["correct"].nunique() < 2:
            continue
        res = _fit_logit("correct ~ z_emb_norm", rdf)
        ci = res.conf_int(alpha=alpha).loc["z_emb_norm"]
        rows.append({"run": r, "coef": float(res.params["z_emb_norm"]),
                     "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                     "p_value": float(res.pvalues["z_emb_norm"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q2 — FWL residual (summary) and variance-stratified slopes (primary)
# ---------------------------------------------------------------------------


def q2_residual(df: pd.DataFrame, *, runs: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """correct ~ z_variance + z_norm_resid + C(target_depth), per run.

    z_variance coef = informativeness effect; z_norm_resid coef = orthogonality
    effect (norm beyond variance), decorrelated. Depth-adjusted, robust SEs."""
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        if rdf["correct"].nunique() < 2:
            continue
        res = _fit_logit("correct ~ z_variance + z_norm_resid + C(target_depth)", rdf)
        for pred, label in [("z_variance", "variance"), ("z_norm_resid", "orthogonality")]:
            ci = res.conf_int(alpha=alpha).loc[pred]
            rows.append({"run": r, "predictor": label, "coef": float(res.params[pred]),
                         "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                         "p_value": float(res.pvalues[pred])})
    return pd.DataFrame(rows)


def variance_stratified_slopes(df: pd.DataFrame, *, runs: list[str], n_strata: int = 3,
                               alpha: float = 0.05) -> pd.DataFrame:
    """Within variance strata, fit correct ~ z_emb_norm + C(target_depth) and report the
    norm slope. The orthogonality claim = a positive norm slope persists at HIGH variance."""
    labels = (["low", "mid", "high"] if n_strata == 3 else list(range(n_strata)))
    df = df.copy()
    df["_vstr"] = pd.qcut(df["variance"], n_strata, labels=labels, duplicates="drop")
    rows = []
    for r in runs:
        for s in labels:
            sub = df[(df["run"] == r) & (df["_vstr"] == s)]
            if len(sub) < 50 or sub["correct"].nunique() < 2:
                continue
            res = _fit_logit("correct ~ z_emb_norm + C(target_depth)", sub)
            if "z_emb_norm" not in res.params:
                continue
            ci = res.conf_int(alpha=alpha).loc["z_emb_norm"]
            rows.append({"run": r, "stratum": s, "n": int(len(sub)),
                         "norm_coef": float(res.params["z_emb_norm"]),
                         "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                         "p_value": float(res.pvalues["z_emb_norm"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive: marginal binned curves + 2-D (variance x norm_resid) grid
# ---------------------------------------------------------------------------


def marginal_binned(df: pd.DataFrame, *, runs: list[str], feature: str, outcome: str = "correct",
                    n_bins: int = 12, n_bootstrap: int = 2000, alpha: float = 0.05,
                    rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    edges = np.unique(np.quantile(df[feature].to_numpy(), np.linspace(0, 1, n_bins + 1)))
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        idx = np.clip(np.digitize(rdf[feature].to_numpy(), edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            vals = rdf.loc[idx == b, outcome].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            m, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            rows.append({"run": r, "feature": feature, "outcome": outcome, "bin": b,
                         "x_mid": float(0.5 * (edges[b] + edges[b + 1])),
                         "mean": m, "ci_low": lo, "ci_high": hi, "n": int(len(vals))})
    return pd.DataFrame(rows)


def two_d_grid(df: pd.DataFrame, *, runs: list[str], fx: str = "variance", fy: str = "norm_resid",
               nbins: int = 8, outcome: str = "correct") -> pd.DataFrame:
    """fx x fy quantile grid of mean outcome. With fx=variance, fy=norm_resid the axes
    are decorrelated, so the grid fills (unlike variance x raw norm)."""
    ex = np.unique(np.quantile(df[fx].to_numpy(), np.linspace(0, 1, nbins + 1)))
    ey = np.unique(np.quantile(df[fy].to_numpy(), np.linspace(0, 1, nbins + 1)))
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        ix = np.clip(np.digitize(rdf[fx].to_numpy(), ex[1:-1]), 0, len(ex) - 2)
        iy = np.clip(np.digitize(rdf[fy].to_numpy(), ey[1:-1]), 0, len(ey) - 2)
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                vals = rdf.loc[(ix == a) & (iy == b), outcome].astype(float).to_numpy()
                rows.append({"run": r, "ix": a, "iy": b,
                             "x_mid": float(0.5 * (ex[a] + ex[a + 1])),
                             "y_mid": float(0.5 * (ey[b] + ey[b + 1])),
                             "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                             "n": int(len(vals))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bonus: cross-model residual interaction (BH-FDR family) + AME
#   correct ~ model x (z_variance + z_norm_resid) + z_target_depth (covariate, no interaction)
# ---------------------------------------------------------------------------


def _restrict_common(df: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    sub = df[df["run"].isin(runs)]
    id_sets = [set(sub.loc[sub["run"] == r, "formula_id"]) for r in runs]
    common = set.intersection(*id_sets) if id_sets else set()
    return sub[sub["formula_id"].isin(common)].copy()


def cross_model_interaction(df: pd.DataFrame, *, runs: list[str], reference_run: str,
                            alpha: float = 0.05, n_sim: int = 2000, rng_seed: int = 0) -> dict:
    """Pooled `correct ~ model x (z_variance + z_norm_resid) + z_target_depth`,
    cluster-robust SE by formula_id. Interaction coef = per-+1-SD slope difference vs the
    reference (does finetuning change geometry-reliance?). BH-FDR over the interaction family."""
    interact = ["z_variance", "z_norm_resid"]
    covar = ["z_target_depth"]
    variants = [r for r in runs if r != reference_run]
    stacked = _restrict_common(df, runs).dropna(subset=["correct"] + interact + covar)
    empty = {"interactions": pd.DataFrame(), "ame": pd.DataFrame(),
             "n_obs": int(len(stacked)), "n_targets": 0}
    if stacked.empty or stacked["correct"].nunique() < 2:
        return empty

    X = pd.DataFrame(index=stacked.index)
    X["const"] = 1.0
    for v in variants:
        X[f"m::{v}"] = (stacked["run"] == v).astype(float)
    for p in interact + covar:
        X[p] = stacked[p].astype(float)
    for v in variants:
        mv = (stacked["run"] == v).astype(float).to_numpy()
        for p in interact:                      # only geometry gets model-interacted
            X[f"m::{v}:{p}"] = mv * stacked[p].astype(float).to_numpy()
    y = stacked["correct"].astype(int).to_numpy()
    groups = stacked["formula_id"].to_numpy()
    try:
        res = sm.Logit(y, X.to_numpy()).fit(disp=False, maxiter=300,
                                            cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        empty["n_targets"] = int(stacked["formula_id"].nunique())
        return empty

    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    params = pd.Series(res.params, index=cols)
    bse = pd.Series(res.bse, index=cols)
    pvals = pd.Series(res.pvalues, index=cols)
    zc = abs(float(_normdist.ppf(alpha / 2)))

    inter_rows = []
    for v in variants:
        for p in interact:
            name = f"m::{v}:{p}"
            b, s, pv = float(params[name]), float(bse[name]), float(pvals[name])
            inter_rows.append({"variant": v, "predictor": p.replace("z_", ""), "coef": b, "se": s,
                               "ci_low": b - zc * s, "ci_high": b + zc * s, "p_value": pv})
    interactions = pd.DataFrame(inter_rows)
    if not interactions.empty:
        interactions["p_value_adj_bh"] = multipletests(
            interactions["p_value"].fillna(1.0).to_numpy(), alpha=alpha, method="fdr_bh")[1]
        interactions["reject_bh"] = interactions["p_value_adj_bh"] < alpha

    # AME (probability scale): effect of +1 SD of each geometry predictor, per model, vs ref.
    cov = np.asarray(res.cov_params())
    rng = np.random.default_rng(rng_seed)
    beta = rng.multivariate_normal(params.to_numpy(), cov, size=n_sim)
    base = stacked[stacked["run"] == reference_run]
    pred_all = interact + covar
    P = base[pred_all].to_numpy(dtype=float)
    main_idx = np.array([col_idx[p] for p in pred_all])

    def _si(b2d, run):
        s = b2d[:, main_idx].copy(); c = b2d[:, col_idx["const"]].copy()
        if run != reference_run:
            c = c + b2d[:, col_idx[f"m::{run}"]]
            for k, p in enumerate(pred_all):
                if p in interact:
                    s[:, k] = s[:, k] + b2d[:, col_idx[f"m::{run}:{p}"]]
        return s, c

    def _ame(run, pred, b2d):
        s, c = _si(b2d, run); lin = c[:, None] + s @ P.T
        k = pred_all.index(pred); L0 = lin - s[:, k][:, None] * P[:, k][None, :]; L1 = L0 + s[:, k][:, None]
        return (1/(1+np.exp(-L1)) - 1/(1+np.exp(-L0))).mean(axis=1)

    p2d = params.to_numpy()[None, :]
    ame_rows = []
    for p in interact:
        pt = {r: float(_ame(r, p, p2d)[0]) for r in runs}
        sim = {r: _ame(r, p, beta) for r in runs}
        for v in variants:
            d = sim[v] - sim[reference_run]
            ame_rows.append({"variant": v, "predictor": p.replace("z_", ""),
                             "ame_ref": pt[reference_run], "ame_var": pt[v],
                             "ame_diff": pt[v] - pt[reference_run],
                             "ci_low": float(np.quantile(d, alpha / 2)),
                             "ci_high": float(np.quantile(d, 1 - alpha / 2))})
    return {"interactions": interactions, "ame": pd.DataFrame(ame_rows),
            "n_obs": int(len(stacked)), "n_targets": int(stacked["formula_id"].nunique())}
