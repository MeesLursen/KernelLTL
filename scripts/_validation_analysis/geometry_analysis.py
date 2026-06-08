"""Embedding-geometry vs. correctness analysis.

Tests the thesis's central (but previously untested) claim chain: conditioning
strength scales with embedding magnitude, whose two failure modes are (i) low
formula variance and (ii) anchor orthogonality. We regress per-target greedy /
top-K correctness on:

    std        = sqrt(Var(satvec))        -- informativeness  (cause i)
    alignment  = ||rho_phi||              -- anchor coverage   (cause ii)
    target_depth                          -- covariate

Mirrors the operator analyses:
  * per-model multivariate logistic (effect size + 95% CI headline, no BH),
  * descriptive marginal binned curves + a 2-D std x alignment grid,
  * a pooled cross-model interaction (model x predictor, cluster-robust SE by
    formula_id, BH-FDR over the interaction family) + AME differences
    (probability scale, parametric-sim CIs) -- the "did finetuning change the
    model's reliance on embedding geometry?" test.

Multiple-comparisons family structure matches the suite: BH-FDR is applied to
the cross-model interaction family; per-model coefficients report CIs (no BH).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from scripts._validation_analysis.extra_metrics import bootstrap_mean_ci

# Core geometry predictors + covariate. Standardised (z-scored) before fitting,
# so coefficients / AMEs read as "per +1 SD".
GEOMETRY_PREDICTORS = ["std", "alignment"]
COVARIATES = ["target_depth"]


def predictor_cols() -> list[str]:
    return [f"z_{c}" for c in GEOMETRY_PREDICTORS + COVARIATES]


# ---------------------------------------------------------------------------
# Frame construction (join geometry features to per-(run, formula_id) correctness)
# ---------------------------------------------------------------------------


def build_frame(
    features: pd.DataFrame,
    correctness: pd.DataFrame,
) -> pd.DataFrame:
    """Join target geometry features onto long-form correctness.

    ``correctness`` columns: run, formula_id, correct, target_depth.
    Standardisation uses the per-target feature distribution (one value per
    formula_id), then is broadcast to every (run, formula_id) row.
    """
    feat = features.copy()
    # z-score on the unique-target distribution (geometry is run-invariant).
    for c in GEOMETRY_PREDICTORS:
        mu, sd = feat[c].mean(), feat[c].std(ddof=0) or 1.0
        feat[f"z_{c}"] = (feat[c] - mu) / sd

    merged = correctness.merge(
        feat[["formula_id"] + GEOMETRY_PREDICTORS + [f"z_{c}" for c in GEOMETRY_PREDICTORS]],
        on="formula_id", how="inner",
    )
    mu_d, sd_d = merged["target_depth"].mean(), merged["target_depth"].std(ddof=0) or 1.0
    merged["z_target_depth"] = (merged["target_depth"] - mu_d) / sd_d
    return merged


# ---------------------------------------------------------------------------
# Per-model logistic  (descriptive; CI headline, no BH)  -- mirrors operator_analysis
# ---------------------------------------------------------------------------


def per_model_logistic(
    df: pd.DataFrame,
    *,
    runs: list[str],
    outcome_col: str = "correct",
    alpha: float = 0.05,
    use_regularized: bool = False,
) -> pd.DataFrame:
    """correct ~ z_std + z_alignment + z_target_depth, per run."""
    preds = predictor_cols()
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        if rdf.empty or rdf[outcome_col].nunique() < 2:
            continue
        X = sm.add_constant(rdf[preds].astype(float), has_constant="add")
        y = rdf[outcome_col].astype(int)
        try:
            res = sm.Logit(y, X).fit(disp=False, maxiter=200)
            conf = res.conf_int(alpha=alpha)
            for c in preds:
                rows.append({
                    "run": r, "predictor": c.replace("z_", ""),
                    "coef": float(res.params[c]),
                    "ci_low": float(conf.loc[c, 0]), "ci_high": float(conf.loc[c, 1]),
                    "p_value": float(res.pvalues[c]), "converged": True,
                })
        except Exception:
            if not use_regularized:
                continue
            res = sm.Logit(y, X).fit_regularized(disp=False, maxiter=200, alpha=1.0)
            for c in preds:
                rows.append({
                    "run": r, "predictor": c.replace("z_", ""),
                    "coef": float(res.params[c]), "ci_low": float("nan"),
                    "ci_high": float("nan"), "p_value": float("nan"), "converged": False,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive marginal binned curves + 2-D grid
# ---------------------------------------------------------------------------


def marginal_binned(
    df: pd.DataFrame,
    *,
    runs: list[str],
    feature: str,          # raw feature name: 'std' or 'alignment'
    n_bins: int = 12,
    outcome_col: str = "correct",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Mean correctness by quantile bin of ``feature``, per run, with bootstrap CIs."""
    rng = np.random.default_rng(rng_seed)
    edges = np.unique(np.quantile(df[feature].to_numpy(), np.linspace(0, 1, n_bins + 1)))
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        idx = np.clip(np.digitize(rdf[feature].to_numpy(), edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            vals = rdf.loc[idx == b, outcome_col].astype(float).to_numpy()
            if len(vals) == 0:
                continue
            m, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            rows.append({
                "run": r, "feature": feature, "bin": b,
                "x_mid": float(0.5 * (edges[b] + edges[b + 1])),
                "mean": m, "ci_low": lo, "ci_high": hi, "n": int(len(vals)),
            })
    return pd.DataFrame(rows)


def two_d_grid(
    df: pd.DataFrame,
    *,
    runs: list[str],
    fx: str = "std",
    fy: str = "alignment",
    nbins: int = 8,
    outcome_col: str = "correct",
) -> pd.DataFrame:
    """Mean correctness over an fx x fy quantile grid, per run (the heatmap data).

    The orthogonality case (high fx=std, low fy=alignment) is the low-correctness
    cell that the theory predicts."""
    ex = np.unique(np.quantile(df[fx].to_numpy(), np.linspace(0, 1, nbins + 1)))
    ey = np.unique(np.quantile(df[fy].to_numpy(), np.linspace(0, 1, nbins + 1)))
    rows = []
    for r in runs:
        rdf = df[df["run"] == r]
        ix = np.clip(np.digitize(rdf[fx].to_numpy(), ex[1:-1]), 0, len(ex) - 2)
        iy = np.clip(np.digitize(rdf[fy].to_numpy(), ey[1:-1]), 0, len(ey) - 2)
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                vals = rdf.loc[(ix == a) & (iy == b), outcome_col].astype(float).to_numpy()
                rows.append({
                    "run": r, "ix": a, "iy": b,
                    "x_mid": float(0.5 * (ex[a] + ex[a + 1])),
                    "y_mid": float(0.5 * (ey[b] + ey[b + 1])),
                    "mean_correct": float(np.mean(vals)) if len(vals) else float("nan"),
                    "n": int(len(vals)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-model pooled interaction + AME  (BH-FDR over the interaction family)
#   Self-contained (copies the tiny design helpers) to stay additive.
# ---------------------------------------------------------------------------


def _restrict_common(df: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    sub = df[df["run"].isin(runs)]
    id_sets = [set(sub.loc[sub["run"] == r, "formula_id"]) for r in runs]
    common = set.intersection(*id_sets) if id_sets else set()
    return sub[sub["formula_id"].isin(common)].copy()


def _build_design(frame: pd.DataFrame, *, variants: list[str], pred_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=frame.index)
    X["const"] = 1.0
    for v in variants:
        X[f"m::{v}"] = (frame["run"] == v).astype(float)
    for p in pred_cols:
        X[p] = frame[p].astype(float)
    for v in variants:
        mv = (frame["run"] == v).astype(float).to_numpy()
        for p in pred_cols:
            X[f"m::{v}:{p}"] = mv * frame[p].astype(float).to_numpy()
    return X


def fit_pooled_interaction(
    df: pd.DataFrame,
    *,
    runs: list[str],
    reference_run: str,
    outcome_col: str = "correct",
    alpha: float = 0.05,
    n_sim: int = 2000,
    rng_seed: int = 0,
) -> dict:
    """Pooled logistic `outcome ~ model x (z_std + z_alignment + z_depth)` with
    cluster-robust SE by formula_id. Returns interaction tests (BH-FDR over the
    family) + AME differences (probability scale). Predictors are standardised,
    so an interaction coef is the per-+1-SD slope difference vs the reference."""
    variants = [r for r in runs if r != reference_run]
    pred_cols = predictor_cols()
    stacked = _restrict_common(df, runs).dropna(subset=[outcome_col] + pred_cols)
    empty = {"interactions": pd.DataFrame(), "interactions_pairwise": pd.DataFrame(),
             "ame": pd.DataFrame(), "ame_pairwise": pd.DataFrame(),
             "n_obs": int(len(stacked)), "n_targets": 0}
    if stacked.empty or stacked[outcome_col].nunique() < 2:
        return empty

    X = _build_design(stacked, variants=variants, pred_cols=pred_cols)
    y = stacked[outcome_col].astype(int).to_numpy()
    groups = stacked["formula_id"].to_numpy()
    try:
        res = sm.Logit(y, X.to_numpy()).fit(
            disp=False, maxiter=300, cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        empty["n_targets"] = int(stacked["formula_id"].nunique())
        return empty

    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    params = pd.Series(res.params, index=cols)
    bse = pd.Series(res.bse, index=cols)
    pvals = pd.Series(res.pvalues, index=cols)
    z = abs(float(norm.ppf(alpha / 2)))

    # --- interaction tests vs reference (treatment coding) -> BH-FDR family ---
    inter_rows = []
    for v in variants:
        for p in pred_cols:
            name = f"m::{v}:{p}"
            if name not in params.index:
                continue
            b, s, pv = float(params[name]), float(bse[name]), float(pvals[name])
            inter_rows.append({
                "variant": v, "predictor": p.replace("z_", ""),
                "coef": b, "se": s, "ci_low": b - z * s, "ci_high": b + z * s,
                "p_value": pv,
            })
    interactions = pd.DataFrame(inter_rows)
    if not interactions.empty:
        from statsmodels.stats.multitest import multipletests
        interactions["p_value_adj_bh"] = multipletests(
            interactions["p_value"].fillna(1.0).to_numpy(), alpha=alpha, method="fdr_bh")[1]
        interactions["reject_bh"] = interactions["p_value_adj_bh"] < alpha

    # --- all-pairs interaction contrasts (per predictor) via Wald t_test ------
    n_params = len(cols)
    pw_meta, contrast_rows = [], []
    for p in pred_cols:
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                c = np.zeros(n_params)
                ca, cb = f"m::{a}:{p}", f"m::{b}:{p}"
                if ca in col_idx:
                    c[col_idx[ca]] += 1.0
                if cb in col_idx:
                    c[col_idx[cb]] -= 1.0
                if not np.any(c):
                    continue
                pw_meta.append((p.replace("z_", ""), a, b))
                contrast_rows.append(c)
    interactions_pairwise = pd.DataFrame()
    if contrast_rows:
        tt = res.t_test(np.vstack(contrast_rows))
        eff = np.atleast_1d(np.asarray(tt.effect)).ravel()
        sd = np.atleast_1d(np.asarray(tt.sd)).ravel()
        pv = np.atleast_1d(np.asarray(tt.pvalue)).ravel()
        ci = np.atleast_2d(tt.conf_int(alpha=alpha))
        interactions_pairwise = pd.DataFrame([
            {"predictor": mta[0], "run_a": mta[1], "run_b": mta[2],
             "coef": float(eff[k]), "se": float(sd[k]),
             "ci_low": float(ci[k, 0]), "ci_high": float(ci[k, 1]), "p_value": float(pv[k])}
            for k, mta in enumerate(pw_meta)
        ])

    # --- AME (g-computation, probability scale) + parametric-sim CIs ----------
    cov = np.asarray(res.cov_params())
    rng = np.random.default_rng(rng_seed)
    beta_draws = rng.multivariate_normal(params.to_numpy(), cov, size=n_sim)
    base = stacked[stacked["run"] == reference_run]
    P = base[pred_cols].to_numpy(dtype=float)
    const_i = col_idx["const"]
    main_idx = np.array([col_idx[p] for p in pred_cols])

    def _slopes_intercept(beta2d, run):
        s = beta2d[:, main_idx].copy()
        c = beta2d[:, const_i].copy()
        if run != reference_run:
            c = c + beta2d[:, col_idx[f"m::{run}"]]
            for k, p in enumerate(pred_cols):
                s[:, k] = s[:, k] + beta2d[:, col_idx[f"m::{run}:{p}"]]
        return s, c

    def _ame_vec(run, pred, beta2d):
        # AME of raising standardised `pred` from its mean (0) to +1 SD.
        s, c = _slopes_intercept(beta2d, run)
        full_lin = c[:, None] + s @ P.T
        k = pred_cols.index(pred)
        s_p = s[:, k]
        x_p = P[:, k]
        L0 = full_lin - s_p[:, None] * x_p[None, :]      # pred := mean
        L1 = L0 + s_p[:, None]                            # pred := +1 SD
        return (1.0 / (1.0 + np.exp(-L1)) - 1.0 / (1.0 + np.exp(-L0))).mean(axis=1)

    params_2d = params.to_numpy()[None, :]
    ame_rows, ame_pw_rows = [], []
    for p in pred_cols:
        pt = {r: float(_ame_vec(r, p, params_2d)[0]) for r in runs}
        sim = {r: _ame_vec(r, p, beta_draws) for r in runs}
        lab = p.replace("z_", "")
        for v in variants:
            d = sim[v] - sim[reference_run]
            ame_rows.append({
                "variant": v, "predictor": lab,
                "ame_ref": pt[reference_run], "ame_var": pt[v],
                "ame_diff": pt[v] - pt[reference_run],
                "ci_low": float(np.quantile(d, alpha / 2)),
                "ci_high": float(np.quantile(d, 1 - alpha / 2)),
            })
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                d = sim[a] - sim[b]
                ame_pw_rows.append({
                    "predictor": lab, "run_a": a, "run_b": b,
                    "ame_a": pt[a], "ame_b": pt[b], "ame_diff": pt[a] - pt[b],
                    "ci_low": float(np.quantile(d, alpha / 2)),
                    "ci_high": float(np.quantile(d, 1 - alpha / 2)),
                })

    return {
        "interactions": interactions,
        "interactions_pairwise": interactions_pairwise,
        "ame": pd.DataFrame(ame_rows),
        "ame_pairwise": pd.DataFrame(ame_pw_rows),
        "n_obs": int(len(stacked)),
        "n_targets": int(stacked["formula_id"].nunique()),
    }
