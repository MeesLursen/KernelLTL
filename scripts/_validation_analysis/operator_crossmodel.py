"""Fair cross-model comparison of operator effects.

Three coordinated pieces (see the thesis methods discussion):

1. **Pooled interaction model** — one logistic regression on the stacked,
   common-target data with ``model x (has_op + target_depth)`` interactions and
   cluster-robust SEs by ``formula_id``. The ``model[variant]:has_OP``
   coefficient is the *test* of whether operator OP's effect differs between
   that variant and the reference run. BH-FDR over the interaction family.

2. **AME differences** — average marginal effect of each operator on
   ``P(outcome)`` per model, differenced variant-vs-reference, on the shared
   validation covariate distribution (probability scale, comparable). CIs by
   parametric simulation from the cluster-robust covariance.

3. **Operator-stratified McNemar** — assumption-light paired cross-check:
   among targets containing OP, McNemar (+ Cohen's κ) of variant vs reference.

All three operate on the *common* ``formula_id`` set so the design matrix is
shared across models (target-side / AST confounding differenced out).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from scripts._validation_analysis.operator_analysis import OPERATORS, COVARIATES
from scripts._validation_analysis.extra_contrast import cohen_kappa, _mcnemar_p_and_effect


def _predictor_cols() -> list[str]:
    return [f"has_{op}" for op in OPERATORS] + list(COVARIATES)


def _restrict_common(df_op: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    """Keep only formula_ids present for every run (shared design)."""
    sub = df_op[df_op["run"].isin(runs)]
    id_sets = [set(sub.loc[sub["run"] == r, "formula_id"]) for r in runs]
    common = set.intersection(*id_sets) if id_sets else set()
    return sub[sub["formula_id"].isin(common)].copy()


def _build_design(
    frame: pd.DataFrame,
    *,
    variants: list[str],
    pred_cols: list[str],
) -> pd.DataFrame:
    """Design matrix: const + model dummies + predictor main effects +
    model x predictor interactions. Reference model is the omitted dummy.

    ``frame`` must have a ``run`` column and the ``pred_cols``.
    """
    X = pd.DataFrame(index=frame.index)
    X["const"] = 1.0
    # Model main effects (one indicator per variant; reference omitted).
    for v in variants:
        X[f"m::{v}"] = (frame["run"] == v).astype(float)
    # Predictor main effects (these are the reference model's slopes).
    for p in pred_cols:
        X[p] = frame[p].astype(float)
    # Interactions: variant slope minus reference slope.
    for v in variants:
        mv = (frame["run"] == v).astype(float).to_numpy()
        for p in pred_cols:
            X[f"m::{v}:{p}"] = mv * frame[p].astype(float).to_numpy()
    return X


def fit_pooled_interaction(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    reference_run: str,
    outcome_col: str,
    alpha: float = 0.05,
    n_sim: int = 2000,
    rng_seed: int = 0,
    extra_controls: tuple[str, ...] = (),
) -> dict:
    """Fit the pooled model and return interaction tests + AME differences.

    ``extra_controls`` (e.g. ``("z_variance", "z_norm_resid")``) are extra columns the caller
    has joined onto ``df_op``; they enter the design model-interacted (so each model keeps its
    own geometry slope -- the full adjustment that the ce_finetune slope change requires) but
    are EXCLUDED from the reported interaction family, so the operator BH-FDR set and the AME
    loop (operators only) are unchanged -- just geometry-adjusted.

    Returns dict with keys:
      ``interactions``  : DataFrame (variant, predictor, op, coef, se,
                          ci_low, ci_high, p_value)  — model x predictor terms
      ``ame``           : DataFrame (variant, op, ame_ref, ame_var, ame_diff,
                          ci_low, ci_high)
      ``n_obs``, ``n_targets`` : ints
    """
    variants = [r for r in runs if r != reference_run]
    # extra_controls appended AFTER operators+covariates so OPERATORS.index(op) still maps
    # into the design; model-interacted (full adjustment) but excluded from reporting below.
    pred_cols = _predictor_cols() + list(extra_controls)
    stacked = _restrict_common(df_op, runs)
    stacked = stacked.dropna(subset=[outcome_col] + pred_cols)
    if stacked.empty or stacked[outcome_col].nunique() < 2:
        return {"interactions": pd.DataFrame(),
                "interactions_pairwise": pd.DataFrame(),
                "ame": pd.DataFrame(), "ame_pairwise": pd.DataFrame(),
                "n_obs": 0, "n_targets": 0}

    X = _build_design(stacked, variants=variants, pred_cols=pred_cols)
    y = stacked[outcome_col].astype(int).to_numpy()
    groups = stacked["formula_id"].to_numpy()

    try:
        res = sm.Logit(y, X.to_numpy()).fit(
            disp=False, maxiter=300,
            cov_type="cluster", cov_kwds={"groups": groups},
        )
    except Exception:
        return {"interactions": pd.DataFrame(),
                "interactions_pairwise": pd.DataFrame(),
                "ame": pd.DataFrame(), "ame_pairwise": pd.DataFrame(),
                "n_obs": int(len(stacked)),
                "n_targets": int(stacked["formula_id"].nunique())}

    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    params = pd.Series(res.params, index=cols)
    bse = pd.Series(res.bse, index=cols)
    pvals = pd.Series(res.pvalues, index=cols)
    z = abs(float(norm.ppf(alpha / 2)))
    n_params = len(cols)

    # --- 1a. Interaction tests vs the reference (treatment coding) -------------
    inter_rows = []
    for v in variants:
        for p in pred_cols:
            if p in extra_controls:        # geometry controls: adjusted-for, not reported
                continue
            name = f"m::{v}:{p}"
            if name not in params.index:
                continue
            b, s, pv = float(params[name]), float(bse[name]), float(pvals[name])
            inter_rows.append({
                "variant": v, "predictor": p, "op": p.replace("has_", ""),
                "coef": b, "se": s,
                "ci_low": b - z * s, "ci_high": b + z * s,
                "p_value": pv,
            })
    interactions = pd.DataFrame(inter_rows)

    # --- 1b. All-pairs interaction contrasts (operators only) -----------------
    # slope_a(OP) - slope_b(OP) = gamma_{a,OP} - gamma_{b,OP}; the reference
    # has no interaction column (its gamma == 0), and the OP main effect
    # cancels in any same-operator slope difference. Wald-tested against the
    # cluster-robust covariance via res.t_test.
    pw_meta = []          # (op, run_a, run_b)
    contrast_rows = []     # parallel restriction-matrix rows
    for op in OPERATORS:
        p = f"has_{op}"
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
                    continue  # a == b (shouldn't happen) or both reference
                pw_meta.append((op, a, b))
                contrast_rows.append(c)

    interactions_pairwise = pd.DataFrame()
    if contrast_rows:
        C = np.vstack(contrast_rows)
        tt = res.t_test(C)
        eff = np.atleast_1d(np.asarray(tt.effect)).ravel()
        sd = np.atleast_1d(np.asarray(tt.sd)).ravel()
        pv = np.atleast_1d(np.asarray(tt.pvalue)).ravel()
        ci = np.atleast_2d(tt.conf_int(alpha=alpha))
        interactions_pairwise = pd.DataFrame([
            {"op": m[0], "run_a": m[1], "run_b": m[2],
             "coef": float(eff[k]), "se": float(sd[k]),
             "ci_low": float(ci[k, 0]), "ci_high": float(ci[k, 1]),
             "p_value": float(pv[k])}
            for k, m in enumerate(pw_meta)
        ])

    # --- 2. AME (g-computation) + parametric-sim CIs --------------------------
    cov = np.asarray(res.cov_params())
    rng = np.random.default_rng(rng_seed)
    beta_draws = rng.multivariate_normal(params.to_numpy(), cov, size=n_sim)

    # Vectorised g-computation. For a fixed model the design is fully
    # determined by the per-target predictor matrix P plus that model's
    # effective intercept/slopes, so AME(op, run, beta) collapses to a few
    # numpy matmuls — algebraically identical to rebuilding the design per
    # call, but ~10^2-10^3x faster (no per-call pandas DataFrame build).
    base = stacked[stacked["run"] == reference_run]
    P = base[pred_cols].to_numpy(dtype=float)        # (T, K) target predictors
    const_i = col_idx["const"]
    main_idx = np.array([col_idx[p] for p in pred_cols])  # (K,)

    def _slopes_intercept(beta2d: np.ndarray, run: str):
        # beta2d: (S, n_params) -> slopes s (S, K), intercept c (S,)
        s = beta2d[:, main_idx].copy()
        c = beta2d[:, const_i].copy()
        if run != reference_run:
            c = c + beta2d[:, col_idx[f"m::{run}"]]
            for k, p in enumerate(pred_cols):
                s[:, k] = s[:, k] + beta2d[:, col_idx[f"m::{run}:{p}"]]
        return s, c

    def _ame_vec(run: str, op: str, beta2d: np.ndarray) -> np.ndarray:
        """Return AME(op) under `run` for each of the S rows of beta2d."""
        s, c = _slopes_intercept(beta2d, run)        # (S,K), (S,)
        full_lin = c[:, None] + s @ P.T              # (S, T)
        k = OPERATORS.index(op)                      # has_op col in pred_cols
        s_op = s[:, k]                               # (S,)
        x_op = P[:, k]                               # (T,)
        L0 = full_lin - s_op[:, None] * x_op[None, :]   # has_op := 0
        L1 = L0 + s_op[:, None]                          # has_op := 1
        sig0 = 1.0 / (1.0 + np.exp(-L0))
        sig1 = 1.0 / (1.0 + np.exp(-L1))
        return (sig1 - sig0).mean(axis=1)            # (S,)

    params_2d = params.to_numpy()[None, :]           # (1, n_params)

    ame_rows, ame_pw_rows = [], []
    for op in OPERATORS:
        # Point + per-sim AME for every run (cache → pairwise comes free).
        ame_pt = {r: float(_ame_vec(r, op, params_2d)[0]) for r in runs}
        ame_sim = {r: _ame_vec(r, op, beta_draws) for r in runs}
        # vs reference
        for v in variants:
            sims = ame_sim[v] - ame_sim[reference_run]
            ame_rows.append({
                "variant": v, "op": op,
                "ame_ref": ame_pt[reference_run], "ame_var": ame_pt[v],
                "ame_diff": ame_pt[v] - ame_pt[reference_run],
                "ci_low": float(np.quantile(sims, alpha / 2)),
                "ci_high": float(np.quantile(sims, 1 - alpha / 2)),
            })
        # all pairs
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                sims = ame_sim[a] - ame_sim[b]
                ame_pw_rows.append({
                    "op": op, "run_a": a, "run_b": b,
                    "ame_a": ame_pt[a], "ame_b": ame_pt[b],
                    "ame_diff": ame_pt[a] - ame_pt[b],
                    "ci_low": float(np.quantile(sims, alpha / 2)),
                    "ci_high": float(np.quantile(sims, 1 - alpha / 2)),
                })
    ame = pd.DataFrame(ame_rows)
    ame_pairwise = pd.DataFrame(ame_pw_rows)

    return {
        "interactions": interactions,
        "interactions_pairwise": interactions_pairwise,
        "ame": ame,
        "ame_pairwise": ame_pairwise,
        "n_obs": int(len(stacked)),
        "n_targets": int(stacked["formula_id"].nunique()),
    }


def operator_stratified_mcnemar(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    reference_run: str,
    outcome_col: str = "correct",
) -> pd.DataFrame:
    """Per-(operator, variant) McNemar + Cohen's κ vs the reference run,
    restricted to targets that contain the operator. Shared common targets."""
    variants = [r for r in runs if r != reference_run]
    stacked = _restrict_common(df_op, runs)
    rows = []
    for op in OPERATORS:
        has_col = f"has_{op}"
        sub = stacked[stacked[has_col] == 1]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="formula_id", columns="run",
                                values=outcome_col).dropna(how="any")
        if reference_run not in pivot.columns:
            continue
        y_ref = pivot[reference_run].astype(int).to_numpy()
        for v in variants:
            if v not in pivot.columns:
                continue
            y_var = pivot[v].astype(int).to_numpy()
            if len(y_ref) == 0:
                continue
            p, eff = _mcnemar_p_and_effect(y_var, y_ref)  # +eff => variant better
            rows.append({
                "op": op, "variant": v,
                "n_targets_with_op": int(len(y_ref)),
                "kappa": cohen_kappa(y_var, y_ref),
                "mcnemar_p": p,
                "mcnemar_effect": eff,
                "acc_ref": float(y_ref.mean()),
                "acc_var": float(y_var.mean()),
            })
    return pd.DataFrame(rows)


def operator_stratified_mcnemar_pairwise(
    df_op: pd.DataFrame,
    *,
    runs: list[str],
    outcome_col: str = "correct",
) -> pd.DataFrame:
    """All-pairs operator-stratified McNemar + Cohen's κ. For each operator,
    restrict to targets containing it (shared common set), then every
    unordered run pair. ``mcnemar_effect`` is signed by ``run_a`` (positive =>
    run_a more often correct than run_b)."""
    stacked = _restrict_common(df_op, runs)
    rows = []
    for op in OPERATORS:
        sub = stacked[stacked[f"has_{op}"] == 1]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="formula_id", columns="run",
                                values=outcome_col).dropna(how="any")
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                if a not in pivot.columns or b not in pivot.columns:
                    continue
                ya = pivot[a].astype(int).to_numpy()
                yb = pivot[b].astype(int).to_numpy()
                if len(ya) == 0:
                    continue
                p, eff = _mcnemar_p_and_effect(ya, yb)  # +eff => run_a better
                rows.append({
                    "op": op, "run_a": a, "run_b": b,
                    "n_targets_with_op": int(len(ya)),
                    "kappa": cohen_kappa(ya, yb),
                    "mcnemar_p": p,
                    "mcnemar_effect": eff,
                    "acc_a": float(ya.mean()),
                    "acc_b": float(yb.mean()),
                })
    return pd.DataFrame(rows)
