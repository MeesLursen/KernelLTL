"""Stages B and C of Experiment 2: correctness models on the geometry frame.

Four questions, four primary readings (the confirmatory family; reported in
full with 95% percentile-bootstrap CIs and NO multiplicity adjustment -- the
project is declared exploratory, the family is small and pre-stated):

  Q1  effect of satisfaction variance    beta_z_variance @ M2
      the TOTAL effect: faithfulness is a mediator of variance and stays out
      of M2; causal under A2 + A3 (ASSUMPTIONS.md)
  Q2  effect of relational faithfulness  beta_z_faith @ M3
      causal under A1 + A3
  Q3  effect of the residual norm u      beta_u @ M3
      causal under A1 + A3
  Q4  operator-structure contrasts       joint has_op fits @ S rung
      comparative-associational; geometry covariates deliberately excluded,
      so the embedding-geometry path stays open (total contrasts)

M3 is the minimal model in which beta_u and beta_z_faith are interpretable:
each closes the other's backdoor through the shared registration-geometry
latent, so they enter together or not at all. u sits in M1/M2 uninterpreted:
by construction it is no descendant of variance, so it cannot disturb the M2
reading; it adds precision and makes M2 -> M3 a single-term step.

The rung LATTICE (logistic, outcome = greedy semantic equivalence; every rung
includes C(depth); every downward edge adds exactly one block)::

                          M0 : C(depth)                        [baseline]
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                      ▼
    S : has_op           M1 : V + u             F1 : V + F     [branch starts]
    [operators.csv,           │ +has_op              │ +has_op
     depth_curve.csv]         ▼                      ▼
    Q4 lives here        M2 : V + u + S         F2 : V + F + S
                         Q1 lives here               │
                              │ +z_faith             │ +u
                              └─────────┬────────────┘
                                        ▼
                              M3 : V + u + F + S               [the meet]
                              Q2 + Q3 live here
                                        │
                                        ▼
                              M3q : M3 + u^2                   [curvature check
                                                                for the Q3 rung]

Secondary strand -- attribution trajectories along the lattice (single-block
steps, so each movement has one interpretation): u along M1 -> M2 -> M3
(syntax share, then shared-latent share), z_faith along F1 -> F2 -> M3
(syntax share, then scale share), z_variance along M1 -> M2 -> M3
(compositional step, then mediation-through-F share). Cross-rung comparisons
are made on the probability scale (marginal effects: average +1 SD
g-computation shifts), because log-odds coefficients are non-collapsible:
they move when outcome-predictive covariates enter even absent confounding.

Depth is entered as cell means (one indicator per depth, no shared intercept),
so each depth coefficient is that depth's absolute log-odds rather than a
contrast against a baseline depth; the covariate coefficients are unchanged.

Descriptive curves are companions, not rungs, and deliberately carry lighter
adjustment than the rung they accompany: the variance-decile curve (depth-
adjusted) pairs with Q1; the u-decile curve (depth-adjusted; variance-adjusted
by construction) pairs with Q3; the faithfulness-decile curves (depth-adjusted
and depth+variance-adjusted) pair with Q2. u and faithfulness curves are never
adjusted for each other: both are coarsenings of the same embedding vector
(scale vs direction), and conditioning one while displaying the other induces
selection distortion through the shared latent.

Inference convention: HC1 robust SEs are attached to the point fits for the
tables, but the REPORTED intervals are 95% percentile-bootstrap CIs in which
the entire pipeline -- variance-bin edges, binned means/SDs, studentisation,
the Fisher-z faithfulness standardisation, and the model fits -- is recomputed
inside every resample.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from frame import OPERATORS, derive_covariates

HAS_COLS = [f"has_{op}" for op in OPERATORS]

M1_TERMS = ["z_variance", "u"]
M2_TERMS = ["z_variance", "u", *HAS_COLS]
F1_TERMS = ["z_variance", "z_faith"]
F2_TERMS = ["z_variance", "z_faith", *HAS_COLS]
M3_TERMS = ["z_variance", "u", "z_faith", *HAS_COLS]
M3Q_TERMS = ["z_variance", "u", "u_sq", "z_faith", *HAS_COLS]

LADDER = [("M0", []), ("M1", M1_TERMS), ("M2", M2_TERMS),
          ("F1", F1_TERMS), ("F2", F2_TERMS), ("M3", M3_TERMS),
          ("M3q", M3Q_TERMS)]
MODEL_TERMS = dict(LADDER)

# Coefficients wired with bootstrap CIs, per rung.
REPORTED_COEFS = {
    "M1": ("z_variance", "u"),
    "M2": ("z_variance", "u"),
    "F1": ("z_variance", "z_faith"),
    "F2": ("z_variance", "z_faith"),
    "M3": ("z_variance", "u", "z_faith"),
    "M3q": ("z_variance", "u", "u_sq", "z_faith"),
}

# Continuous covariates whose +1 SD marginal effects are reported per rung.
AME_TERMS = {
    "M1": ("z_variance", "u"),
    "M2": ("z_variance", "u"),
    "F1": ("z_variance", "z_faith"),
    "F2": ("z_variance", "z_faith"),
    "M3": ("z_variance", "u", "z_faith"),
}

# Attribution trajectories: term -> single-block steps along the lattice.
TRAJECTORIES = {
    "u": (("M1", "M2"), ("M2", "M3")),
    "z_faith": (("F1", "F2"), ("F2", "M3")),
    "z_variance": (("M1", "M2"), ("M2", "M3")),
}

DEFAULT_VAR_ADJUST_BINS = 10

ALPHA = 0.05


# ------------------------------ fitting ------------------------------------ #

def _design(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Design matrix with cell-means depth coding.

    Depth is entered as one indicator per level with NO shared intercept, so
    each ``depth_d`` coefficient is that depth's absolute log-odds (at the
    covariate reference) rather than a contrast against a baseline depth. The
    column space equals a reference-cell + intercept design, so the covariate
    coefficients (u, z_variance, z_faith, has_op, ...) and their SEs are
    unchanged; only the depth block is reparametrised.
    """
    cols: dict[str, np.ndarray] = {}
    for t in terms:
        cols[t] = df[t].to_numpy(dtype=np.float64)
    for d in sorted(df["depth"].unique()):
        cols[f"depth_{d}"] = (df["depth"] == d).to_numpy(dtype=np.float64)
    return pd.DataFrame(cols, index=df.index)


def _fit(y: np.ndarray, X: pd.DataFrame, *, cov_type: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Binomial())
        return model.fit(cov_type=cov_type) if cov_type else model.fit()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def marginal_gap(params: pd.Series, X: pd.DataFrame, col: str) -> float:
    """Average predicted P(correct) with ``col`` forced to 1 minus forced to 0."""
    b = params.reindex(X.columns).to_numpy()
    X1, X0 = X.copy(), X.copy()
    X1[col], X0[col] = 1.0, 0.0
    return float(_sigmoid(X1.to_numpy() @ b).mean()
                 - _sigmoid(X0.to_numpy() @ b).mean())


def marginal_shift(params: pd.Series, X: pd.DataFrame, col: str,
                   delta: float = 1.0) -> float:
    """Average predicted P(correct) after adding ``delta`` to ``col``, minus observed.

    The probability-scale (collapsible) companion to a log-odds coefficient;
    the scale on which effects are compared ACROSS lattice rungs.
    """
    b = params.reindex(X.columns).to_numpy()
    X1 = X.copy()
    X1[col] = X1[col] + delta
    return float(_sigmoid(X1.to_numpy() @ b).mean()
                 - _sigmoid(X.to_numpy() @ b).mean())


def attenuation(a: float, b: float) -> tuple[float, float]:
    """Movement of a coefficient from rung value ``a`` to ``b``: (delta, ratio)."""
    delta = float(a - b)
    ratio = float(1.0 - b / a) if abs(a) > 1e-8 else float("nan")
    return delta, ratio


# ------------------------------ the curves ---------------------------------- #

def curve_rates(df: pd.DataFrame, curve_bins: int, *, col: str = "u",
                var_bins: int | None = None) -> np.ndarray:
    """Adjusted correctness per ``col`` quantile bin (marginal standardisation).

    Returns (curve_bins,) adjusted rates; NaN for bins a degenerate qcut drops.
    ``col`` selects the binning covariate: ``u``, ``variance``, or
    ``relational_faithfulness``. Depth is always in the adjustment set; when
    ``var_bins`` is given, variance quantile-bin indicators join it (the
    depth+variance-adjusted faithfulness variant).
    """
    bins = pd.qcut(df[col], curve_bins, labels=False, duplicates="drop")
    y = df["correct"].to_numpy(dtype=np.float64)
    observed = sorted(int(b) for b in bins.dropna().unique())

    cols: dict[str, np.ndarray] = {"const": np.ones(len(df))}
    for b in observed[1:]:
        cols[f"bin_{b}"] = (bins == b).to_numpy(dtype=np.float64)
    for d in sorted(df["depth"].unique())[1:]:
        cols[f"depth_{d}"] = (df["depth"] == d).to_numpy(dtype=np.float64)
    if var_bins is not None:
        vb = pd.qcut(df["variance"], var_bins, labels=False, duplicates="drop")
        for v in sorted(int(x) for x in vb.dropna().unique())[1:]:
            cols[f"vadj_{v}"] = (vb == v).to_numpy(dtype=np.float64)
    X = pd.DataFrame(cols, index=df.index)
    beta = _fit(y, X).params.reindex(X.columns).to_numpy()

    rates = np.full(curve_bins, np.nan)
    Xc = X.copy()
    for b in observed:
        for bb in observed[1:]:
            Xc[f"bin_{bb}"] = 1.0 if bb == b else 0.0
        rates[b] = _sigmoid(Xc.to_numpy() @ beta).mean()
    return rates


def curve_descriptives(df: pd.DataFrame, curve_bins: int, *,
                       col: str = "u", label: str = "u") -> pd.DataFrame:
    bins = pd.qcut(df[col], curve_bins, labels=False, duplicates="drop")
    g = df.assign(_bin=bins).groupby("_bin")
    out = pd.DataFrame({
        "bin": sorted(int(b) for b in bins.dropna().unique())}).set_index("bin")
    out["n"] = g.size()
    out[f"mean_{label}"] = g[col].mean()
    out["raw_rate"] = g["correct"].mean()
    out["mean_sem_dist"] = g["sem_dist"].mean()
    return out.reset_index()


# ---------------------------- point estimates ------------------------------- #

def point_ladder(dfc: pd.DataFrame) -> pd.DataFrame:
    """All lattice fits with HC1 SEs (point estimates; CIs come from the bootstrap)."""
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for name, terms in LADDER:
        res = _fit(y, _design(dfc, terms), cov_type="HC1")
        for term in res.params.index:
            rows.append({"model": name, "term": term,
                         "estimate": float(res.params[term]),
                         "hc1_se": float(res.bse[term])})
    return pd.DataFrame(rows)


def point_marginals(dfc: pd.DataFrame) -> pd.DataFrame:
    """+1 SD marginal effects per rung (the cross-rung comparison scale)."""
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for name, terms in AME_TERMS.items():
        X = _design(dfc, MODEL_TERMS[name])
        params = _fit(y, X).params
        for term in terms:
            rows.append({"model": name, "term": term,
                         "estimate": marginal_shift(params, X, term)})
    return pd.DataFrame(rows)


def point_operators(dfc: pd.DataFrame) -> pd.DataFrame:
    """The S rung (H2c operator exhibit): joint and single-operator fits.

    log_odds_joint: operator presence at matched depth AND matched co-occurring
    operators (the defensible adjusted comparison); gap_joint is its 0 -> 1
    marginally standardised probability companion. log_odds_single: the
    one-operator-at-a-time fit, kept as the co-occurrence-confounded companion
    column -- its movement against the joint column exhibits that confounding.
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    Xop = _design(dfc, HAS_COLS)
    joint = _fit(y, Xop, cov_type="HC1")
    rows = []
    for op in OPERATORS:
        single = _fit(y, _design(dfc, [f"has_{op}"]), cov_type="HC1")
        rows.append({
            "operator": op,
            "prevalence": float(dfc[f"has_{op}"].mean()),
            "log_odds_joint": float(joint.params[f"has_{op}"]),
            "hc1_se_joint": float(joint.bse[f"has_{op}"]),
            "gap_joint": marginal_gap(joint.params, Xop, col=f"has_{op}"),
            "log_odds_single": float(single.params[f"has_{op}"]),
            "hc1_se_single": float(single.bse[f"has_{op}"]),
        })
    return pd.DataFrame(rows)


def point_depth_curve(dfc: pd.DataFrame) -> pd.DataFrame:
    """Raw and operator-standardised correctness per depth level.

    adj_rate comes from the S-rung joint fit: every target keeps its observed
    operator profile, its depth cell is forced to d, and the predicted
    probabilities are averaged (marginal standardisation over the operator mix).
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    Xop = _design(dfc, HAS_COLS)
    bvec = _fit(y, Xop).params.reindex(Xop.columns).to_numpy()
    depths = sorted(int(d) for d in dfc["depth"].unique())
    rows = []
    for d in depths:
        mask = dfc["depth"] == d
        Xd = Xop.copy()
        for dd in depths:
            Xd[f"depth_{dd}"] = 1.0 if dd == d else 0.0
        rows.append({"depth": d, "n": int(mask.sum()),
                     "raw_rate": float(dfc.loc[mask, "correct"].mean()),
                     "adj_rate": float(_sigmoid(Xd.to_numpy() @ bvec).mean())})
    return pd.DataFrame(rows)


# ------------------------------ bootstrap ----------------------------------- #

def _attenuation_keys(prefix: str = "") -> list[str]:
    keys = []
    for term, steps in TRAJECTORIES.items():
        for a, b in steps:
            for stat in ("delta", "ratio"):
                keys.append(f"{prefix}att_{term}_{a}_{b}_{stat}")
    return keys


BOOT_KEYS = (
    [f"{m}_{t}" for m, terms in REPORTED_COEFS.items() for t in terms]
    + [f"ame_{m}_{t}" for m, terms in AME_TERMS.items() for t in terms]
    + _attenuation_keys() + _attenuation_keys("ame_")
)

_FIT_RUNGS = ("M1", "M2", "F1", "F2", "M3", "M3q")


def bootstrap(df_raw: pd.DataFrame, idx: np.ndarray, *, n_bins: int,
              curve_bins: int, depth_levels: list[int],
              var_adjust_bins: int = DEFAULT_VAR_ADJUST_BINS,
              log_every: int = 1000) -> dict:
    """Whole-pipeline percentile bootstrap over targets.

    Every resample re-derives the covariates (bin edges, binned means/SDs,
    studentisation, the Fisher-z standardisation) before refitting, so
    first-stage estimation uncertainty is inside the intervals.
    ``depth_levels`` fixes the depth axis of the depth-curve stores globally;
    a resample missing a level contributes NaN there.
    """
    B = idx.shape[0]
    n_ops = len(OPERATORS)
    store: dict = {k: np.full(B, np.nan) for k in BOOT_KEYS}
    store["curve"] = np.full((B, curve_bins), np.nan)
    store["var_curve"] = np.full((B, curve_bins), np.nan)
    store["faith_curve"] = np.full((B, curve_bins), np.nan)
    store["faith_curve_vd"] = np.full((B, curve_bins), np.nan)
    store["op_single"] = np.full((B, n_ops), np.nan)
    store["op_joint"] = np.full((B, n_ops), np.nan)
    store["op_gap"] = np.full((B, n_ops), np.nan)
    store["depth_raw"] = np.full((B, len(depth_levels)), np.nan)
    store["depth_adj"] = np.full((B, len(depth_levels)), np.nan)
    failures = {k: 0 for k in ("lattice", "curve", "var_curve", "faith_curve",
                               "faith_curve_vd", "operators", "op_single")}

    for b in range(B):
        sub = df_raw.iloc[idx[b]].reset_index(drop=True)
        dfc = derive_covariates(sub, n_bins=n_bins)
        y = dfc["correct"].to_numpy(dtype=np.float64)
        try:
            fits: dict = {}
            for name in _FIT_RUNGS:
                X = _design(dfc, MODEL_TERMS[name])
                fits[name] = (_fit(y, X).params, X)

            for name, terms in REPORTED_COEFS.items():
                params = fits[name][0]
                for term in terms:
                    store[f"{name}_{term}"][b] = params[term]

            for name, terms in AME_TERMS.items():
                params, X = fits[name]
                for term in terms:
                    store[f"ame_{name}_{term}"][b] = marginal_shift(params, X, term)

            for term, steps in TRAJECTORIES.items():
                for a, bb in steps:
                    d, r = attenuation(fits[a][0][term], fits[bb][0][term])
                    store[f"att_{term}_{a}_{bb}_delta"][b] = d
                    store[f"att_{term}_{a}_{bb}_ratio"][b] = r
                    d, r = attenuation(store[f"ame_{a}_{term}"][b],
                                       store[f"ame_{bb}_{term}"][b])
                    store[f"ame_att_{term}_{a}_{bb}_delta"][b] = d
                    store[f"ame_att_{term}_{a}_{bb}_ratio"][b] = r
        except Exception:
            failures["lattice"] += 1
        try:
            store["curve"][b] = curve_rates(dfc, curve_bins)
        except Exception:
            failures["curve"] += 1
        try:
            store["var_curve"][b] = curve_rates(dfc, curve_bins, col="variance")
        except Exception:
            failures["var_curve"] += 1
        try:
            store["faith_curve"][b] = curve_rates(
                dfc, curve_bins, col="relational_faithfulness")
        except Exception:
            failures["faith_curve"] += 1
        try:
            store["faith_curve_vd"][b] = curve_rates(
                dfc, curve_bins, col="relational_faithfulness",
                var_bins=var_adjust_bins)
        except Exception:
            failures["faith_curve_vd"] += 1
        try:
            Xop = _design(dfc, HAS_COLS)
            op_params = _fit(y, Xop).params
            for j, op in enumerate(OPERATORS):
                store["op_joint"][b, j] = op_params[f"has_{op}"]
                store["op_gap"][b, j] = marginal_gap(op_params, Xop,
                                                     col=f"has_{op}")
            present = sorted(int(d) for d in dfc["depth"].unique())
            bvec = op_params.reindex(Xop.columns).to_numpy()
            for k, d in enumerate(depth_levels):
                mask = dfc["depth"] == d
                if not mask.any():
                    continue
                store["depth_raw"][b, k] = float(dfc.loc[mask, "correct"].mean())
                Xd = Xop.copy()
                for dd in present:
                    Xd[f"depth_{dd}"] = 1.0 if dd == d else 0.0
                store["depth_adj"][b, k] = float(_sigmoid(Xd.to_numpy() @ bvec).mean())
        except Exception:
            failures["operators"] += 1
        try:
            for j, op in enumerate(OPERATORS):
                store["op_single"][b, j] = _fit(
                    y, _design(dfc, [f"has_{op}"])).params[f"has_{op}"]
        except Exception:
            failures["op_single"] += 1
        if log_every and (b + 1) % log_every == 0:
            print(f"[exp2] bootstrap {b + 1}/{B}", file=sys.stderr, flush=True)

    store["failures"] = failures
    return store


def ci(samples: np.ndarray) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = np.nanpercentile(samples, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)],
                                  axis=0)
    return lo, hi
