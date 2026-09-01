"""Stages B and C of Experiment 2: correctness models on the geometry frame.

THE SPECIFICATION IS NOT LINEAR, AND THAT IS A RESULT RATHER THAN A DETAIL.
Linearity in the logit is REJECTED for two of the three geometry covariates
(decile indicators against the linear term, likelihood ratio on 8 df):

    z_variance  LR 23.43  p = 2.9e-3      rejected
    u           LR 37.76  p = 8.3e-6      rejected
    z_faith     LR  8.22  p = 0.41        holds

(over a D + S base -- everything computationally prior and nothing else, so
the search does not presuppose its own outcome.) The models here therefore do
not use linear V or linear u anywhere. The forms were selected by
``spec_search`` BEFORE any estimand was read, and the search is reported in
full rather than summarised:

    V : quadratic. Captures 78% of the departure; residual against a decile
        reference p = 0.649. Adequate AT DECILE RESOLUTION only -- against a
        20-bin reference the quadratic still fails (p = 0.035), as do the
        cubic (p = 0.043) and deciles themselves (p = 0.008). V carries
        structure below decile width, most likely at the p(1-p) ceiling, and
        going one polynomial order higher does not reach it.
    u : deciles. The quadratic captures only 49% of the departure and its
        residual survives (p = 0.0072); deciles are not improved on by a
        20-bin cut (p = 0.41). Nine indicators, decile 0 as reference.
    F : linear. Nothing to repair at any resolution tested.

Both scale quantities were introduced as MONOTONE GOODS -- more variance means
the satisfaction vector says more, higher u means the anchor set is not
under-representing the embedding -- and both turn out to have an OPTIMUM. The
direction quantity does not. That contrast is the chapter's finding, and it is
why the specification search sits in the results rather than in an appendix.

Four questions and where they are read:

  Q1  satisfaction variance V      the V CURVE @ M2       (no scalar; see below)
  Q2  relational faithfulness F    beta_z_faith @ M4, with its +1 SD AME
  Q3  the residual norm u          the u CURVE @ M4       (no scalar)
  Q4  operator contrasts           joint has_op fit @ M1
      geometry deliberately absent, so the geometry path stays open: these are
      TOTAL contrasts and are not additive with the geometry readings.

NO SCALAR FOR V OR u. An average marginal effect is the mean vertical
displacement from sliding every target one SD along the response curve. On an
inverted U that averages positive displacements left of the peak against
negative ones right of it, so it measures where the population sits relative
to the optimum rather than the strength of the relationship. V's +1 SD AME is
-3.71 pp at M2 -- NEGATIVE, because the peak sits near the middle of the
distribution so shifting everyone up pushes most targets past it -- against
+1.18 pp under the rejected linear form. Both are defensible summaries of the
same curve and they disagree in sign, which is the disqualification. For u
under decile coding an AME of a shift is not even defined. The curve is the
estimand for both; only F, which is linear, gets a scalar.

The rung LATTICE. Syntax is COMPUTATIONALLY PRIOR (C5: S and D are pure
functions of the formula string, fixed before any trace is drawn), so it sits
at the base rather than entering as an adjustment step. Every rung includes
C(depth); every downward edge adds exactly one block::

              M0 : C(depth)                         [baseline]
               |  + has_op
               v
              M1 : + S                              Q4 lives here
               |  + z_variance + z_variance_sq       [operators.csv,
               v                                      depth_curve.csv]
              M2 : + V                              Q1's rung
              / \\
      + u    /   \\    + z_faith
            v     v
     M3u        M3F
            \\     /
      + F     \\  /   + u
               v
              M4 : + V + u + F                      Q2 + Q3's rung  [the meet]

The fork is symmetric, so "what does adding F do to u" (M3u -> M4) and "what
does adding u do to F" (M3F -> M4) are the same kind of step. It also gives V
two single-block steps from a common base: M2 -> M3u is what u alone does to V
(C1 predicts ~nothing) and M2 -> M3F is what F alone does (C4 predicts a lot).
That pair is C1-against-C4 measured rather than argued, and the older chained
lattice could not produce it because u was already inside M2.

M2 -> M3F is A STEP TOO FAR for Q1 and is computed only to price it: F is
downstream of V (C4), so conditioning on it changes what the V curve refers to.
The reported V curve is the one at M2.

CURVE SEQUENCES (``CURVE_SEQ``). Each covariate's curve is shown at raw, then
at D+S, then adding the other geometry blocks one at a time. The raw rate is
the empirical per-bin rate and carries no model. Because the curves are
descriptive rather than estimands, their sequence may include the D+S step that
the lattice no longer has -- which is how the syntax-absorption comparison
survives the move of syntax to the base.

Depth is entered as cell means (one indicator per depth, no shared intercept),
so each depth coefficient is that depth's absolute log-odds rather than a
contrast against a baseline depth; the covariate coefficients are unchanged.
u's decile indicators are derived in ``frame.derive_covariates``, not here, so
the bootstrap re-cuts them on every resample along with u itself.

Inference convention: HC1 robust SEs are attached to the point fits for the
tables, but the REPORTED intervals are 95% percentile-bootstrap CIs in which
the entire pipeline -- variance-bin edges, binned means/SDs, studentisation,
u's decile cuts, the Fisher-z faithfulness standardisation, and the model fits
-- is recomputed inside every resample. Cross-rung comparisons are made on the
probability scale, because log-odds coefficients are non-collapsible: they move
when outcome-predictive covariates enter even absent confounding (S3).

The intervals are CONDITIONAL ON THE SELECTED SPECIFICATION and do not account
for the search that chose it. Standard post-selection inference caveat; the
search is reported so a reader can price it.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from frame import OPERATORS, U_DEC_COLS, derive_covariates

HAS_COLS = [f"has_{op}" for op in OPERATORS]

# V's model representation: quadratic, per the specification search.
V_TERMS = ["z_variance", "z_variance_sq"]

M0_TERMS: list[str] = []
M1_TERMS = [*HAS_COLS]
M2_TERMS = [*V_TERMS, *HAS_COLS]
M3U_TERMS = [*V_TERMS, *U_DEC_COLS, *HAS_COLS]
M3F_TERMS = [*V_TERMS, "z_faith", *HAS_COLS]
M4_TERMS = [*V_TERMS, *U_DEC_COLS, "z_faith", *HAS_COLS]

LADDER = [("M0", M0_TERMS), ("M1", M1_TERMS), ("M2", M2_TERMS),
          ("M3u", M3U_TERMS), ("M3F", M3F_TERMS), ("M4", M4_TERMS)]
MODEL_TERMS = dict(LADDER)

# The linear specification, REJECTED. Retained solely so the rejected fits can
# be tabulated beside the search that rejected them -- without them the search
# is unauditable. Never read as a result.
REJECTED_LINEAR = {
    "L-M2": ["z_variance", *HAS_COLS],
    "L-M4": ["z_variance", "u", "z_faith", *HAS_COLS],
}

# Coefficients wired with bootstrap CIs, per rung. u's nine decile indicators
# are deliberately absent: individually they are not quantities anyone reads,
# and the curve is u's estimand.
REPORTED_COEFS = {
    "M2": ("z_variance", "z_variance_sq"),
    "M3u": ("z_variance", "z_variance_sq"),
    "M3F": ("z_variance", "z_variance_sq", "z_faith"),
    "M4": ("z_variance", "z_variance_sq", "z_faith"),
}

# +1 SD marginal effects. F ONLY -- an AME of a shift is uninterpretable on a
# non-monotone curve and undefined under decile coding (module docstring).
AME_TERMS = {
    "M3F": ("z_faith",),
    "M4": ("z_faith",),
}

# Attribution trajectories: term -> single-block steps. Only F has a scalar to
# attenuate; V's and u's attenuation is read off their curve sequences.
TRAJECTORIES = {
    "z_faith": (("M3F", "M4"),),
}

# Curve sequences: binning covariate -> ordered (label, adjustment set). The
# raw per-bin rate precedes all of these and carries no model. `primary` is the
# rung whose curve IS the reported estimand; the rest are the attenuation
# sequence, and for V the last step is priced but not read (C4).
CURVE_SEQ = {
    "z_variance": (("DS", [*HAS_COLS]),
                   ("DSu", [*HAS_COLS, *U_DEC_COLS]),
                   ("DSF", [*HAS_COLS, "z_faith"])),
    "u": (("DS", [*HAS_COLS]),
          ("DSV", [*HAS_COLS, *V_TERMS]),
          ("DSVF", [*HAS_COLS, *V_TERMS, "z_faith"])),
    "z_faith": (("DS", [*HAS_COLS]),
                ("DSV", [*HAS_COLS, *V_TERMS]),
                ("DSVu", [*HAS_COLS, *V_TERMS, *U_DEC_COLS])),
}
CURVE_PRIMARY = {"z_variance": "DS", "u": "DSVF", "z_faith": "DSVu"}

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
    """Movement of a coefficient from rung value ``a`` to ``b``: (delta, ratio).

    REPORT THE DELTA, NOT THE RATIO. The ratio is a quotient of two random
    quantities, so its bootstrap distribution is heavy-tailed whenever the
    lower rung ``a`` can sit near zero -- and here it does. Realised 95 %
    percentile intervals on the AME scale:

        term / step        delta (pp)            ratio
        u   M1->M2         +0.41 [-0.06, +0.92]  -0.57 [-10.04, +8.65]
        u   M2->M3         +0.46 [+0.15, +0.75]  -0.40 [ -4.85, +3.94]
        V   M2->M3         +0.66 [+0.21, +1.06]  +0.56 [ -0.02, +3.62]
        V   M1->M2         +0.78 [+0.24, +1.31]  +0.40 [ +0.13, +0.99]
        F   F1->F2         +1.16 [+0.67, +1.68]  +0.40 [ +0.22, +0.75]
        F   F2->M3         -0.37 [-0.61, +0.01]  -0.21 [ -0.73, +0.01]

    beta_u @ M1 is -0.055, which is what blows up both u rows. Every delta is
    narrower than 1.1 pp; only F1->F2 has a ratio worth quoting. Both columns
    stay in the tables (the ratio is the scale-free statement when it behaves),
    but no ratio may appear in the text without its CI attached, and none for
    u at all.

    Sign: delta = a - b, so its sign tracks the direction of the CHANGE, not
    distance from zero. Multiply by sign(a) before describing a movement as
    shrinkage -- otherwise a negative coefficient growing more negative reads
    as attenuation.
    """
    delta = float(a - b)
    ratio = float(1.0 - b / a) if abs(a) > 1e-8 else float("nan")
    return delta, ratio


# ------------------------------ the curves ---------------------------------- #

# A curve replaces the displayed covariate's model terms with decile-bin
# indicators and keeps everything else. At its PRIMARY step the adjustment set
# is exactly the rung whose estimand it is, so curve and rung are the same
# object at two levels of parametric commitment; the earlier steps are the
# attenuation sequence.
#
# The old cross-adjustment prohibition is gone, and the reason it was ever
# needed is worth recording: it applied only when adjustment was PARTIAL. F is
# a collider (V -> F <- U_geo, S/D -> F), so conditioning a u curve on F alone
# opens U_geo <-> V and U_geo <-> S. At the full set those paths are blocked by
# S and D. Empirically the distinction is small (adding F to the u curve moves
# it <= 0.010; adding u to the variance curve <= 0.002, itself a confirmation
# of C1), so it is stated with its magnitude rather than as a caveat.

def curve_rates(df: pd.DataFrame, curve_bins: int, *, col: str,
                adjust: list[str]) -> np.ndarray:
    """Adjusted correctness per ``col`` quantile bin (marginal standardisation).

    Returns (curve_bins,) adjusted rates; NaN for bins a degenerate qcut drops.
    ``col`` is the binning covariate on its MODEL scale (``z_variance``, ``u``
    or ``z_faith``); ``adjust`` is one entry of ``CURVE_SEQ[col]``. Depth is
    always present. Both arguments are required: there is no default adjustment
    set any more, because each covariate is shown at a sequence of them.
    """
    bins = pd.qcut(df[col], curve_bins, labels=False, duplicates="drop")
    y = df["correct"].to_numpy(dtype=np.float64)
    observed = sorted(int(b) for b in bins.dropna().unique())

    cols: dict[str, np.ndarray] = {"const": np.ones(len(df))}
    for b in observed[1:]:
        cols[f"bin_{b}"] = (bins == b).to_numpy(dtype=np.float64)
    for d in sorted(df["depth"].unique())[1:]:
        cols[f"depth_{d}"] = (df["depth"] == d).to_numpy(dtype=np.float64)
    for t in adjust:
        cols[t] = df[t].to_numpy(dtype=np.float64)
    X = pd.DataFrame(cols, index=df.index)
    beta = _fit(y, X).params.reindex(X.columns).to_numpy()

    rates = np.full(curve_bins, np.nan)
    Xc = X.copy()
    for b in observed:
        for bb in observed[1:]:
            Xc[f"bin_{bb}"] = 1.0 if bb == b else 0.0
        rates[b] = _sigmoid(Xc.to_numpy() @ beta).mean()
    return rates


def curve_descriptives(df: pd.DataFrame, curve_bins: int, *, col: str,
                       extra_means: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-bin descriptives for a curve, binned on ``col``'s MODEL scale.

    ``raw_rate`` is the empirical rate and carries no model -- it is the first
    step of every curve sequence. ``extra_means`` averages further columns per
    bin so a figure can label the axis in interpretable units (raw variance,
    raw rho) while the bin positions stay on the scale the models use. Binning
    on the model scale rather than the raw one also removes a Jensen gap: the
    map to z_faith is convex, so atanh(mean rho) != mean(atanh rho), and the
    first decile's plotted position was off by 2.3% of the axis.
    """
    bins = pd.qcut(df[col], curve_bins, labels=False, duplicates="drop")
    g = df.assign(_bin=bins).groupby("_bin")
    out = pd.DataFrame({
        "bin": sorted(int(b) for b in bins.dropna().unique())}).set_index("bin")
    out["n"] = g.size()
    out[f"mean_{col}"] = g[col].mean()
    for extra in extra_means:
        out[f"mean_{extra}"] = g[extra].mean()
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
    """Q4, at rung M1: all eight operators jointly, plus C(depth).

    ``log_odds`` is operator presence at matched depth AND matched co-occurring
    operators; ``gap`` is its 0 -> 1 marginally standardised probability
    companion. Geometry is deliberately absent, so the geometry path stays open
    and these are TOTAL contrasts -- not additive with the geometry readings,
    since all three geometry covariates are 12-16% operator-determined.

    The one-operator-at-a-time fits are gone. They defended a choice nobody
    contests (all eight enter together, as in any regression) using the
    vocabulary of confounding this chapter does not use, and nothing was read
    from them.
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    Xop = _design(dfc, HAS_COLS)
    joint = _fit(y, Xop, cov_type="HC1")
    return pd.DataFrame([{
        "operator": op,
        "prevalence": float(dfc[f"has_{op}"].mean()),
        "log_odds": float(joint.params[f"has_{op}"]),
        "hc1_se": float(joint.bse[f"has_{op}"]),
        "gap": marginal_gap(joint.params, Xop, col=f"has_{op}"),
    } for op in OPERATORS])


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


# -------------------- specification search and adequacy --------------------- #

# Candidate forms for a geometry covariate. Box-Tidwell is unavailable here and
# not merely awkward: it adds x*ln(x) and so requires x > 0, while all three
# covariates are centred and take negative values -- and u, a signed residual
# by construction, has no positive scale at any point. The replacement is a
# grouped lack-of-fit test (Hosmer-Lemeshow, Applied Logistic Regression 4.2.1):
# swap the term for quantile indicators and compare by likelihood ratio. It is
# sign-agnostic, assumes no functional form for the alternative, and its
# alternative hypothesis IS the decile curve we plot -- so the test and the
# figure are one object at two levels of commitment, which is a virtue for
# exposition and a reason not to count them as separate evidence.
SPEC_BASE = [*HAS_COLS]        # D + S: everything computationally prior, and
                               # nothing else, so the search is not circular.


def _spec_forms(df: pd.DataFrame, col: str, fine_bins: int) -> dict:
    """Design columns for each candidate form of ``col``."""
    q10 = pd.qcut(df[col], 10, labels=False, duplicates="drop")
    qf = pd.qcut(df[col], fine_bins, labels=False, duplicates="drop")
    v = df[col].to_numpy(dtype=np.float64)
    forms = {
        "linear": {col: v},
        "quadratic": {col: v, f"{col}^2": v ** 2},
        "cubic": {col: v, f"{col}^2": v ** 2, f"{col}^3": v ** 3},
        "deciles": {f"d{k}": (q10 == k).to_numpy(np.float64)
                    for k in range(1, int(q10.max()) + 1)},
        f"bins{fine_bins}": {f"f{k}": (qf == k).to_numpy(np.float64)
                             for k in range(1, int(qf.max()) + 1)},
    }
    return forms


def spec_search(dfc: pd.DataFrame, *, fine_bins: int = 20) -> pd.DataFrame:
    """The ladder that chose the specification. Reported in full, not summarised.

    Every candidate form of every geometry covariate is tested against two
    flexible references -- deciles and a ``fine_bins`` cut -- over a common
    D + S base. Reporting both references is what exposes that the quadratic is
    adequate for V at decile resolution and inadequate at 20 bins, which a
    single reference would have hidden.
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for col in ("z_variance", "u", "z_faith"):
        forms = _spec_forms(dfc, col, fine_bins)
        fits = {}
        for name, extra in forms.items():
            X = _design(dfc, SPEC_BASE)
            for k, vec in extra.items():
                X[k] = vec
            fits[name] = (_fit(y, X), X.shape[1])
        for ref in ("deciles", f"bins{fine_bins}"):
            ref_fit, ref_k = fits[ref]
            for name, (fit, k) in fits.items():
                if k >= ref_k:
                    continue
                lr = 2.0 * (ref_fit.llf - fit.llf)
                dfree = ref_k - k
                rows.append({"term": col, "form": name, "reference": ref,
                             "lr": float(lr), "df": int(dfree),
                             "p": float(stats.chi2.sf(lr, dfree))})
    return pd.DataFrame(rows)


def spec_curves(dfc: pd.DataFrame, *, n_grid: int = 60,
                span: tuple[float, float] = (0.5, 99.5)) -> pd.DataFrame:
    """Fitted response curves for the linear and quadratic forms, D + S base.

    The specification figure overlays these on the decile points from
    ``curve_*.csv``'s DS step -- which is the SAME base, so the points and the
    lines are comparable rather than merely adjacent. Marginally standardised,
    like everything else on the probability scale.

    Emitted for every covariate under both forms even where only one is used:
    for u the selected form is the decile indicators, so the "fit" to draw is
    the decile curve itself and only the rejected linear line is overlaid.
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for col in ("z_variance", "u", "z_faith"):
        lo, hi = np.percentile(dfc[col].to_numpy(), span)
        grid = np.linspace(lo, hi, n_grid)
        for form in ("linear", "quadratic"):
            X = _design(dfc, SPEC_BASE)
            X[col] = dfc[col].to_numpy(dtype=np.float64)
            if form == "quadratic":
                X[f"{col}_sq"] = X[col] ** 2
            params = _fit(y, X).params.reindex(X.columns).to_numpy()
            for g in grid:
                Xg = X.copy()
                Xg[col] = g
                if form == "quadratic":
                    Xg[f"{col}_sq"] = g ** 2
                rows.append({"term": col, "form": form, "x": float(g),
                             "rate": float(_sigmoid(Xg.to_numpy() @ params).mean())})
    return pd.DataFrame(rows)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Concordance: P(a correct target scores above an incorrect one)."""
    r = stats.rankdata(p)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def model_adequacy(dfc: pd.DataFrame) -> pd.DataFrame:
    """The adequacy checks the score equations do NOT already enforce.

    Calibration is absent by design. Maximum likelihood solves X'(y - p) = 0,
    one equation per design column, so predicted and observed match exactly
    within every depth cell, every operator group, and overall -- checking it
    would be checking that the solver converged. Where calibration CAN fail is
    in groupings outside the column space, and there it is algebraically the
    same quantity as the decile curve's departure from the fitted form, which
    ``spec_search`` already tests with more power.

    What survives is what the score equations leave free:
      * DISCRIMINATION (AUC), which depends only on the ordering of the
        predictions. Reported for nested blocks in both orders, so the
        syntax/geometry overlap is a reported number rather than an artefact
        of whichever block is entered first.
      * THE LINK (Pregibon): eta^2 is not in the column space.
      * INFLUENCE (dfbeta), which is not the leverage reported on the transform
        figures -- leverage is a property of the design, influence combines it
        with how badly the outcome is predicted.
    """
    y = dfc["correct"].to_numpy(dtype=np.float64)
    geo = [*V_TERMS, *U_DEC_COLS, "z_faith"]
    blocks = {"D": [], "D+S": [*HAS_COLS], "D+G": geo, "D+S+G": [*HAS_COLS, *geo]}
    rows = []
    auc = {}
    for name, terms in blocks.items():
        X = _design(dfc, terms)
        auc[name] = _auc(y, np.asarray(_fit(y, X).predict(X)))
        rows.append({"stat": f"auc_{name}", "value": auc[name]})
    rows += [
        {"stat": "auc_syntax_given_D", "value": auc["D+S"] - auc["D"]},
        {"stat": "auc_syntax_unique", "value": auc["D+S+G"] - auc["D+G"]},
        {"stat": "auc_geometry_given_D", "value": auc["D+G"] - auc["D"]},
        {"stat": "auc_geometry_unique", "value": auc["D+S+G"] - auc["D+S"]},
        # Identity, not a consistency check: both directions reduce to the same
        # expression, so their agreeing carries no information.
        {"stat": "auc_shared",
         "value": auc["D+S"] + auc["D+G"] - auc["D"] - auc["D+S+G"]},
    ]
    for name, terms in MODEL_TERMS.items():
        X = _design(dfc, terms)
        fit = _fit(y, X)
        eta = X.to_numpy() @ fit.params.reindex(X.columns).to_numpy()
        lt = _fit(y, pd.DataFrame({"eta": eta, "eta2": eta ** 2}, index=dfc.index))
        rows.append({"stat": f"link_z_{name}",
                     "value": float(lt.params["eta2"] / lt.bse["eta2"])})
    for rung, terms in REPORTED_COEFS.items():
        X = _design(dfc, MODEL_TERMS[rung])
        infl = _fit(y, X).get_influence()
        dfb = pd.DataFrame(infl.dfbetas, columns=X.columns)
        for term in terms:
            rows.append({"stat": f"max_abs_dfbeta_{rung}_{term}",
                         "value": float(dfb[term].abs().max())})
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

# Curve stores: one (B, curve_bins) array per (covariate, sequence step). The
# whole sequence is bootstrapped, not just the primary step, so the attenuation
# figure carries intervals rather than bare point estimates -- and so that an
# across-step difference within one bin, which is a PAIRED contrast on the same
# targets, can be given an interval far tighter than either step's marginal one.
CURVE_KEYS = [f"curve_{col}_{step}" for col, seq in CURVE_SEQ.items()
              for step, _ in seq]

_FIT_RUNGS = tuple(name for name, _ in LADDER if name != "M0")


def bootstrap(df_raw: pd.DataFrame, idx: np.ndarray, *, n_bins: int,
              curve_bins: int, depth_levels: list[int],
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
    for key in CURVE_KEYS:
        store[key] = np.full((B, curve_bins), np.nan)
    store["op_joint"] = np.full((B, n_ops), np.nan)
    store["op_gap"] = np.full((B, n_ops), np.nan)
    store["depth_raw"] = np.full((B, len(depth_levels)), np.nan)
    store["depth_adj"] = np.full((B, len(depth_levels)), np.nan)
    failures = {k: 0 for k in ("lattice", *CURVE_KEYS, "operators")}

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
        for col, seq in CURVE_SEQ.items():
            for step, adjust in seq:
                key = f"curve_{col}_{step}"
                try:
                    store[key][b] = curve_rates(dfc, curve_bins, col=col,
                                                adjust=adjust)
                except Exception:
                    failures[key] += 1
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
