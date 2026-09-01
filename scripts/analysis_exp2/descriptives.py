"""Stage A of Experiment 2: model-free audit of the representation.

Every function here consumes only dataset/kernel-derived quantities -- no
model outcome enters (the shuffle-null check reads the *shuffle-ablation*
run, whose whole point is that the conditioning is destroyed). All outputs
are descriptive or diagnostic: population-level statements about the 4,000
validation targets, no inference attached.

These tables are also the SOLE SOURCE for the construction figures. Nothing in
``plot_exp2`` recomputes a statistic that appears here: two implementations of
one number with no cross-check is how a figure comes to disagree with its own
table, which has happened twice in this pipeline already.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from frame import OPERATORS, U_DEC_COLS

TERC_LABELS = ("low", "mid", "high")
GEOMETRY = ("z_variance", "u", "z_faith")


def _terciles(s: pd.Series) -> pd.Series:
    return pd.qcut(s, 3, labels=TERC_LABELS)


def _leverage_share(v: np.ndarray, q: float) -> float:
    """Share of sum(z^2) owned by the bottom ``q`` per cent.

    A GLM weights each observation by its squared distance from the covariate
    mean, so this is the fraction of the design's pull that the lower tail
    carries. Scale-free: invariant to any affine transform of ``v``.
    """
    z = (v - v.mean()) / v.std()
    k = int(len(v) * q / 100)
    return float((z ** 2)[np.argsort(v)][:k].sum() / (z ** 2).sum())


# ------------------------------- occupancy ---------------------------------- #

def occupancy(df: pd.DataFrame, *, rows: str, cols: str,
              bins: int = 10) -> pd.DataFrame:
    """Quantile-bin cross-tabulation of two covariates. Long form.

    Serves two different arguments and it is worth keeping them apart:

    * ``variance`` x ``emb_norm`` MOTIVATES A CONSTRUCTION. The two are almost
      the same variable -- 49 of 100 decile cells are empty and the corners are
      unpopulated -- so the design supplies no norm contrast at fixed variance
      and u had to be built to manufacture one.
    * ``z_variance`` x ``z_faith`` LIMITS A CONCLUSION. High F never co-occurs
      with low V, so when the F curve forces F to its top decile for every
      target, the ones in the bottom V deciles get a prediction for a
      combination that never occurs. That is a positivity limit of the design,
      and by C4 (low variance -> noisy Spearman -> F attenuated) more data
      cannot fill it in.
    """
    rb = pd.qcut(df[rows], bins, labels=False, duplicates="drop")
    cb = pd.qcut(df[cols], bins, labels=False, duplicates="drop")
    # Reindex over the full grid: the EMPTY cells are the finding, and a
    # groupby on integer labels drops them silently rather than reporting zero.
    grid = pd.MultiIndex.from_product(
        [range(int(rb.max()) + 1), range(int(cb.max()) + 1)],
        names=["row_bin", "col_bin"])
    counts = (pd.DataFrame({"row_bin": rb, "col_bin": cb})
              .groupby(["row_bin", "col_bin"]).size()
              .reindex(grid, fill_value=0).reset_index(name="n"))
    counts.insert(0, "rows", rows)
    counts.insert(1, "cols", cols)
    return counts


def norm_variance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Why the raw norm cannot enter a model that already contains variance.

    The occupancy grid shows the emptiness; these are its consequences. The
    VIF pair is the exhibit: entering the two jointly is degenerate, and the
    construction takes it to exactly 1.
    """
    ln = df["log10_norm"].to_numpy()
    lv = np.log10(df["variance"].to_numpy())
    slope, intercept = np.polyfit(lv, ln, 1)
    resid = ln - (intercept + slope * lv)

    def vif(a: np.ndarray, b: np.ndarray) -> float:
        r2 = float(np.corrcoef(a, b)[0, 1] ** 2)
        return 1.0 / (1.0 - r2) if r2 < 1 else np.inf

    rows = [
        # The ridge. A pure-scale embedding would give slope 1/2 exactly
        # (||emb|| ~ sqrt(V)); the excess is how much further the norm falls
        # away as variance drops, on top of what scale alone demands.
        ("ridge_slope", float(slope)),
        ("ridge_intercept", float(intercept)),
        ("ridge_r2", float(np.corrcoef(lv, ln)[0, 1] ** 2)),
        ("ridge_slope_scale_reference", 0.5),
        # The 1/2 slope is a CEILING, not a central tendency. emb(phi) is a
        # projection of the satisfaction vector onto the anchor set, so
        # ||emb|| <= K sqrt(V) by Cauchy-Schwarz with K = ||F_c||/N a property
        # of the anchor set alone; on log axes that is an upper bound of slope
        # exactly 1/2, attained only where the anchors capture the formula
        # fully. Every target lies on or below it, and the vertical gap to it
        # IS the under-exposure u measures. This intercept is the empirical
        # envelope: the largest observed log10||emb|| - (1/2) log10 V.
        ("scale_ceiling_intercept", float((ln - 0.5 * lv).max())),
        ("scale_ceiling_gap_median", float(-(ln - 0.5 * lv - (ln - 0.5 * lv).max())
                                           .mean())),
        ("ridge_resid_sd", float(resid.std())),
        ("corr_log10_norm_log10_variance", float(np.corrcoef(ln, lv)[0, 1])),
        ("spearman_emb_norm_variance",
         float(stats.spearmanr(df["emb_norm"], df["variance"]).statistic)),
        ("vif_emb_norm_with_variance",
         vif(df["emb_norm"].to_numpy(), df["variance"].to_numpy())),
        ("vif_log10_norm_with_log10_variance", vif(ln, lv)),
        # The construction's target, and its realised value.
        ("vif_u_with_z_variance",
         vif(df["u"].to_numpy(), df["z_variance"].to_numpy())),
        ("spearman_u_variance",
         float(stats.spearmanr(df["u"], df["variance"]).statistic)),
    ]
    # C1's edge case, quantified rather than asserted: inside the lowest
    # variance bins "within-bin" does not mean "at matched variance".
    lo = df[df["vbin"] <= 1]
    rows += [
        ("c1_edge_frac_targets_bins01", float(len(lo) / len(df))),
        ("c1_edge_bin0_variance_decades",
         float(np.log10(df.loc[df["vbin"] == 0, "variance"].max()
                        / df.loc[df["vbin"] == 0, "variance"].min()))),
        ("c1_edge_bin0_spearman_u_variance",
         float(stats.spearmanr(df.loc[df["vbin"] == 0, "u"],
                               df.loc[df["vbin"] == 0, "variance"]).statistic)),
        ("c1_edge_bins01_leverage_share",
         float(((df["u"] / df["u"].std()) ** 2)[df["vbin"] <= 1].sum()
               / ((df["u"] / df["u"].std()) ** 2).sum())),
    ]
    return pd.DataFrame(rows, columns=["stat", "value"])


def norm_variance_curve(df: pd.DataFrame) -> pd.DataFrame:
    """The binned conditional E[log10_norm | variance] that u residualises against.

    Column names carry the base explicitly: a bare ``mean_log_norm`` tells a
    reader nothing about whether the units are nats or decades, and the two
    differ by ln 10. They are DECADES.
    """
    g = df.groupby("vbin")
    out = pd.DataFrame({"vbin": sorted(df["vbin"].unique())}).set_index("vbin")
    out["n"] = g.size()
    out["min_variance"] = g["variance"].min()
    out["max_variance"] = g["variance"].max()
    out["mean_variance"] = g["variance"].mean()
    out["mean_log10_norm"] = g["log10_norm"].mean()
    out["sd_log10_norm"] = g["log10_norm"].std()
    out["mean_u"] = g["u"].mean()
    out["sd_u"] = g["u"].std()
    return out.reset_index()


# --------------------------- covariate distributions ------------------------ #

def covariate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Shape and leverage for every covariate, before and after its transform.

    One table for all of it, because the ARGUMENT is a comparison: the two
    quantities that needed constructing are the two whose lower 5 per cent
    carried more than half the design's pull, and V -- which needed nothing --
    sits at the uniform reference. Reference values under a uniform and a
    normal covariate are included so "13.9 %" can be read without simulating.
    """
    fz = np.arctanh(df["relational_faithfulness"].clip(-1 + 1e-6, 1 - 1e-6))
    series = {
        "relational_faithfulness": df["relational_faithfulness"].to_numpy(),
        "fisher_z_faith": fz.to_numpy(),
        "z_faith": df["z_faith"].to_numpy(),
        "emb_norm": df["emb_norm"].to_numpy(),
        "log10_norm": df["log10_norm"].to_numpy(),
        "u": df["u"].to_numpy(),
        "variance": df["variance"].to_numpy(),
        "z_variance": df["z_variance"].to_numpy(),
    }
    rows = []
    for name, v in series.items():
        rows.append({
            "covariate": name, "n": len(v),
            "mean": float(v.mean()), "sd": float(v.std()),
            "skew": float(stats.skew(v)),
            "min": float(v.min()), "max": float(v.max()),
            "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "leverage_bottom_5pct": _leverage_share(v, 5.0),
            "leverage_top_5pct": _leverage_share(-v, 5.0),
        })
    # Closed-form-ish references, so the table interprets itself.
    rng = np.random.default_rng(0)
    for name, draw in (("_reference_uniform", rng.uniform),
                       ("_reference_normal", rng.normal)):
        lev = np.mean([_leverage_share(draw(size=len(df)), 5.0) for _ in range(200)])
        rows.append({"covariate": name, "n": len(df), "mean": np.nan, "sd": np.nan,
                     "skew": 0.0, "min": np.nan, "max": np.nan, "p05": np.nan,
                     "p50": np.nan, "p95": np.nan,
                     "leverage_bottom_5pct": float(lev),
                     "leverage_top_5pct": float(lev)})
    return pd.DataFrame(rows)


def faith_by_variance(df: pd.DataFrame) -> pd.DataFrame:
    """C4's reliability channel, in one table.

    Variance sets the signal-to-noise ratio of the Spearman estimate, so rank
    noise attenuates F toward zero where variance is low. Attenuation shrinks
    everything -- the level, the spread, and the responsiveness to any
    covariate -- and all three move together across the strata, inversely to
    the estimation-noise proxy. Three strata is suggestive, not decisive, but
    the correspondence is exact in ordering including the non-obvious part:
    F peaks at MID variance, which is also where the noise floor is lowest.
    """
    d = df.assign(_v=_terciles(df["variance"]), _u=_terciles(df["u"]))
    g = d.groupby("_v", observed=False)
    out = pd.DataFrame({
        "variance_terc": list(TERC_LABELS)}).set_index("variance_terc")
    out["n"] = g.size()
    out["mean_variance"] = g["variance"].mean()
    out["mean_z_faith"] = g["z_faith"].mean()
    out["sd_z_faith"] = g["z_faith"].std()
    out["mean_sem_dist"] = g["sem_dist"].mean()
    # The u-responsiveness of F inside each stratum: the low->mid tercile step.
    cell = d.groupby(["_v", "_u"], observed=False)["z_faith"].agg(["mean", "size", "std"])
    step, step_se = [], []
    for terc in TERC_LABELS:
        lo, mid = cell.loc[(terc, "low")], cell.loc[(terc, "mid")]
        step.append(float(mid["mean"] - lo["mean"]))
        step_se.append(float(np.hypot(mid["std"] / np.sqrt(mid["size"]),
                                      lo["std"] / np.sqrt(lo["size"]))))
    out["u_step_low_to_mid"] = step
    out["u_step_se"] = step_se
    return out.reset_index()


# ------------------------------- syntax bridge ------------------------------ #

def design_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """How syntax-determined is each geometry covariate?

    Two jobs. It certifies IDENTIFICATION -- if syntax fixed a covariate
    entirely there would be no within-syntax variation left to read a
    coefficient from, and 84-88 per cent of each covariate's variance survives
    operator adjustment. And it quantifies the ENTANGLEMENT: the geometry
    covariates are computed downstream of the formula string (C5), so the
    operator contrasts at M1 -- which carry no geometry term -- already contain
    whatever an operator does by shifting V, u and F. The two sets of numbers
    are therefore not additive and not rankable against each other.

    Computed without outcome data.
    """
    H = df[[f"has_{op}" for op in OPERATORS]].to_numpy(dtype=np.float64)
    D = pd.get_dummies(df["depth"], prefix="d").to_numpy(dtype=np.float64)
    pattern = (df[[f"has_{op}" for op in OPERATORS]].astype(int).astype(str)
               .agg("".join, axis=1))
    rows = []
    for col in GEOMETRY:
        y = df[col].to_numpy(dtype=np.float64)
        for label, X in (("has_op", np.column_stack([np.ones(len(y)), H])),
                         ("has_op_depth",
                          np.column_stack([np.ones(len(y)), H, D[:, 1:]]))):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rows.append((f"r2_{col}_on_{label}",
                         1.0 - (y - X @ beta).var() / y.var()))
        within = y - pd.Series(y).groupby(pattern.values).transform("mean").to_numpy()
        rows += [(f"r2_{col}_on_pattern_fe", 1.0 - within.var() / y.var()),
                 (f"within_pattern_sd_{col}", float(np.sqrt(within.var()))),
                 (f"total_sd_{col}", float(y.std()))]
    sizes = pattern.value_counts()
    rows += [("n_patterns", float(sizes.size)),
             ("median_pattern_size", float(sizes.median()))]
    return pd.DataFrame(rows, columns=["stat", "value"])


def operator_signature(df: pd.DataFrame) -> pd.DataFrame:
    """Per-operator shift in each geometry covariate, in SD units.

    JOINT coefficients (all eight operators entered together), not marginal
    present-minus-absent differences. Operators co-occur, so a marginal
    difference for one of them absorbs the systematic absence of the others --
    and the correctness contrasts these are compared against are themselves
    joint, so a marginal geometry delta would not be like for like.

    All three covariates are standardised, so the deltas are directly
    comparable across the three panels of the syntax-geometry figure.
    """
    Hc = np.column_stack([np.ones(len(df)),
                          df[[f"has_{op}" for op in OPERATORS]].to_numpy(np.float64)])
    beta = {col: np.linalg.lstsq(Hc, df[col].to_numpy(np.float64), rcond=None)[0]
            for col in GEOMETRY}
    return pd.DataFrame([{
        "operator": op,
        "prevalence": float(df[f"has_{op}"].mean()),
        **{f"delta_{col}": float(beta[col][j + 1]) for col in GEOMETRY},
    } for j, op in enumerate(OPERATORS)])


def depth_operator_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Operator prevalence within each depth cell.

    The mechanism behind the depth curve's operator standardisation: a depth-2
    target and a depth-5 target are not the same kind of formula with different
    nesting, they have nearly disjoint operator profiles. Every operator's
    prevalence swings by at least 20 points across the depth range.
    """
    g = df.groupby("depth")
    out = pd.DataFrame({"depth": sorted(df["depth"].unique())}).set_index("depth")
    out["n"] = g.size()
    for op in OPERATORS:
        out[f"has_{op}"] = g[f"has_{op}"].mean()
    return out.reset_index()


def shuffle_null(df: pd.DataFrame, shuffle_greedy: pd.DataFrame) -> pd.DataFrame:
    """Chance-level equivalence rate under shuffled embeddings.

    Falsification check on target guessability: a rate ~0, flat in variance,
    kills the concern that low-variance targets are mechanically easier to
    hit. Variance-patterned nonzero rates would flag exactly that confound.
    """
    m = df[["formula_id", "variance"]].merge(
        shuffle_greedy[["formula_id", "is_semantic_equivalent"]],
        on="formula_id", validate="one_to_one")
    m["variance_terc"] = _terciles(m["variance"])
    rows = [{"slice": "overall",
             "n": len(m),
             "equiv_rate": float(m["is_semantic_equivalent"].mean())}]
    for terc, sub in m.groupby("variance_terc", observed=False):
        rows.append({"slice": f"variance_{terc}",
                     "n": len(sub),
                     "equiv_rate": float(sub["is_semantic_equivalent"].mean())})
    return pd.DataFrame(rows)
