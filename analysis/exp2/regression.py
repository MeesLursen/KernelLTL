"""Experiment 2: logistic regressions of recovery on syntax and on what the representation keeps.

The ladder (thesis eq. diag_ladder), with depth as cell means (one indicator per depth,
no intercept) and operators as presence indicators:

    M0  depth
    M1  M0 + operator presence
    M2  M1 + Var(phi)
    M3  M2 + mean ||d||^2 over the k neighbours + R^2_phi + pooled R^2_{phi phi'}

Outcome: greedy recovery. An invalid output counts as a failure; every covariate is a
property of the target (and of its validation neighbours), so invalid outputs need none.

Functional form. Each continuous covariate enters linearly, quadratically, cubically or
as decile indicators, chosen by the likelihood-ratio test against decile indicators
(grouped lack-of-fit test, ported from the earlier Exp 2 code): Var first, at M2 where it
is read, then the three M3 covariates jointly with Var fixed at its form. The most
parsimonious form not rejected at 5% against the decile model is selected; AIC is
reported beside it, and a 20-bin reference is reported for Var.

Readings are on the probability scale, by g-computation over the targets: curves at the
covariate's decile midpoints, the contrast between its 90th and 10th percentiles,
operator contrasts (present vs absent), depth curves, and at M3 the representation's
cost: the change in mean predicted recovery when each target's two retentions are
raised to their own ceilings kappa (a modified treatment policy, which never asks for a
retention the trace sample cannot produce).

Inference: HC1 standard errors with the point fits; reported intervals are percentile
bootstrap intervals over targets, with every target's quantities, the transforms and the
decile edges held fixed. Separation and non-convergence are detected per fit and counted.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

from analysis.common.bootstrap import index_matrix, percentile_ci, replicate
from analysis.common.formulas import OPERATOR_NAMES

COVARIATES =("var", "d2", "r2_phi", "r2_pair")
RETENTIONS = ("r2_phi", "r2_pair")
POLY = {"linear": 1, "quadratic": 2, "cubic": 3}
BINNED = {"deciles": 10, "fine": 20}
CURVE_Q = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)   # decile midpoints
CONTRAST_Q = (0.10, 0.90)
LOW_END_Q = (0.05, 0.25)
SPEC_ALPHA = 0.05

RUNGS = {"M0": (False, ()), "M1": (True, ()), "M2": (True, ("var",)),
         "M3": (True, ("var", "d2", "r2_phi", "r2_pair"))}
BLOCKS = {"D": (False, ()), "D+S": (True, ()), "D+G": (False, COVARIATES),
          "D+S+G": (True, COVARIATES)}

PREDICTIONS = (
    ("P1a", "M3_contrast_r2_phi_p90_minus_p10", +1, "R^2_phi raises recovery"),
    ("P1b", "M3_contrast_r2_pair_p90_minus_p10", +1, "the pooled pair retention raises recovery"),
    ("P2", "M3_contrast_d2_p90_minus_p10", +1, "a larger true distance raises recovery"),
    ("P3", "M3_minus_M2_var_low_end_rise", -1,
     "the low end of the variance curve is flatter at M3 than at M2"),
)


# --------------------------------------------------------------------------- #
# Data and design
# --------------------------------------------------------------------------- #

@dataclass
class Data:
    y: np.ndarray
    depth: np.ndarray
    ops: np.ndarray
    cov: dict[str, np.ndarray]
    ceil: dict[str, np.ndarray]
    depths: tuple[int, ...]           # depth cells present in the data
    op_names: tuple[str, ...]         # operators that vary across targets (others cannot be estimated)

    @classmethod
    def from_targets(cls, df: pd.DataFrame, k: int) -> "Data":
        present = [o for o in OPERATOR_NAMES if 0 < df[f"op_{o}"].sum() < len(df)]
        return cls(y=df["recovered"].to_numpy(np.float64),
                   depth=df["depth"].to_numpy(np.int64),
                   ops=df[[f"op_{o}" for o in present]].to_numpy(np.float64),
                   depths=tuple(int(d) for d in sorted(df["depth"].unique())),
                   op_names=tuple(present),
                   cov={"var": df["var"].to_numpy(np.float64),
                        "d2": df[f"mean_d2_k{k}"].to_numpy(np.float64),
                        "r2_phi": df["R2_phi"].to_numpy(np.float64),
                        "r2_pair": df[f"pooled_R2_k{k}"].to_numpy(np.float64)},
                   ceil={"r2_phi": df["kappa_phi"].to_numpy(np.float64),
                         "r2_pair": df[f"pooled_kappa_k{k}"].to_numpy(np.float64)})


@dataclass(frozen=True)
class Transform:
    form: str
    mean: float = 0.0
    sd: float = 1.0
    edges: tuple = ()

    @classmethod
    def fit(cls, x: np.ndarray, form: str) -> "Transform":
        if form in POLY:
            return cls(form, float(np.mean(x)), float(np.std(x)) or 1.0)
        q = np.quantile(x, np.linspace(0.0, 1.0, BINNED[form] + 1)[1:-1])
        return cls(form, edges=tuple(np.unique(q)))

    @property
    def n_params(self) -> int:
        return POLY[self.form] if self.form in POLY else len(self.edges)

    def columns(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = np.asarray(x, dtype=np.float64)
        if self.form in POLY:
            z = (x - self.mean) / self.sd
            deg = POLY[self.form]
            return np.column_stack([z ** p for p in range(1, deg + 1)]), ["z", "z^2", "z^3"][:deg]
        bins = np.searchsorted(np.asarray(self.edges), x, side="left")   # right-closed bins
        nb = len(self.edges) + 1
        if nb == 1:
            return np.empty((x.size, 0)), []
        return (np.column_stack([(bins == b).astype(np.float64) for b in range(1, nb)]),
                [f"bin{b}" for b in range(1, nb)])


def design(data: Data, spec: tuple, transforms: dict, *, depth: int | None = None,
           op: tuple[str, float] | None = None, cov: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Design matrix for all targets, optionally with depth, one operator or covariates set."""
    use_ops, covs = spec
    n = data.y.size
    dep = data.depth if depth is None else np.full(n, depth)
    cols = [(dep == d).astype(np.float64)[:, None] for d in data.depths]
    names = [f"depth_{d}" for d in data.depths]
    if use_ops:
        O = data.ops.copy()
        if op is not None:
            O[:, data.op_names.index(op[0])] = op[1]
        cols.append(O)
        names += [f"op_{o}" for o in data.op_names]
    for c in covs:
        x = data.cov[c]
        if cov is not None and c in cov:
            v = cov[c]
            x = np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, dtype=np.float64)
        block, bnames = transforms[c].columns(x)
        cols.append(block)
        names += [f"{c}:{b}" for b in bnames]
    return np.hstack(cols), names


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def fit_glm(y: np.ndarray, X: np.ndarray, *, cov_type: str | None = None):
    """Logistic GLM; returns (result or None, flagged), flagged on separation or non-convergence."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.GLM(y, X, family=sm.families.Binomial())
            res = model.fit(cov_type=cov_type) if cov_type else model.fit()
        except Exception:                                     # noqa: BLE001 -- counted, not raised
            return None, True
    kinds = {type(w.message).__name__ for w in caught}
    params = np.asarray(res.params)
    flagged = (not bool(getattr(res, "converged", True))
               or bool(kinds & {"PerfectSeparationWarning", "ConvergenceWarning"})
               or not np.all(np.isfinite(params))
               or float(np.max(np.abs(X @ params))) > 30.0)
    return res, flagged


def lr_test(llf_small: float, k_small: int, llf_big: float, k_big: int) -> tuple[float, int, float]:
    lr = 2.0 * (llf_big - llf_small)
    df = k_big - k_small
    return lr, df, float(stats.chi2.sf(lr, df)) if df > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Specification search
# --------------------------------------------------------------------------- #

def spec_search(data: Data) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict] = []

    def fitted(forms: dict[str, str], spec: tuple) -> tuple[float, int, float]:
        tr = {c: Transform.fit(data.cov[c], f) for c, f in forms.items()}
        X, _ = design(data, spec, tr)
        res, _ = fit_glm(data.y, X)
        if res is None:
            raise RuntimeError(f"specification search: the fit with forms {forms} failed")
        return float(res.llf), X.shape[1], float(res.aic)

    # Stage 1: Var at M2 (depth + operators + Var); nothing else in the model.
    stage1 = {f: fitted({"var": f}, RUNGS["M2"]) for f in ("linear", "quadratic", "cubic", "deciles", "fine")}
    for ref in ("deciles", "fine"):
        llf_r, k_r, _ = stage1[ref]
        for form, (llf, k, aic) in stage1.items():
            if k >= k_r:
                continue
            lr, df, p = lr_test(llf, k, llf_r, k_r)
            rows.append({"stage": "1_var_at_M2", "forms": f"var={form}", "reference": f"var={ref}",
                         "params": k, "lr": lr, "df": df, "p": p, "aic": aic})
    chosen_var = "deciles"
    for form in ("linear", "quadratic", "cubic"):
        llf, k, _ = stage1[form]
        if lr_test(llf, k, *stage1["deciles"][:2])[2] >= SPEC_ALPHA:
            chosen_var = form
            break

    # Stage 2: the three M3 covariates jointly, Var fixed at its Stage 1 form.
    cands = ("linear", "quadratic", "deciles")
    grid = {}
    for fd, fp, fq in itertools.product(cands, repeat=3):
        forms = {"var": chosen_var, "d2": fd, "r2_phi": fp, "r2_pair": fq}
        grid[(fd, fp, fq)] = fitted(forms, RUNGS["M3"])
    llf_ref, k_ref, _ = grid[("deciles", "deciles", "deciles")]
    for cell, (llf, k, aic) in grid.items():
        lr, df, p = lr_test(llf, k, llf_ref, k_ref)
        rows.append({"stage": "2_M3_grid", "forms": f"d2={cell[0]},r2_phi={cell[1]},r2_pair={cell[2]}",
                     "reference": "all deciles", "params": k, "lr": lr, "df": df, "p": p, "aic": aic})
    admissible = [(k, aic, cell) for cell, (llf, k, aic) in grid.items()
                  if cell == ("deciles", "deciles", "deciles")
                  or lr_test(llf, k, llf_ref, k_ref)[2] >= SPEC_ALPHA]
    chosen_cell = min(admissible)[2]
    chosen = {"var": chosen_var, "d2": chosen_cell[0], "r2_phi": chosen_cell[1], "r2_pair": chosen_cell[2]}

    # Stage 3: each M3 covariate at its reporting base, the others at their chosen forms.
    for pos, c in enumerate(("d2", "r2_phi", "r2_pair")):
        ref_cell = tuple("deciles" if i == pos else chosen_cell[i] for i in range(3))
        llf_r, k_r, _ = grid[ref_cell]
        for form in ("linear", "quadratic"):
            cell = tuple(form if i == pos else chosen_cell[i] for i in range(3))
            llf, k, aic = grid[cell]
            lr, df, p = lr_test(llf, k, llf_r, k_r)
            rows.append({"stage": "3_reporting", "forms": f"{c}={form}", "reference": f"{c}=deciles",
                         "params": k, "lr": lr, "df": df, "p": p, "aic": aic})
    table = pd.DataFrame(rows)
    table["selected"] = False
    table.loc[(table["stage"] == "2_M3_grid")
              & (table["forms"] == f"d2={chosen_cell[0]},r2_phi={chosen_cell[1]},r2_pair={chosen_cell[2]}"),
              "selected"] = True
    table.loc[(table["stage"] == "1_var_at_M2") & (table["forms"] == f"var={chosen_var}")
              & (table["reference"] == "var=deciles"), "selected"] = True
    return table, chosen


# --------------------------------------------------------------------------- #
# Models, scenarios and readings (g-computation)
# --------------------------------------------------------------------------- #

@dataclass
class Model:
    name: str
    spec: tuple
    X: np.ndarray
    names: list[str]
    scenarios: dict[str, np.ndarray]
    depths: tuple[int, ...]
    op_names: tuple[str, ...]


def quantile_values(data: Data) -> dict[str, dict[float, float]]:
    qs = sorted(set(CURVE_Q) | set(CONTRAST_Q) | set(LOW_END_Q))
    return {c: {q: float(np.quantile(data.cov[c], q)) for q in qs} for c in COVARIATES}


def build_model(data: Data, name: str, spec: tuple, transforms: dict, qvals: dict) -> Model:
    X, names = design(data, spec, transforms)
    sc: dict[str, np.ndarray] = {}
    for d in data.depths:
        sc[f"depth={d}"] = design(data, spec, transforms, depth=d)[0]
    if spec[0]:
        for o in data.op_names:
            sc[f"{o}=1"] = design(data, spec, transforms, op=(o, 1.0))[0]
            sc[f"{o}=0"] = design(data, spec, transforms, op=(o, 0.0))[0]
    for c in spec[1]:
        qs = set(CURVE_Q) | set(CONTRAST_Q) | (set(LOW_END_Q) if c == "var" else set())
        for q in sorted(qs):
            sc[f"{c}@{q:.2f}"] = design(data, spec, transforms, cov={c: qvals[c][q]})[0]
    if set(RETENTIONS) <= set(spec[1]):
        sc["ceiling"] = design(data, spec, transforms, cov={c: data.ceil[c] for c in RETENTIONS})[0]
        sc["full_retention"] = design(data, spec, transforms, cov={c: 1.0 for c in RETENTIONS})[0]
    return Model(name, spec, X, names, sc, data.depths, data.op_names)


def readings(model: Model, beta: np.ndarray, rows: np.ndarray, depth_rows: np.ndarray) -> dict:
    sc = model.scenarios

    def mean_p(X: np.ndarray) -> float:
        return float(expit(X[rows] @ beta).mean())

    base_p = expit(model.X[rows] @ beta)
    out = {"mean_predicted": float(base_p.mean())}
    for d in model.depths:
        out[f"depth_curve_{d}"] = mean_p(sc[f"depth={d}"])
    out["depth_spread_shallowest_minus_deepest"] = (out[f"depth_curve_{model.depths[0]}"]
                                                    - out[f"depth_curve_{model.depths[-1]}"])
    if model.spec[0]:
        for o in model.op_names:
            out[f"op_contrast_{o}"] = mean_p(sc[f"{o}=1"]) - mean_p(sc[f"{o}=0"])
    for c in model.spec[1]:
        for q in CURVE_Q:
            out[f"curve_{c}_{q:.2f}"] = mean_p(sc[f"{c}@{q:.2f}"])
        out[f"contrast_{c}_p90_minus_p10"] = mean_p(sc[f"{c}@0.90"]) - mean_p(sc[f"{c}@0.10"])
        if c == "var":
            out["var_low_end_rise"] = mean_p(sc["var@0.25"]) - mean_p(sc["var@0.05"])
    if "ceiling" in sc:
        diff = expit(sc["ceiling"][rows] @ beta) - base_p
        out["representation_cost"] = float(diff.mean())
        for d in model.depths:
            mask = depth_rows == d
            out[f"representation_cost_depth_{d}"] = float(diff[mask].mean()) if mask.any() else np.nan
        out["full_retention_diagnostic"] = mean_p(sc["full_retention"]) - out["mean_predicted"]
    return out


def all_values(models: dict[str, Model], data: Data, rows: np.ndarray) -> dict[str, float]:
    """Every bootstrapped number for one resample (rows), keyed; NaN where a fit failed."""
    vals: dict[str, float] = {}
    y, dep = data.y[rows], data.depth[rows]
    per: dict[str, dict] = {}
    for name, mdl in models.items():
        res, flagged = fit_glm(y, mdl.X[rows])
        vals[f"flag_{name}"] = float(flagged)
        if res is None:
            continue
        beta = np.asarray(res.params)
        vals.update({f"coef_{name}_{nm}": float(b) for nm, b in zip(mdl.names, beta)})
        per[name] = readings(mdl, beta, rows, dep)
        vals.update({f"{name}_{k}": v for k, v in per[name].items()})
    if "M2" in per and "M3" in per:
        vals["M3_minus_M2_var_low_end_rise"] = per["M3"]["var_low_end_rise"] - per["M2"]["var_low_end_rise"]
    if "M1" in per and "M3" in per:
        vals["M3_minus_M1_depth_spread"] = (per["M3"]["depth_spread_shallowest_minus_deepest"]
                                           - per["M1"]["depth_spread_shallowest_minus_deepest"])
    return vals


# --------------------------------------------------------------------------- #
# Checks of the specification
# --------------------------------------------------------------------------- #

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Concordance: P(a recovered target scores above a failed one)."""
    ranks = stats.rankdata(p)
    n1 = float(y.sum())
    n0 = float(y.size - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def adequacy(data: Data, transforms: dict) -> pd.DataFrame:
    rows: list[dict] = []
    aucs = {}
    for name, spec in BLOCKS.items():
        X, _ = design(data, spec, transforms)
        res, _ = fit_glm(data.y, X)
        aucs[name] = auc(data.y, expit(X @ np.asarray(res.params)))
        rows.append({"statistic": f"auc_{name}", "value": aucs[name]})
    rows += [
        {"statistic": "auc_syntax_given_depth", "value": aucs["D+S"] - aucs["D"]},
        {"statistic": "auc_syntax_unique", "value": aucs["D+S+G"] - aucs["D+G"]},
        {"statistic": "auc_geometry_given_depth", "value": aucs["D+G"] - aucs["D"]},
        {"statistic": "auc_geometry_unique", "value": aucs["D+S+G"] - aucs["D+S"]},
        {"statistic": "auc_shared", "value": aucs["D+S"] + aucs["D+G"] - aucs["D"] - aucs["D+S+G"]},
    ]
    for name, spec in RUNGS.items():
        X, names = design(data, spec, transforms)
        res, _ = fit_glm(data.y, X)
        eta = X @ np.asarray(res.params)
        link, _ = fit_glm(data.y, np.column_stack([np.ones_like(eta), eta, eta ** 2]))
        rows.append({"statistic": f"link_test_z_eta2_{name}",
                     "value": float(link.params[2] / link.bse[2])})
        dfb = np.abs(np.asarray(res.get_influence().dfbetas)).max(axis=0)
        rows += [{"statistic": f"max_abs_dfbeta_{name}_{nm}", "value": float(v)}
                 for nm, v in zip(names, dfb)]
    return pd.DataFrame(rows)


def occupancy(data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = (("var", "r2_phi"), ("var", "r2_pair"), ("r2_phi", "r2_pair"), ("d2", "r2_pair"), ("var", "d2"))
    cells, summary = [], []
    for a, b in pairs:
        ta, tb = Transform.fit(data.cov[a], "deciles"), Transform.fit(data.cov[b], "deciles")
        ba = np.searchsorted(np.asarray(ta.edges), data.cov[a], side="left")
        bb = np.searchsorted(np.asarray(tb.edges), data.cov[b], side="left")
        na, nb = len(ta.edges) + 1, len(tb.edges) + 1
        counts = np.zeros((na, nb), dtype=np.int64)
        np.add.at(counts, (ba, bb), 1)
        cells += [{"rows": a, "cols": b, "row_decile": i, "col_decile": j, "targets": int(counts[i, j])}
                  for i in range(na) for j in range(nb)]
        summary.append({"rows": a, "cols": b, "cells": na * nb, "empty_cells": int((counts == 0).sum()),
                        "cells_below_10": int((counts < 10).sum())})
    return pd.DataFrame(cells), pd.DataFrame(summary)


def positivity(data: Data, *, near: float = 0.05) -> pd.DataFrame:
    """Support for the ceiling counterfactual, per variance decile."""
    tv = Transform.fit(data.cov["var"], "deciles")
    bins = np.searchsorted(np.asarray(tv.edges), data.cov["var"], side="left")
    rows = []
    for b in range(len(tv.edges) + 1):
        mask = bins == b
        if not mask.any():
            continue
        row = {"var_decile": b, "targets": int(mask.sum())}
        for c in RETENTIONS:
            gap = data.ceil[c][mask] - data.cov[c][mask]
            row[f"{c}_mean_gap_to_ceiling"] = float(gap.mean())
            row[f"{c}_share_within_{near}_of_ceiling"] = float(np.mean(gap <= near))
            row[f"{c}_share_ceiling_above_observed_max"] = float(
                np.mean(data.ceil[c][mask] > data.cov[c][mask].max() + 1e-12))
        rows.append(row)
    return pd.DataFrame(rows)


def design_sparsity(data: Data, transforms: dict) -> pd.DataFrame:
    """Recovered and failed targets per binary design column of M3 (separation risk)."""
    X, names = design(data, RUNGS["M3"], transforms)
    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        if not np.all(np.isin(col, (0.0, 1.0))):
            continue
        on = col == 1.0
        rows.append({"column": nm, "targets": int(on.sum()), "recovered": int(data.y[on].sum()),
                     "failed": int((1 - data.y[on]).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(targets: pd.DataFrame, *, k: int, sensitivity_ks: list[int], b: int, seed: int,
        workers: int) -> dict[str, pd.DataFrame | dict]:
    data = Data.from_targets(targets, k)
    spec_table, forms = spec_search(data)
    transforms = {c: Transform.fit(data.cov[c], forms[c]) for c in COVARIATES}
    qvals = quantile_values(data)
    models = {name: build_model(data, name, spec, transforms, qvals) for name, spec in RUNGS.items()}

    n = data.y.size
    point = all_values(models, data, np.arange(n))
    keys = list(point)
    idx = index_matrix(n, b=b, seed=seed)

    def statistic(rows: np.ndarray) -> np.ndarray:
        vals = all_values(models, data, rows)
        return np.array([vals.get(key, np.nan) for key in keys], dtype=np.float64)

    reps = replicate(statistic, idx, workers=workers)
    lo, hi = percentile_ci(reps)
    failed = np.isnan(reps).sum(axis=0)
    table = pd.DataFrame({"key": keys, "estimate": [point[key] for key in keys],
                          "ci_lo": lo, "ci_hi": hi, "failed_resamples": failed})

    coef_rows = []
    for name, mdl in models.items():
        res, flagged = fit_glm(data.y, mdl.X, cov_type="HC1")
        if res is None:
            raise RuntimeError(f"the point fit of {name} failed; see regression_separation.csv")
        for nm, beta, se in zip(mdl.names, np.asarray(res.params), np.asarray(res.bse)):
            key = f"coef_{name}_{nm}"
            j = keys.index(key)
            coef_rows.append({"rung": name, "term": nm, "estimate": float(beta), "se_hc1": float(se),
                              "z_hc1": float(beta / se), "ci_lo": lo[j], "ci_hi": hi[j],
                              "flagged_in_point_fit": flagged})
    readings_table = table[~table["key"].str.startswith(("coef_", "flag_"))].reset_index(drop=True)

    # block test: both retentions out of M3
    X_full = models["M3"].X
    X_red, _ = design(data, (True, ("var", "d2")), transforms)
    res_f, _ = fit_glm(data.y, X_full)
    res_r, _ = fit_glm(data.y, X_red)
    lr, df, p = lr_test(float(res_r.llf), X_red.shape[1], float(res_f.llf), X_full.shape[1])
    block = pd.DataFrame([{"test": "R2_phi and pooled R2 jointly, M3 vs M3 without them",
                           "lr": lr, "df": df, "p": p}])

    pred_rows = []
    for pid, key, sign, text in PREDICTIONS:
        j = keys.index(key)
        est, l, h = point[key], lo[j], hi[j]
        verdict = ("supported" if (sign > 0 and l > 0) or (sign < 0 and h < 0)
                   else "contradicted" if (sign > 0 and h < 0) or (sign < 0 and l > 0)
                   else "inconclusive")
        pred_rows.append({"prediction": pid, "statement": text, "reading": key, "expected_sign": sign,
                          "estimate": est, "ci_lo": l, "ci_hi": h, "verdict": verdict})

    sep_rows = [{"rung": name, "flagged_in_point_fit": bool(point[f"flag_{name}"]),
                 "flagged_resamples": int(np.nansum(reps[:, keys.index(f"flag_{name}")])),
                 "resamples": b} for name in models]

    sens_rows = []
    for ks in sensitivity_ks:
        dk = Data.from_targets(targets, ks)
        tk = {c: Transform.fit(dk.cov[c], forms[c]) for c in COVARIATES}
        mk = build_model(dk, "M3", RUNGS["M3"], tk, quantile_values(dk))
        res, flagged = fit_glm(dk.y, mk.X, cov_type="HC1")
        rd = readings(mk, np.asarray(res.params), np.arange(n), dk.depth)
        Xr, _ = design(dk, (True, ("var", "d2")), tk)
        rr, _ = fit_glm(dk.y, Xr)
        lr_k, df_k, p_k = lr_test(float(rr.llf), Xr.shape[1], float(res.llf), mk.X.shape[1])
        for key in ("contrast_d2_p90_minus_p10", "contrast_r2_phi_p90_minus_p10",
                    "contrast_r2_pair_p90_minus_p10", "representation_cost"):
            sens_rows.append({"k": ks, "reading": key, "estimate": rd[key], "flagged": flagged})
        sens_rows.append({"k": ks, "reading": "block_lr_p", "estimate": p_k, "flagged": flagged})

    occ_cells, occ_summary = occupancy(data)
    return {
        "forms": forms,
        "spec_search": spec_table,
        "coefficients": pd.DataFrame(coef_rows),
        "readings": readings_table,
        "block_test": block,
        "predictions": pd.DataFrame(pred_rows),
        "adequacy": adequacy(data, transforms),
        "occupancy_cells": occ_cells,
        "occupancy_summary": occ_summary,
        "positivity": positivity(data),
        "separation": pd.DataFrame(sep_rows),
        "design_sparsity": design_sparsity(data, transforms),
        "sensitivity": pd.DataFrame(sens_rows),
    }
