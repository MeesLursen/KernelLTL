"""Stages B and C of Experiment 2: correctness models on the geometry frame.

Ladder (logistic, outcome = greedy semantic equivalence):

  M0        correct ~ C(depth)                            baseline
  M1        correct ~ z_variance + u + C(depth)           PRIMARY (beta_u)
  contrast  M1 + low_faith                                descriptive gap for
            the least-faithful tail (marginally standardised); doubles as the
            sensitivity check on beta_u (compare across the two fits)
  M2        M1 + has_op (8)                               attribution stage;
            attenuation of beta_u = does syntax absorb the geometric effect

Plus the descriptive u-decile curve (depth-adjusted via marginal
standardisation) and the per-operator H2c table.

Inference convention: HC1 robust SEs are attached to the point fits for the
tables, but the REPORTED intervals are 95% percentile-bootstrap CIs in which
the entire pipeline -- variance-bin edges, binned means/SDs, studentisation,
the low-faithfulness quantile cut, and the model fit -- is recomputed inside
every resample.
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
CONTRAST_TERMS = ["z_variance", "u", "low_faith"]
M2_TERMS = ["z_variance", "u", *HAS_COLS]
LADDER = [("M0", []), ("M1", M1_TERMS), ("contrast", CONTRAST_TERMS),
          ("M2", M2_TERMS)]

ALPHA = 0.05


# ------------------------------ fitting ------------------------------------ #

def _design(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {"const": np.ones(len(df))}
    for t in terms:
        cols[t] = df[t].to_numpy(dtype=np.float64)
    for d in sorted(df["depth"].unique())[1:]:
        cols[f"depth_{d}"] = (df["depth"] == d).to_numpy(dtype=np.float64)
    return pd.DataFrame(cols, index=df.index)


def _fit(y: np.ndarray, X: pd.DataFrame, *, cov_type: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Binomial())
        return model.fit(cov_type=cov_type) if cov_type else model.fit()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def marginal_gap(params: pd.Series, X: pd.DataFrame, col: str = "low_faith") -> float:
    """Average predicted P(correct) with ``col`` forced to 1 minus forced to 0."""
    b = params.reindex(X.columns).to_numpy()
    X1, X0 = X.copy(), X.copy()
    X1[col], X0[col] = 1.0, 0.0
    return float(_sigmoid(X1.to_numpy() @ b).mean()
                 - _sigmoid(X0.to_numpy() @ b).mean())


# ------------------------------ the curve ----------------------------------- #

def curve_rates(df: pd.DataFrame, curve_bins: int) -> np.ndarray:
    """Depth-adjusted correctness per u quantile bin (marginal standardisation).

    Returns (curve_bins,) adjusted rates; NaN for bins a degenerate qcut drops.
    """
    bins = pd.qcut(df["u"], curve_bins, labels=False, duplicates="drop")
    y = df["correct"].to_numpy(dtype=np.float64)
    observed = sorted(int(b) for b in bins.dropna().unique())

    cols: dict[str, np.ndarray] = {"const": np.ones(len(df))}
    for b in observed[1:]:
        cols[f"bin_{b}"] = (bins == b).to_numpy(dtype=np.float64)
    for d in sorted(df["depth"].unique())[1:]:
        cols[f"depth_{d}"] = (df["depth"] == d).to_numpy(dtype=np.float64)
    X = pd.DataFrame(cols, index=df.index)
    beta = _fit(y, X).params.reindex(X.columns).to_numpy()

    rates = np.full(curve_bins, np.nan)
    Xc = X.copy()
    for b in observed:
        for bb in observed[1:]:
            Xc[f"bin_{bb}"] = 1.0 if bb == b else 0.0
        rates[b] = _sigmoid(Xc.to_numpy() @ beta).mean()
    return rates


def curve_descriptives(df: pd.DataFrame, curve_bins: int) -> pd.DataFrame:
    bins = pd.qcut(df["u"], curve_bins, labels=False, duplicates="drop")
    g = df.assign(bin=bins).groupby("bin")
    out = pd.DataFrame({
        "bin": sorted(int(b) for b in bins.dropna().unique())}).set_index("bin")
    out["n"] = g.size()
    out["mean_u"] = g["u"].mean()
    out["raw_rate"] = g["correct"].mean()
    out["mean_sem_dist"] = g["sem_dist"].mean()
    return out.reset_index()


# ---------------------------- point estimates ------------------------------- #

def point_ladder(dfc: pd.DataFrame) -> pd.DataFrame:
    """All four fits with HC1 SEs (point estimates; CIs come from the bootstrap)."""
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for name, terms in LADDER:
        res = _fit(y, _design(dfc, terms), cov_type="HC1")
        for term in res.params.index:
            rows.append({"model": name, "term": term,
                         "estimate": float(res.params[term]),
                         "hc1_se": float(res.bse[term])})
    return pd.DataFrame(rows)


def point_h2c(dfc: pd.DataFrame) -> pd.DataFrame:
    """Per-operator depth-adjusted log-odds of correctness (descriptive)."""
    y = dfc["correct"].to_numpy(dtype=np.float64)
    rows = []
    for op in OPERATORS:
        res = _fit(y, _design(dfc, [f"has_{op}"]), cov_type="HC1")
        rows.append({"operator": op,
                     "prevalence": float(dfc[f"has_{op}"].mean()),
                     "log_odds": float(res.params[f"has_{op}"]),
                     "hc1_se": float(res.bse[f"has_{op}"])})
    return pd.DataFrame(rows)


def point_contrast(dfc: pd.DataFrame) -> dict:
    y = dfc["correct"].to_numpy(dtype=np.float64)
    X1 = _design(dfc, M1_TERMS)
    Xc = _design(dfc, CONTRAST_TERMS)
    m1 = _fit(y, X1)
    ct = _fit(y, Xc)
    low = dfc["low_faith"] == 1.0
    return {
        "n_low": int(low.sum()), "n_rest": int((~low).sum()),
        "faith_cut": float(dfc.loc[low, "relational_faithfulness"].max())
        if low.any() else float("nan"),
        "raw_rate_low": float(dfc.loc[low, "correct"].mean()),
        "raw_rate_rest": float(dfc.loc[~low, "correct"].mean()),
        "mean_sem_dist_low": float(dfc.loc[low, "sem_dist"].mean()),
        "mean_sem_dist_rest": float(dfc.loc[~low, "sem_dist"].mean()),
        "adj_gap": marginal_gap(ct.params, Xc),
        "beta_u_m1": float(m1.params["u"]),
        "beta_u_contrast": float(ct.params["u"]),
    }


# ------------------------------ bootstrap ----------------------------------- #

BOOT_KEYS = [
    "M1_z_variance", "M1_u",
    "contrast_low_faith", "contrast_u", "contrast_gap",
    "M2_z_variance", "M2_u",
    "atten_delta", "atten_ratio",
]


def bootstrap(df_raw: pd.DataFrame, idx: np.ndarray, *, n_bins: int,
              faith_tail: float, curve_bins: int,
              log_every: int = 1000) -> dict:
    """Whole-pipeline percentile bootstrap over targets.

    Every resample re-derives the covariates (bin edges, binned means/SDs,
    studentisation, the low-faith cut) before refitting, so first-stage
    estimation uncertainty is inside the intervals.
    """
    B = idx.shape[0]
    store = {k: np.full(B, np.nan) for k in BOOT_KEYS}
    store["curve"] = np.full((B, curve_bins), np.nan)
    store["h2c"] = np.full((B, len(OPERATORS)), np.nan)
    failures = {k: 0 for k in ("ladder", "curve", "h2c")}

    for b in range(B):
        sub = df_raw.iloc[idx[b]].reset_index(drop=True)
        dfc = derive_covariates(sub, n_bins=n_bins, faith_tail=faith_tail)
        y = dfc["correct"].to_numpy(dtype=np.float64)
        try:
            m1 = _fit(y, _design(dfc, M1_TERMS)).params
            Xc = _design(dfc, CONTRAST_TERMS)
            ct = _fit(y, Xc).params
            m2 = _fit(y, _design(dfc, M2_TERMS)).params
            store["M1_z_variance"][b] = m1["z_variance"]
            store["M1_u"][b] = m1["u"]
            store["contrast_low_faith"][b] = ct["low_faith"]
            store["contrast_u"][b] = ct["u"]
            store["contrast_gap"][b] = marginal_gap(ct, Xc)
            store["M2_z_variance"][b] = m2["z_variance"]
            store["M2_u"][b] = m2["u"]
            store["atten_delta"][b] = m1["u"] - m2["u"]
            if abs(m1["u"]) > 1e-8:
                store["atten_ratio"][b] = 1.0 - m2["u"] / m1["u"]
        except Exception:
            failures["ladder"] += 1
        try:
            store["curve"][b] = curve_rates(dfc, curve_bins)
        except Exception:
            failures["curve"] += 1
        try:
            for j, op in enumerate(OPERATORS):
                store["h2c"][b, j] = _fit(
                    y, _design(dfc, [f"has_{op}"])).params[f"has_{op}"]
        except Exception:
            failures["h2c"] += 1
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
