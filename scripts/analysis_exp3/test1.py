"""Test 1 -- observational: signed variance error against the norm residual u.

Setting. Within a variance bin, targets differ in embedding norm only through
how well the anchors register them; that residual, studentised within the bin,
is Experiment 2's ``u``. If the decoder reads the norm as a variance signal it
should *over*-estimate the variance of targets that are louder than their
variance-matched peers and *under*-estimate the quieter ones:

    delta_phi := V_gen - V_target   increases with u   within every bin.

Competing signature. A decoder that treats a faint norm as "do not trust the
conditioning" and falls back toward a typical output moves low-u targets
toward that output's variance from both sides: delta rises with u in bins
above the crossover and *falls* with u in bins below it. The by-range table
and the interaction fit (delta ~ u * (b0 + b1 V)) are there to read the sign
pattern; norm-as-variance predicts b1 = 0 and b0 > 0.

Unadjusted and adjusted. Every slope is reported twice: as is, and with the
Experiment 2 adjustment set partialled out -- target depth (categorical) and
the eight operator-presence indicators -- alongside the bin fixed effects.
Adjustment is done once on the pooled sample (Frisch-Waugh-Lovell: residualise
delta and u on [bin dummies, depth, operators], then regress), so per-bin
adjusted slopes remain estimable with ~45 targets per bin.

Every estimate carries a percentile-bootstrap CI over targets, with u, the
bins and the adjustment re-derived on every resample exactly as in
Experiment 2. Two subsets: ``misses`` (non-equivalent valid generations; hits
have delta = 0 by construction and only dilute) and ``valid`` (all valid
generations, a sign test that keeps the hits as zeros).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "analysis_exp2"))
from frame import DEFAULT_N_BINS, OPERATORS, derive_covariates, tokenize   # noqa: E402

ALPHA = 0.05
MIN_BIN_N = 8          # a bin with fewer members in a subset contributes no slope
N_RANGES = 4           # target-variance quartiles for the by-range table

SUBSETS = ("misses", "valid")
ARMS = ("", "_adj")    # column suffixes: unadjusted / adjusted


def subset_mask(df: pd.DataFrame, subset: str) -> np.ndarray:
    valid = ~df["is_invalid"].to_numpy(dtype=bool)
    if subset == "valid":
        return valid
    if subset == "misses":
        return valid & ~df["is_semantic_equivalent"].to_numpy(dtype=bool)
    raise ValueError(subset)


# ------------------------------ adjustment --------------------------------- #

def adjustment_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Depth dummies (first level dropped) + the eight operator indicators."""
    depth = df["target_depth"].to_numpy(dtype=int)
    levels = sorted(np.unique(depth))
    cols, names = [], []
    for lv in levels[1:]:
        cols.append((depth == lv).astype(np.float64)); names.append(f"depth_{lv}")
    toks = [tokenize(s) for s in df["target_formula_str"]]
    for op in OPERATORS:
        cols.append(np.fromiter((float(op in t) for t in toks), dtype=np.float64, count=len(toks)))
        names.append(f"has_{op}")
    return np.column_stack(cols), names


def _bin_dummies(vbin: np.ndarray, n_groups: int) -> np.ndarray:
    D = np.zeros((len(vbin), n_groups))
    D[np.arange(len(vbin)), vbin] = 1.0
    return D


def _residualise(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Y minus its least-squares projection on the columns of Z (Y may be 2-D)."""
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ beta


# ------------------------------ estimators --------------------------------- #

def _grouped_slopes(vbin: np.ndarray, u: np.ndarray, y: np.ndarray, *, n_groups: int,
                    min_n: int = MIN_BIN_N) -> tuple[np.ndarray, np.ndarray]:
    """OLS slope of y on u within each group; NaN where the group is too small or u is flat."""
    n = np.bincount(vbin, minlength=n_groups).astype(np.float64)
    su = np.bincount(vbin, weights=u, minlength=n_groups)
    sy = np.bincount(vbin, weights=y, minlength=n_groups)
    suu = np.bincount(vbin, weights=u * u, minlength=n_groups)
    suy = np.bincount(vbin, weights=u * y, minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu, my = su / n, sy / n
        var_u = suu / n - mu * mu
        cov = suy / n - mu * my
        slope = cov / var_u
    slope[(n < min_n) | ~(var_u > 0)] = np.nan
    return slope, n


def _pooled_slope(uc: np.ndarray, yc: np.ndarray) -> float:
    denom = float(np.dot(uc, uc))
    return float(np.dot(uc, yc) / denom) if denom > 0 else np.nan


def _interaction(uc: np.ndarray, yc: np.ndarray, v: np.ndarray, Z: np.ndarray) -> tuple[float, float, float]:
    """delta ~ u*(b0 + b1 V) with Z partialled out of all three regressors. Returns (b0, b1, crossover)."""
    X = _residualise(np.column_stack([uc, uc * v]), Z)
    b, *_ = np.linalg.lstsq(X, yc, rcond=None)
    b0, b1 = float(b[0]), float(b[1])
    cross = -b0 / b1 if b1 != 0 else np.nan
    return b0, b1, cross


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _derive(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, int]:
    cov = derive_covariates(df, n_bins=n_bins)
    vbin = cov["vbin"].to_numpy(dtype=np.int64)
    return cov["u"].to_numpy(dtype=np.float64), vbin, int(vbin.max()) + 1


def _range_of(variance: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the target-variance range each row falls in (edges fixed on the full frame)."""
    return np.clip(np.searchsorted(edges, variance, side="right") - 1, 0, len(edges) - 2)


class _Estimates:
    """All Test 1 statistics for one (resample, subset), unadjusted and adjusted."""

    def __init__(self, vbin: np.ndarray, u: np.ndarray, delta: np.ndarray, variance: np.ndarray,
                 rng_idx: np.ndarray, X_adj: np.ndarray, n_groups: int, n_ranges: int):
        self.G, self.R = n_groups, n_ranges
        D = _bin_dummies(vbin, n_groups)
        self.per_bin, self.pooled, self.by_range, self.inter = {}, {}, {}, {}
        for arm, Z in (("", D), ("_adj", np.column_stack([D, X_adj]))):
            R = _residualise(np.column_stack([u, delta]), Z)
            uc, yc = R[:, 0], R[:, 1]
            self.per_bin[arm], _ = _grouped_slopes(vbin, uc, yc, n_groups=n_groups)
            self.pooled[arm] = _pooled_slope(uc, yc)
            self.by_range[arm] = np.array([
                _pooled_slope(uc[rng_idx == r], yc[rng_idx == r]) if (rng_idx == r).sum() >= MIN_BIN_N
                else np.nan for r in range(n_ranges)])
            self.inter[arm] = _interaction(uc, yc, variance, Z)


# --------------------------------- driver ---------------------------------- #

def run_test1(
    df: pd.DataFrame, *, idx: np.ndarray, n_bins: int = DEFAULT_N_BINS,
    v0: float | None = None, log=None,
) -> dict[str, pd.DataFrame]:
    """``df``: one row per target with the Experiment 2 feature columns
    (``emb_norm``, ``variance``, ``relational_faithfulness``), the greedy record
    columns (``target_formula_str``, ``target_depth``, ``is_invalid``,
    ``is_semantic_equivalent``) and ``delta_variance`` (NaN where invalid).
    ``idx``: (B, n) bootstrap index matrix over the rows of ``df``.
    """
    df = df.sort_values("formula_id").reset_index(drop=True)
    n, B = len(df), idx.shape[0]
    delta = df["delta_variance"].to_numpy(dtype=np.float64)
    variance = df["variance"].to_numpy(dtype=np.float64)
    X_adj_full, adj_names = adjustment_matrix(df)
    edges = np.quantile(variance, np.linspace(0, 1, N_RANGES + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rng_full = _range_of(variance, edges)
    masks = {s: subset_mask(df, s) for s in SUBSETS}

    # Point estimates on the full frame.
    u, vbin, G = _derive(df, n_bins)
    per_target = pd.DataFrame({
        "formula_id": df["formula_id"], "vbin": vbin, "u": u, "variance": variance,
        "variance_range": rng_full, "delta_variance": delta,
        "target_depth": df["target_depth"],
        "is_invalid": df["is_invalid"].to_numpy(dtype=bool),
        "is_semantic_equivalent": df["is_semantic_equivalent"].to_numpy(dtype=bool),
    })
    point = {s: _Estimates(vbin[m], u[m], delta[m], variance[m], rng_full[m], X_adj_full[m], G, N_RANGES)
             for s, m in masks.items()}
    bin_var = np.bincount(vbin, weights=variance, minlength=G) / np.maximum(np.bincount(vbin, minlength=G), 1)

    # Whole-pipeline bootstrap: u, bins and the adjustment re-derived per resample.
    if log is not None:
        log(f"test1: bootstrapping {B} resamples over {n} targets "
            f"(adjustment: {', '.join(adj_names)})")
    base_cols = ["formula_id", "emb_norm", "variance", "relational_faithfulness"]
    boot = {s: {"per_bin": {a: np.full((B, G), np.nan) for a in ARMS},
                "pooled": {a: np.full(B, np.nan) for a in ARMS},
                "by_range": {a: np.full((B, N_RANGES), np.nan) for a in ARMS},
                "inter": {a: np.full((B, 3), np.nan) for a in ARMS}} for s in SUBSETS}
    for b in range(B):
        rows = idx[b]
        ub, vb, Gb = _derive(df.loc[rows, base_cols].reset_index(drop=True), n_bins)
        db, vr, rb, Xb = delta[rows], variance[rows], rng_full[rows], X_adj_full[rows]
        for s in SUBSETS:
            mb = masks[s][rows]
            e = _Estimates(vb[mb], ub[mb], db[mb], vr[mb], rb[mb], Xb[mb], Gb, N_RANGES)
            for a in ARMS:
                k = min(G, Gb)
                boot[s]["per_bin"][a][b, :k] = e.per_bin[a][:k]
                boot[s]["pooled"][a][b] = e.pooled[a]
                boot[s]["by_range"][a][b] = e.by_range[a]
                boot[s]["inter"][a][b] = e.inter[a]
        if log is not None and (b + 1) % 1000 == 0:
            log(f"  {b + 1}/{B}")

    lo_q, hi_q = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)

    def ci(arr: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            return np.nanpercentile(arr, lo_q, axis=axis), np.nanpercentile(arr, hi_q, axis=axis)

    per_bin_rows, pooled_rows, range_rows = [], [], []
    for s in SUBSETS:
        m = masks[s]
        e = point[s]
        cnt = np.bincount(vbin[m], minlength=G)
        sp = np.full(G, np.nan)
        for g in range(G):
            sel = m & (vbin == g)
            if sel.sum() >= MIN_BIN_N:
                sp[g] = _spearman(u[sel], delta[sel])
        for g in range(G):
            row = {"subset": s, "vbin": g, "n": int(cnt[g]), "mean_variance": float(bin_var[g]),
                   "spearman": float(sp[g])}
            for a in ARMS:
                lo, hi = ci(boot[s]["per_bin"][a])
                row.update({f"slope{a}": float(e.per_bin[a][g]),
                            f"slope{a}_lo": float(lo[g]), f"slope{a}_hi": float(hi[g])})
            per_bin_rows.append(row)

        row = {"subset": s, "n": int(m.sum()), "n_bins": int(np.isfinite(e.per_bin[""]).sum())}
        for a in ARMS:
            lo, hi = ci(boot[s]["pooled"][a])
            row.update({f"slope{a}": e.pooled[a], f"slope{a}_lo": float(lo), f"slope{a}_hi": float(hi)})
            fin = np.isfinite(e.per_bin[a])
            row[f"n_bins_positive{a}"] = int((e.per_bin[a][fin] > 0).sum())
            b0, b1, cross = e.inter[a]
            ilo, ihi = ci(boot[s]["inter"][a])
            row.update({f"b_u{a}": b0, f"b_u{a}_lo": float(ilo[0]), f"b_u{a}_hi": float(ihi[0]),
                        f"b_uV{a}": b1, f"b_uV{a}_lo": float(ilo[1]), f"b_uV{a}_hi": float(ihi[1]),
                        f"crossover{a}": cross, f"crossover{a}_lo": float(ilo[2]),
                        f"crossover{a}_hi": float(ihi[2])})
            if v0 is not None and np.isfinite(v0):
                below = fin & (bin_var < v0)
                above = fin & (bin_var >= v0)
                row.update({"v0": float(v0),
                            f"n_bins_below_v0_positive{a}": int((e.per_bin[a][below] > 0).sum()),
                            f"n_bins_above_v0_positive{a}": int((e.per_bin[a][above] > 0).sum()),
                            "n_bins_below_v0": int(below.sum()), "n_bins_above_v0": int(above.sum())})
        pooled_rows.append(row)

        for r in range(N_RANGES):
            sel = m & (rng_full == r)
            row = {"subset": s, "range": r, "v_lo": float(edges[r]), "v_hi": float(edges[r + 1]),
                   "n": int(sel.sum()), "mean_variance": float(variance[sel].mean()) if sel.any() else np.nan}
            for a in ARMS:
                lo, hi = ci(boot[s]["by_range"][a])
                row.update({f"slope{a}": float(e.by_range[a][r]),
                            f"slope{a}_lo": float(lo[r]), f"slope{a}_hi": float(hi[r])})
            range_rows.append(row)

    return {"per_target": per_target, "per_bin": pd.DataFrame(per_bin_rows),
            "pooled": pd.DataFrame(pooled_rows), "by_range": pd.DataFrame(range_rows),
            "adjustment": adj_names}
