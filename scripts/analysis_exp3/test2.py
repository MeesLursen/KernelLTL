"""Test 2 -- interventional: the decoder's response to the embedding norm at fixed direction.

Input is the sweep of ``scripts/validation_variance_rescaling.py``: for every
target and every scale c the decoder was run greedily on ``c * emb(phi)``. Three
readings of the decoder are distinguished by their signatures:

                          V_gen vs c     direction cos    equivalence      entropy
  reads norm as variance  monotone up    high, flat       peaks at c = 1   falls in c
  reads norm as confidence  flat         high, flat       peaks at c = 1   falls in c
  direction only            flat         high, flat       flat             flat

Two further predictions of the mechanism (residual stream + layer norm): as
c -> 0 the output should converge on the zero-embedding output, and the
direction cosine should *drop* at small c (the injection is drowned by the
residual, direction and all).

Estimands, each with a percentile-bootstrap CI over targets:

  * per scale: validity, equivalence and mean semantic distance (the Experiment 1
    outcomes), mean generated variance and |2p-1|, mean signed variance error,
    mean direction cosine, mean output entropy, and the share of targets whose
    generated string equals their scale-0 / scale-1 string;
  * per target: Spearman rank correlation between c and V_gen over the valid
    positive scales; then the mean rho, and the shares of targets whose generated
    variance is increasing, decreasing, or *invariant* in c -- the last is the
    direction-only signature;
  * per target-variance tercile: the same per-scale curves, to see whether the
    response differs where the variance channel is faint.

All generations at scale 0 are the same fixed formula (Experiment 1's zero
ablation), so the monotonicity statistic is computed over c > 0 and the c = 0
point is reported as the prior V_0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "analysis_exp1"))
from bootstrap import mean_ci                                          # noqa: E402

MIN_VALID_SCALES = 3   # a target needs this many valid positive scales for a rank correlation


def _pivot(long: pd.DataFrame, col: str) -> pd.DataFrame:
    """(targets x scales) matrix of ``col``, targets sorted by formula_id."""
    return long.pivot(index="formula_id", columns="scale", values=col).sort_index()


def _ci_row(values: np.ndarray, idx: np.ndarray, prefix: str) -> dict:
    est, lo, hi, n_eff = mean_ci(values, idx)
    return {prefix: est, f"{prefix}_lo": lo, f"{prefix}_hi": hi, f"{prefix}_n": n_eff}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_VALID_SCALES:
        return np.nan
    if np.ptp(y) == 0:
        return 0.0          # generated variance identical across scales: invariant
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def per_scale_table(long: pd.DataFrame, *, idx: np.ndarray) -> pd.DataFrame:
    scales = sorted(long["scale"].unique())
    valid = ~_pivot(long, "is_invalid").astype(bool)
    equiv = _pivot(long, "is_semantic_equivalent").astype(float)
    dist = _pivot(long, "semantic_distance").astype(float)
    gvar = _pivot(long, "generated_variance").astype(float)
    gp = _pivot(long, "generated_base_rate").astype(float)
    tvar = _pivot(long, "target_variance").astype(float)
    cosd = _pivot(long, "direction_cosine").astype(float)
    ent = _pivot(long, "mean_token_entropy").astype(float)
    strs = _pivot(long, "generated_formula_str")

    ref0 = strs[0.0] if 0.0 in strs.columns else None
    ref1 = strs[1.0] if 1.0 in strs.columns else None

    rows = []
    for c in scales:
        v = valid[c].to_numpy()
        row = {"scale": c, "n_targets": int(len(v))}
        row.update(_ci_row(v.astype(float), idx, "validity"))
        row.update(_ci_row(equiv[c].to_numpy(), idx, "equivalence"))
        row.update(_ci_row(dist[c].to_numpy(), idx, "semantic_distance"))
        row.update(_ci_row(gvar[c].to_numpy(), idx, "generated_variance"))
        row.update(_ci_row(np.abs(2.0 * gp[c].to_numpy() - 1.0), idx, "generated_abs_m"))
        row.update(_ci_row((gvar[c] - tvar[c]).to_numpy(), idx, "delta_variance"))
        row.update(_ci_row(cosd[c].to_numpy(), idx, "direction_cosine"))
        row.update(_ci_row(ent[c].to_numpy(), idx, "entropy"))
        row.update(_ci_row(tvar[c].to_numpy(), idx, "target_variance"))
        if ref0 is not None:
            row.update(_ci_row((strs[c] == ref0).to_numpy(dtype=float), idx, "same_as_scale0"))
        if ref1 is not None:
            row.update(_ci_row((strs[c] == ref1).to_numpy(dtype=float), idx, "same_as_scale1"))
        rows.append(row)
    return pd.DataFrame(rows)


def per_target_monotonicity(long: pd.DataFrame) -> pd.DataFrame:
    gvar = _pivot(long, "generated_variance").astype(float)
    tvar = _pivot(long, "target_variance").astype(float).iloc[:, 0]
    pos = [c for c in gvar.columns if c > 0]
    out = []
    for fid, row in gvar.iterrows():
        y = row[pos].to_numpy(dtype=float)
        ok = np.isfinite(y)
        x = np.asarray(pos, dtype=float)[ok]
        yy = y[ok]
        rho = _spearman(x, yy)
        out.append({
            "formula_id": int(fid),
            "n_valid_positive_scales": int(ok.sum()),
            "rho_scale_variance": rho,
            # np.ptp, not np.std: the std of identical floats can round to ~1e-17.
            "invariant": bool(ok.sum() >= MIN_VALID_SCALES and np.ptp(yy) == 0),
            "target_variance": float(tvar.loc[fid]),
        })
    return pd.DataFrame(out).sort_values("formula_id").reset_index(drop=True)


def monotonicity_summary(per_target: pd.DataFrame, *, idx: np.ndarray) -> pd.DataFrame:
    rho = per_target["rho_scale_variance"].to_numpy(dtype=float)
    eligible = np.isfinite(rho)
    inc = np.where(eligible, (rho > 0).astype(float), np.nan)
    dec = np.where(eligible, (rho < 0).astype(float), np.nan)
    inv = np.where(eligible, per_target["invariant"].to_numpy(dtype=float), np.nan)
    row = {"n_targets": int(len(rho)), "n_eligible": int(eligible.sum())}
    row.update(_ci_row(rho, idx, "mean_rho"))
    row.update(_ci_row(inc, idx, "share_increasing"))
    row.update(_ci_row(dec, idx, "share_decreasing"))
    row.update(_ci_row(inv, idx, "share_invariant"))
    return pd.DataFrame([row])


def by_tercile_table(long: pd.DataFrame, *, idx: np.ndarray) -> pd.DataFrame:
    """Per (target-variance tercile, scale) curves; the bootstrap conditions within
    the tercile via NaN-coding, so the tercile's size varies across resamples."""
    tvar = _pivot(long, "target_variance").astype(float).iloc[:, 0]
    terc = pd.qcut(tvar, 3, labels=False)
    valid = ~_pivot(long, "is_invalid").astype(bool)
    equiv = _pivot(long, "is_semantic_equivalent").astype(float)
    gvar = _pivot(long, "generated_variance").astype(float)
    cosd = _pivot(long, "direction_cosine").astype(float)
    rows = []
    for t in (0, 1, 2):
        sel = (terc == t).to_numpy()
        for c in sorted(long["scale"].unique()):
            row = {"tercile": t, "scale": c, "n_targets": int(sel.sum()),
                   "tercile_mean_variance": float(tvar[sel].mean())}
            row.update(_ci_row(np.where(sel, valid[c].to_numpy(dtype=float), np.nan), idx, "validity"))
            row.update(_ci_row(np.where(sel, equiv[c].to_numpy(), np.nan), idx, "equivalence"))
            row.update(_ci_row(np.where(sel, gvar[c].to_numpy(), np.nan), idx, "generated_variance"))
            row.update(_ci_row(np.where(sel, cosd[c].to_numpy(), np.nan), idx, "direction_cosine"))
            rows.append(row)
    return pd.DataFrame(rows)


def run_test2(long: pd.DataFrame, *, idx: np.ndarray, log=None) -> dict[str, pd.DataFrame]:
    if log is not None:
        log(f"test2: {long['formula_id'].nunique()} targets x {long['scale'].nunique()} scales")
    per_scale = per_scale_table(long, idx=idx)
    per_target = per_target_monotonicity(long)
    summary = monotonicity_summary(per_target, idx=idx)
    terciles = by_tercile_table(long, idx=idx)
    return {"per_scale": per_scale, "per_target": per_target,
            "monotonicity": summary, "by_tercile": terciles}
