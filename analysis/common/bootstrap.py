"""Percentile bootstrap over targets (thesis Sec. "Statistical conventions").

Targets are resampled with replacement to the original sample size, the
statistic (a NaN-aware mean over per-target values) is recomputed, and the
2.5th / 97.5th percentiles of the resampled values form the 95% interval.
One index matrix is drawn per (run, slice) and reused across all metrics of
that slice, so intervals within a table are computed on identical resamples.

Conditional metrics (NaN-coded per-target values, e.g. distinct-correct over
solved targets) condition *within* each resample via the NaN-aware mean, so
the varying size of the conditioning set is propagated into the interval.
"""

from __future__ import annotations

import warnings

import numpy as np

DEFAULT_B = 10_000
DEFAULT_SEED = 0
ALPHA = 0.05


def index_matrix(n: int, *, b: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> np.ndarray:
    """(B, n) resample indices, PCG64-seeded."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(b, n), dtype=np.int64)


def mean_ci(
    values: np.ndarray,
    idx: np.ndarray,
    *,
    chunk: int = 1000,
) -> tuple[float, float, float, int]:
    """(estimate, ci_lo, ci_hi, n_effective) for the NaN-aware mean of ``values``.

    ``n_effective`` is the number of non-NaN per-target values entering the
    point estimate. Chunked over resamples to bound memory.
    """
    values = np.asarray(values, dtype=np.float64)
    n_eff = int(np.sum(~np.isnan(values)))
    if n_eff == 0:
        return float("nan"), float("nan"), float("nan"), 0
    est = float(np.nanmean(values))

    b = idx.shape[0]
    stats = np.empty(b, dtype=np.float64)
    # A resample of a conditional (NaN-coded) metric can contain zero members;
    # its statistic is NaN and nanpercentile below simply drops that resample.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for start in range(0, b, chunk):
            block = values[idx[start:start + chunk]]      # (chunk, n)
            stats[start:start + chunk] = np.nanmean(block, axis=1)
    lo, hi = np.nanpercentile(stats, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return est, float(lo), float(hi), n_eff
