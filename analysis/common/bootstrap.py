"""Percentile bootstrap over targets (thesis Sec. "Statistical conventions").

Targets are resampled with replacement to the original sample size, the
statistic (a NaN-aware mean over per-target values) is recomputed, and the
2.5th / 97.5th percentiles of the resampled values form the 95% interval.
One index matrix is drawn per (run, slice) and reused across all metrics of
that slice, so intervals within a table are computed on identical resamples.

Conditional metrics (NaN-coded per-target values, e.g. distinct-correct over
solved targets) condition *within* each resample via the NaN-aware mean, so
the varying size of the conditioning set is propagated into the interval.

Statistics that are not means (regression readings, ratios of sums) go
through ``replicate``, which evaluates an arbitrary vector-valued statistic on
the same kind of index matrix, and ``percentile_ci``, which turns the
replicates into intervals with the same percentile convention.
"""

from __future__ import annotations

import math
import multiprocessing
import warnings
from typing import Callable

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


# --------------------------------------------------------------------------- #
# General statistics (added for the geometry analysis and Experiment 2)
# --------------------------------------------------------------------------- #

def percentile_ci(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5th / 97.5th percentiles per column of a (B, K) replicate array.

    NaN-aware: a resample whose statistic could not be evaluated is stored as
    NaN and dropped column by column.
    """
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.ndim == 1:
        reps = reps[:, None]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        lo, hi = np.nanpercentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=0)
    return lo, hi


_STATISTIC: Callable[[np.ndarray], np.ndarray] | None = None
_IDX: np.ndarray | None = None


def _replicate_chunk(bounds: tuple[int, int]) -> np.ndarray:
    start, stop = bounds
    return np.stack([np.asarray(_STATISTIC(_IDX[b]), dtype=np.float64)
                     for b in range(start, stop)])


def replicate(
    statistic: Callable[[np.ndarray], np.ndarray],
    idx: np.ndarray,
    *,
    workers: int = 1,
) -> np.ndarray:
    """Evaluate ``statistic(rows)`` on every resample (row) of ``idx``.

    ``statistic`` maps one resample, an index array into the targets, to a 1-D
    vector of fixed length K, with NaN where it cannot be evaluated. Returns a
    (B, K) array. With ``workers > 1`` the resamples are split into contiguous
    chunks over a fork-based process pool; the result does not depend on the
    number of workers, because each replicate depends only on its own row.
    """
    global _STATISTIC, _IDX
    b = idx.shape[0]
    if workers <= 1 or b < 2:
        return np.stack([np.asarray(statistic(idx[i]), dtype=np.float64) for i in range(b)])
    _STATISTIC, _IDX = statistic, idx
    try:
        size = max(1, math.ceil(b / (4 * workers)))
        bounds = [(s, min(b, s + size)) for s in range(0, b, size)]
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_replicate_chunk, bounds)
    finally:
        _STATISTIC = _IDX = None
    return np.concatenate(parts, axis=0)
