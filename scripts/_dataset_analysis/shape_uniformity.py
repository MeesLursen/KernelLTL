"""Shape-uniformity diagnostics under the P1 reference.

For each depth ``d`` present in the dataset:

* Compute the empirical distribution over arity-typed topologies.
* Compute the **shape entropy ratio** ``H_d / log(N_eq(d))`` ∈ [0, 1].
* Boltzmann-sample ``mc_n`` topologies of depth exactly ``d`` (uniform-over-
  topologies reference); evaluate the same shape metrics on the sample;
  compute the **KS distance** between the dataset's distribution and the
  uniform reference for each metric.

Outputs a self-contained dict (serialised to ``shape_uniformity.json`` by
the orchestrator) and the raw arrays needed to render the plots.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable

from formula_class import Formula
from scipy.stats import ks_2samp

from .boltzmann_sampler import enumerate_eq, n_eq, sample_eq
from .metrics import (
    Topology,
    branching_ratio,
    formula_to_topology,
    leaf_depth_variance,
    longest_path_concentration,
    mean_branch_imbalance,
    n_nodes,
)


# Headline shape metrics computed on Topology — kept in lockstep with
# what plots.py expects.
SHAPE_METRICS: dict[str, callable] = {
    "n_nodes": n_nodes,
    "branching_ratio": branching_ratio,
    "mean_branch_imbalance": mean_branch_imbalance,
    "longest_path_concentration": longest_path_concentration,
    "leaf_depth_variance": leaf_depth_variance,
}


def _safe_metric(fn, topo: Topology) -> float | None:
    try:
        v = fn(topo)
    except Exception:
        return None
    return v if v is not None else None


def empirical_shape_distribution(
    formulas: Iterable[Formula],
) -> dict[int, Counter]:
    """Map ``depth -> Counter[Topology]`` for every formula in the dataset."""
    out: dict[int, Counter] = {}
    for f in formulas:
        topo = formula_to_topology(f)
        d = _topo_depth_cached(topo)
        out.setdefault(d, Counter())[topo] += 1
    return out


# Avoid recomputing depths inside `empirical_shape_distribution` for big
# datasets — depth is cheap but called per formula.
def _topo_depth_cached(topo: Topology) -> int:
    if topo.arity == 0:
        return 0
    return 1 + max(_topo_depth_cached(c) for c in topo.kids)


def shape_entropy_ratio(distribution: Counter, depth: int) -> float | None:
    """``H(empirical) / log(N_eq(depth))`` ∈ [0, 1]."""
    total = sum(distribution.values())
    if total == 0:
        return None
    n_total = n_eq(depth)
    if n_total <= 1:
        return 1.0  # only one shape possible — trivially uniform
    h = 0.0
    for c in distribution.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(n_total)


def empirical_metric_arrays(
    distribution: Counter,
) -> dict[str, list[float]]:
    """For each shape metric, the per-formula values in the dataset."""
    out: dict[str, list[float]] = {name: [] for name in SHAPE_METRICS}
    for topo, count in distribution.items():
        for name, fn in SHAPE_METRICS.items():
            v = _safe_metric(fn, topo)
            if v is None:
                continue
            out[name].extend([float(v)] * count)
    return out


def reference_metric_arrays(
    depth: int,
    mc_n: int,
    rng: random.Random,
) -> tuple[list[Topology], dict[str, list[float]]]:
    """Boltzmann-sample ``mc_n`` topologies at ``depth`` and evaluate metrics."""
    topos = [sample_eq(depth, rng) for _ in range(mc_n)]
    arrays: dict[str, list[float]] = {name: [] for name in SHAPE_METRICS}
    for topo in topos:
        for name, fn in SHAPE_METRICS.items():
            v = _safe_metric(fn, topo)
            if v is None:
                continue
            arrays[name].append(float(v))
    return topos, arrays


def reference_distribution(
    depth: int,
    mc_n: int,
    rng: random.Random,
) -> Counter:
    """Counter of topologies sampled uniformly at depth ``depth``.

    For ``depth ≤ 4`` we return an explicit per-shape ``mc_n / N_eq(depth)``-
    weighted reference (i.e. the exact uniform pmf rescaled to a sample
    size of ``mc_n``); otherwise we just sample.
    """
    if depth <= 4 and n_eq(depth) <= 100_000:
        # Return the exact uniform pmf as if mc_n samples had been drawn.
        per_shape = mc_n // n_eq(depth)
        c: Counter = Counter()
        for sh in enumerate_eq(depth):
            c[sh] = per_shape
        return c
    c = Counter()
    for _ in range(mc_n):
        c[sample_eq(depth, rng)] += 1
    return c


def ks_distance(empirical: list[float], reference: list[float]) -> tuple[float, float]:
    """``(D, p_value)`` from a two-sample KS test. ``(NaN, NaN)`` if either side empty."""
    if not empirical or not reference:
        return (float("nan"), float("nan"))
    res = ks_2samp(empirical, reference)
    return (float(res.statistic), float(res.pvalue))


def compute_shape_uniformity(
    formulas: Iterable[Formula],
    *,
    mc_n: int = 100_000,
    rng_seed: int = 0,
) -> dict:
    """Compute every shape-uniformity diagnostic the script needs.

    Returns a dict with keys:
      - ``depths`` (sorted list of depths discovered)
      - ``n_eq`` (per-depth count of all topologies of depth exactly d)
      - ``shape_entropy_ratio`` (per-depth scalar, in [0, 1] or ``None``)
      - ``shape_rank`` (per-depth list of (shape_repr, empirical_pmf,
        uniform_pmf), only for d ≤ 4)
      - ``ks_distances`` (dict[depth][metric] -> (D, p))
      - ``empirical_metric_arrays`` and ``reference_metric_arrays`` for
        downstream plots
    """
    rng = random.Random(rng_seed)
    distributions = empirical_shape_distribution(formulas)
    depths = sorted(distributions)

    out = {
        "depths": depths,
        "n_eq": {d: n_eq(d) for d in depths},
        "shape_entropy_ratio": {},
        "shape_rank": {},
        "ks_distances": {},
        "empirical_metric_arrays": {},
        "reference_metric_arrays": {},
    }

    for d in depths:
        dist = distributions[d]
        out["shape_entropy_ratio"][d] = shape_entropy_ratio(dist, d)

        emp_arr = empirical_metric_arrays(dist)
        out["empirical_metric_arrays"][d] = emp_arr

        _, ref_arr = reference_metric_arrays(d, mc_n, rng)
        out["reference_metric_arrays"][d] = ref_arr

        ks = {}
        for name in SHAPE_METRICS:
            ks[name] = ks_distance(emp_arr.get(name, []), ref_arr.get(name, []))
        out["ks_distances"][d] = ks

        if d <= 4 and n_eq(d) <= 100_000:
            shapes = enumerate_eq(d)
            total = sum(dist.values())
            uniform_p = 1.0 / n_eq(d)
            ranked = sorted(
                ((sh, dist.get(sh, 0) / total if total else 0.0, uniform_p)
                 for sh in shapes),
                key=lambda r: -r[1],
            )
            out["shape_rank"][d] = ranked

    return out
