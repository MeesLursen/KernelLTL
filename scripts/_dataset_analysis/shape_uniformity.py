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

from .boltzmann_sampler import n_eq, sample_eq
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
      - ``n_formulas`` (per-depth count of formulas in the dataset)
      - ``shape_entropy_ratio`` (per-depth scalar, in [0, 1] or ``None``)
      - ``shape_rank`` (per-depth dict ``{"empirical": [...], "reference":
        [...]}``, each a list of ``(shape, p)`` sorted by descending ``p``).
        ``reference`` is a Boltzmann-sampled uniform draw at N = n_formulas
        (regime-appropriate baseline at any depth).
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
        "n_formulas": {d: int(sum(distributions[d].values())) for d in depths},
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

        # Rank: empirical from seen shapes only (no enumeration needed),
        # reference is a Boltzmann-sampled uniform draw at the same N so
        # the comparison is regime-correct at every depth.
        total = int(sum(dist.values()))
        if total > 0:
            emp_ranked = sorted(
                ((sh, c / total) for sh, c in dist.items()),
                key=lambda r: -r[1],
            )
            ref_counter: Counter = Counter()
            for _ in range(total):
                ref_counter[sample_eq(d, rng)] += 1
            ref_ranked = sorted(
                ((sh, c / total) for sh, c in ref_counter.items()),
                key=lambda r: -r[1],
            )
            out["shape_rank"][d] = {
                "empirical": emp_ranked,
                "reference": ref_ranked,
            }

    return out
