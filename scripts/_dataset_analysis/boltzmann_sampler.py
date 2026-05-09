"""Uniform sampler over arity-typed tree topologies (the P1 reference).

Implements:

* Counts ``n_eq(d)`` = #topologies of depth exactly ``d``, ``n_le(d)`` =
  #topologies of depth ≤ ``d``, computed exactly via Python's
  arbitrary-precision ``int``.
* ``sample_eq(d, rng)`` / ``sample_le(d, rng)`` — uniform random draws
  using cumulative-integer comparisons (no float precision loss even at
  ``d = 5`` where counts approach ``10^9``).
* ``enumerate_eq(d)`` / ``enumerate_le(d)`` — exact enumeration of every
  topology, feasible for ``d ≤ 4`` (``n_le(4) = 33,673``).

Counts recursion (binary trees with arities ∈ {0, 1, 2}, depth defined as
longest root-to-leaf path):

    n_eq(0) = 1
    n_le(0) = 1
    n_eq(d) = n_eq(d-1)                               # unary root
            + n_eq(d-1)**2                            # binary root, both children depth d-1
            + 2 * n_eq(d-1) * n_le(d-2)               # binary root, one child depth d-1
    n_le(d) = n_le(d-1) + n_eq(d)
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Iterator

from .metrics import LEAF, Topology


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def n_eq(d: int) -> int:
    """Number of arity-typed topologies of depth exactly ``d``."""
    if d < 0:
        return 0
    if d == 0:
        return 1
    a = n_eq(d - 1)
    b = n_le(d - 2) if d >= 2 else 0
    return a + a * a + 2 * a * b


@lru_cache(maxsize=None)
def n_le(d: int) -> int:
    """Number of arity-typed topologies of depth ≤ ``d``."""
    if d < 0:
        return 0
    if d == 0:
        return 1
    return n_le(d - 1) + n_eq(d)


# ---------------------------------------------------------------------------
# Uniform samplers
# ---------------------------------------------------------------------------


def _weighted_pick(weights: list[int], rng: random.Random) -> int:
    """Pick an index ``i`` with probability ``weights[i] / sum(weights)``.

    Uses integer cumulative comparisons — no float arithmetic, so it stays
    exact even when individual weights overflow common float precision.
    """
    total = sum(weights)
    if total <= 0:
        raise ValueError("All weights are zero; nothing to pick from.")
    r = rng.randrange(total)
    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i
    raise RuntimeError("unreachable")  # pragma: no cover


def sample_le(d: int, rng: random.Random) -> Topology:
    """Sample uniformly from topologies of depth ≤ ``d``."""
    if d < 0:
        raise ValueError(f"d must be non-negative (got {d})")
    if d == 0:
        return LEAF

    n0 = 1                       # leaf at root
    n1 = n_le(d - 1)             # unary root, one subtree of depth ≤ d-1
    n2 = n_le(d - 1) ** 2        # binary root, two subtrees of depth ≤ d-1
    pick = _weighted_pick([n0, n1, n2], rng)

    if pick == 0:
        return LEAF
    if pick == 1:
        return Topology(1, (sample_le(d - 1, rng),))
    return Topology(
        2, (sample_le(d - 1, rng), sample_le(d - 1, rng))
    )


def sample_eq(d: int, rng: random.Random) -> Topology:
    """Sample uniformly from topologies of depth exactly ``d``."""
    if d < 0:
        raise ValueError(f"d must be non-negative (got {d})")
    if d == 0:
        return LEAF

    a = n_eq(d - 1)
    b = n_le(d - 2) if d >= 2 else 0

    w_unary = a                       # unary root + child at exact depth d-1
    w_binary_eqeq = a * a             # both children at exact depth d-1
    w_binary_eqle = a * b             # left exact, right ≤ d-2
    w_binary_leeq = a * b             # left ≤ d-2, right exact

    pick = _weighted_pick(
        [w_unary, w_binary_eqeq, w_binary_eqle, w_binary_leeq], rng
    )

    if pick == 0:
        return Topology(1, (sample_eq(d - 1, rng),))
    if pick == 1:
        return Topology(2, (sample_eq(d - 1, rng), sample_eq(d - 1, rng)))
    if pick == 2:
        return Topology(2, (sample_eq(d - 1, rng), sample_le(d - 2, rng)))
    return Topology(2, (sample_le(d - 2, rng), sample_eq(d - 1, rng)))


# ---------------------------------------------------------------------------
# Enumeration (exact, for verification + low-depth rank plots)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _enum_le_tuple(d: int) -> tuple[Topology, ...]:
    if d == 0:
        return (LEAF,)
    sub = _enum_le_tuple(d - 1)
    out: list[Topology] = [LEAF]
    for c in sub:
        out.append(Topology(1, (c,)))
    for left in sub:
        for right in sub:
            out.append(Topology(2, (left, right)))
    return tuple(out)


def enumerate_le(d: int) -> tuple[Topology, ...]:
    """Every topology of depth ≤ ``d``. Cached. Infeasible above d ≈ 4."""
    return _enum_le_tuple(d)


def enumerate_eq(d: int) -> tuple[Topology, ...]:
    """Every topology of depth exactly ``d``."""
    if d == 0:
        return (LEAF,)
    seen = enumerate_le(d - 1)
    seen_set = set(seen)
    return tuple(t for t in enumerate_le(d) if t not in seen_set)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def sample_eq_iter(d: int, n: int, rng: random.Random) -> Iterator[Topology]:
    for _ in range(n):
        yield sample_eq(d, rng)
