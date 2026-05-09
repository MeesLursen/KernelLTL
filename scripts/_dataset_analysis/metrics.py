"""Per-formula structural and operator metrics.

Two layers:

* ``Topology`` — an arity-typed tree (leaf / unary / binary), unlabeled.
  Used for both extracting the structural skeleton of a parsed
  ``Formula`` and as the output of the uniform-arity-topology Boltzmann
  sampler.
* ``Formula`` (from :mod:`formula_class`) — keeps operator and proposition
  labels. Used for the operator-frequency and proposition-count metrics.

Topology-level metrics work on either representation: they only need
arity + recursion through children. Both ``Formula`` AST nodes and
``Topology`` nodes expose enough structure to compute them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

from formula_class import (
    And,
    Atom,
    Eventually,
    Formula,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Until,
)


# ---------------------------------------------------------------------------
# Operator inventory
# ---------------------------------------------------------------------------

UNARY_OPERATORS = ("NOT", "X", "F", "G")
BINARY_OPERATORS = ("AND", "OR", "IMPLIES", "U")
ALL_OPERATORS = UNARY_OPERATORS + BINARY_OPERATORS

N_UNARY = len(UNARY_OPERATORS)
N_BINARY = len(BINARY_OPERATORS)

_FORMULA_CLASS_TO_OP = {
    Not: "NOT",
    Next: "X",
    Eventually: "F",
    Globally: "G",
    And: "AND",
    Or: "OR",
    Implies: "IMPLIES",
    Until: "U",
}


def operator_name(node: Formula) -> str | None:
    """Return the operator symbol for an internal ``Formula`` node, or ``None`` for an ``Atom``."""
    return _FORMULA_CLASS_TO_OP.get(type(node))


def is_atom(node: Formula) -> bool:
    return isinstance(node, Atom)


def is_unary(node: Formula) -> bool:
    return isinstance(node, (Not, Next, Eventually, Globally))


def is_binary(node: Formula) -> bool:
    return isinstance(node, (And, Or, Implies, Until))


def children(node: Formula) -> tuple[Formula, ...]:
    if is_atom(node):
        return ()
    if is_unary(node):
        return (node.child,)
    return (node.left, node.right)


def walk(node: Formula) -> Iterator[Formula]:
    yield node
    for c in children(node):
        yield from walk(c)


# ---------------------------------------------------------------------------
# Topology — arity-typed tree, unlabeled
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """Arity-typed tree node. ``arity`` ∈ {0, 1, 2}."""

    arity: int
    kids: tuple["Topology", ...] = ()

    def __post_init__(self) -> None:
        if self.arity not in (0, 1, 2):
            raise ValueError(f"arity must be 0, 1, or 2 (got {self.arity})")
        if len(self.kids) != self.arity:
            raise ValueError(
                f"expected {self.arity} children, got {len(self.kids)}"
            )


LEAF = Topology(0, ())


def formula_to_topology(node: Formula) -> Topology:
    """Strip operator/proposition labels, keep arity-typed structure."""
    if is_atom(node):
        return LEAF
    if is_unary(node):
        return Topology(1, (formula_to_topology(node.child),))
    return Topology(
        2,
        (formula_to_topology(node.left), formula_to_topology(node.right)),
    )


def _topo_children(topo: Topology) -> tuple[Topology, ...]:
    return topo.kids


def _topo_walk(topo: Topology) -> Iterator[Topology]:
    yield topo
    for c in topo.kids:
        yield from _topo_walk(c)


# ---------------------------------------------------------------------------
# Topology metrics — work on Topology OR Formula (both expose arity)
# ---------------------------------------------------------------------------


def _arity(node) -> int:
    if isinstance(node, Topology):
        return node.arity
    if is_atom(node):
        return 0
    if is_unary(node):
        return 1
    return 2


def _kids(node) -> tuple:
    if isinstance(node, Topology):
        return node.kids
    return children(node)


def depth(node) -> int:
    if _arity(node) == 0:
        return 0
    return 1 + max(depth(c) for c in _kids(node))


def n_nodes(node) -> int:
    return 1 + sum(n_nodes(c) for c in _kids(node))


def n_unary_internal(node) -> int:
    count = 1 if _arity(node) == 1 else 0
    return count + sum(n_unary_internal(c) for c in _kids(node))


def n_binary_internal(node) -> int:
    count = 1 if _arity(node) == 2 else 0
    return count + sum(n_binary_internal(c) for c in _kids(node))


def n_leaves(node) -> int:
    if _arity(node) == 0:
        return 1
    return sum(n_leaves(c) for c in _kids(node))


def branching_ratio(node) -> float | None:
    """``n_binary / (n_unary + n_binary)``. ``None`` if the formula has no internal nodes."""
    nu = n_unary_internal(node)
    nb = n_binary_internal(node)
    total = nu + nb
    if total == 0:
        return None
    return nb / total


def mean_branch_imbalance(node) -> float | None:
    """Mean over all binary internal nodes of ``|d_L − d_R| / max(d_L, d_R)``.

    ``None`` if the formula has no binary nodes.
    """
    imbalances: list[float] = []
    for sub in (
        _topo_walk(node) if isinstance(node, Topology) else walk(node)
    ):
        if _arity(sub) != 2:
            continue
        left, right = _kids(sub)
        dl, dr = depth(left), depth(right)
        m = max(dl, dr)
        if m == 0:
            imbalances.append(0.0)
        else:
            imbalances.append(abs(dl - dr) / m)
    if not imbalances:
        return None
    return sum(imbalances) / len(imbalances)


def root_to_leaf_path_lengths(node) -> list[int]:
    """Lengths (in operators) of every root-to-leaf path."""
    if _arity(node) == 0:
        return [0]
    out: list[int] = []
    for c in _kids(node):
        out.extend(1 + length for length in root_to_leaf_path_lengths(c))
    return out


def leaf_depth_variance(node) -> float:
    """Population variance of root-to-leaf path lengths."""
    paths = root_to_leaf_path_lengths(node)
    if len(paths) <= 1:
        return 0.0
    mean = sum(paths) / len(paths)
    return sum((p - mean) ** 2 for p in paths) / len(paths)


def longest_path_concentration(node) -> float:
    """``(longest_path_length + 1) / n_nodes``.

    The +1 comes from counting the leaf at the end of the path. For a unary
    chain this is 1.0 (all nodes lie on the one root-to-leaf path); for a
    fully balanced binary tree this is small (~ ``(d+1) / (2^(d+1)−1)``).
    """
    longest = max(root_to_leaf_path_lengths(node))
    return (longest + 1) / n_nodes(node)


# ---------------------------------------------------------------------------
# Formula-only metrics (require operator/proposition labels)
# ---------------------------------------------------------------------------


def operator_counts(formula: Formula) -> Counter:
    """Counts of each operator symbol in the formula."""
    counts: Counter = Counter({op: 0 for op in ALL_OPERATORS})
    for sub in walk(formula):
        op = operator_name(sub)
        if op is not None:
            counts[op] += 1
    return counts


def n_unique_propositions(formula: Formula) -> int:
    return len(formula.atoms())


def n_proposition_occurrences(formula: Formula) -> int:
    return sum(1 for sub in walk(formula) if is_atom(sub))


def shape_metrics(formula: Formula) -> dict:
    """Bundle every per-formula shape metric we'll need downstream."""
    topo = formula_to_topology(formula)
    return {
        "depth": depth(topo),
        "n_nodes": n_nodes(topo),
        "n_leaves": n_leaves(topo),
        "n_unary_internal": n_unary_internal(topo),
        "n_binary_internal": n_binary_internal(topo),
        "branching_ratio": branching_ratio(topo),
        "mean_branch_imbalance": mean_branch_imbalance(topo),
        "longest_path_concentration": longest_path_concentration(topo),
        "leaf_depth_variance": leaf_depth_variance(topo),
        "topology": topo,
    }
