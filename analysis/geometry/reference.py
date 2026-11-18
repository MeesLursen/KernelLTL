"""The reference differences of thesis Table reference_shares, for every choice of atoms.

Each instance is a pair of formulae, or a formula held against a constant:

  next         p vs X p                                   one per atom
  until        (p U q) vs q                               one per ordered pair p != q
  persistence  G(p_1 AND ... AND p_k) vs bottom          one per set of k atoms
  eventuality  F(p_1 OR ... OR p_k) vs top               one per set of k atoms
  changes      p holds initially and changes exactly c    one per atom, c = 1, 2
               times, vs bottom:
                 c = 1: p AND (p U G ~p)
                 c = 2: p AND (p U (~p AND (~p U G p)))

The share of the traces on which an instance disagrees is given per process of the
trace distribution: the closed forms printed in the table, and, where the table
simplifies a sum (until), the finite sum it was derived from, so that the two can be
checked against each other.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from formula_class import And, Atom, Eventually, Formula, Globally, Next, Not, Or, Until

KINDS = ("next", "until", "persistence", "eventuality", "changes")


@dataclass(frozen=True)
class Instance:
    kind: str
    param: int | None            # k for persistence / eventuality, c for changes
    atoms: tuple[int, ...]
    phi: Formula
    other: Formula | None        # the second formula, or None against a constant
    constant: str | None         # "bottom" / "top" when held against a constant


def _fold(atoms: tuple[int, ...], op) -> Formula:
    out: Formula = Atom(atoms[0])
    for a in atoms[1:]:
        out = op(out, Atom(a))
    return out


def changes_formula(p: Formula, c: int) -> Formula:
    if c == 1:
        return And(p, Until(p, Globally(Not(p))))
    if c == 2:
        return And(p, Until(p, And(Not(p), Until(Not(p), Globally(p)))))
    raise ValueError(f"changes formula defined for c = 1, 2, not {c}")


def instances(n_ap: int) -> list[Instance]:
    out: list[Instance] = []
    for i in range(n_ap):
        out.append(Instance("next", None, (i,), Atom(i), Next(Atom(i)), None))
    for i, j in itertools.permutations(range(n_ap), 2):
        out.append(Instance("until", None, (i, j), Until(Atom(i), Atom(j)), Atom(j), None))
    for k in range(1, n_ap + 1):
        for subset in itertools.combinations(range(n_ap), k):
            out.append(Instance("persistence", k, subset, Globally(_fold(subset, And)), None, "bottom"))
    for k in range(1, n_ap + 1):
        for subset in itertools.combinations(range(n_ap), k):
            out.append(Instance("eventuality", k, subset, Eventually(_fold(subset, Or)), None, "top"))
    for c in (1, 2):
        for i in range(n_ap):
            out.append(Instance("changes", c, (i,), changes_formula(Atom(i), c), None, "bottom"))
    return out


# --------------------------------------------------------------------------- #
# Shares under the two processes (thesis Table reference_shares)
# --------------------------------------------------------------------------- #

def share_iid(kind: str, param: int | None, T: int) -> float:
    """Closed form under the first process (i.i.d. Bernoulli(1/2) truth values)."""
    if kind == "next":
        return 0.5
    if kind == "until":
        return (1.0 - 4.0 ** (1 - T)) / 6.0
    if kind in ("persistence", "eventuality"):
        return 0.5 ** (param * T)
    if kind == "changes":
        return math.comb(T - 1, param) * 0.5 ** T
    raise ValueError(kind)


def share_markov(kind: str, param: int | None, T: int, p: float) -> float:
    """Closed form under the second process (uniform start, flips with probability p)."""
    if kind == "next":
        return p
    if kind == "until":
        return (1.0 - (1.0 - p) ** (2 * (T - 1))) / (4.0 * (2.0 - p))
    if kind in ("persistence", "eventuality"):
        return (0.5 * (1.0 - p) ** (T - 1)) ** param
    if kind == "changes":
        return 0.5 * math.comb(T - 1, param) * p ** param * (1.0 - p) ** (T - 1 - param)
    raise ValueError(kind)


def share_until_sum(T: int, p: float | None) -> float:
    """The finite sum behind the until row: q first holds at step j >= 1, p holds before.

    p = None gives the first process, a float the second.
    """
    if p is None:
        return sum((0.5 ** j * 0.5) * 0.5 ** j for j in range(1, T))
    return sum((0.5 * (1 - p) ** (j - 1) * p) * (0.5 * (1 - p) ** (j - 1)) for j in range(1, T))


def share_mixture(kind: str, param: int | None, T: int, p: float, lam: float) -> float:
    return (1.0 - lam) * share_iid(kind, param, T) + lam * share_markov(kind, param, T, p)
