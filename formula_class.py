from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import random

# ------------------------- AST classes -------------------------

@dataclass(frozen=True)
class Formula:
    def atoms(self) -> Set[str]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
    
    def eval(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Formula):
    name: tuple

    def atoms(self) -> Set[tuple]:
        return {self.name}

    def __str__(self) -> str:
        return self.name[0]
    



@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(!{paren(self.child)})"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"



# Temporal unary
@dataclass(frozen=True)
class Next(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(X {self.child})"


@dataclass(frozen=True)
class Finally(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(F {self.child})"


@dataclass(frozen=True)
class Eventually(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(E {self.child})"


# Temporal binary
@dataclass(frozen=True)
class Until(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"




# helper
def paren(f: Formula) -> str:
    s = str(f)
    if s.startswith('(') and s.endswith(')'):
        return s
    return f"({s})"


# ------------------------- Random formula generator -------------------------

# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'F', 'E']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS


def random_formula(p_leaf: float = 0.3,
                   max_depth: int = 6,
                   n_atoms: int = 5,
                   root_must_be_operator: bool = True,
                   rng: Optional[random.Random] = None) -> Formula:
    """Generate a random formula.

    - p_leaf: probability to create an atomic proposition at a *non-root* node.
    - max_depth: maximum recursion depth (root at depth 0). When depth >= max_depth, we force a leaf.
    - n_atoms: number of distinct atomic proposition names (p0..p{n_atoms-1}).
    - root_must_be_operator: per user spec, force the root to be an operator node.
    """
    rng = rng or random
    atoms = [(f"p{i}",i) for i in range(n_atoms)]

    def gen(depth: int, force_internal: bool) -> Formula:
        # If we're at max depth -> force leaf
        if depth >= max_depth:
            return Atom(rng.choice(atoms))

        # Decide leaf or internal
        if force_internal:
            make_leaf = False
        else:
            make_leaf = rng.random() < p_leaf

        if make_leaf:
            return Atom(rng.choice(atoms))

        # Otherwise pick an operator uniformly
        op = rng.choice(_ALL_OPS)
        if op in _UNARY_OPS:
            # unary
            child = gen(depth + 1, force_internal=False)
            if op == 'NOT':
                return Not(child)
            if op == 'X':
                return Next(child)
            if op == 'F':
                return Finally(child)
            if op == 'E':
                return Eventually(child)
        else:
            # binary
            left = gen(depth + 1, force_internal=False)
            right = gen(depth + 1, force_internal=False)
            if op == 'AND':
                return And(left, right)
            if op == 'OR':
                return Or(left, right)
            if op == 'IMPLIES':
                return Implies(left, right)
            if op == 'U':
                return Until(left, right)

        # fallback (should not happen)
        return Atom(rng.choice(atoms))

    return gen(0, force_internal=root_must_be_operator)

