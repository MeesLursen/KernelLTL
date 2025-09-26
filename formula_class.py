from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import random

# ------------------------- AST classes -------------------------

@dataclass(frozen=True)
class Formula:
    def atoms(self) -> Set[tuple]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
    
    def __eq__(self,other) -> bool:
        raise NotImplementedError
    
    def eval_trace(self, trace : np.ndarray) -> bool:
        raise NotImplementedError



@dataclass(frozen=True)
class Atom(Formula):
    name: tuple

    def atoms(self) -> Set[tuple]:
        return {self.name}

    def __str__(self) -> str:
        return self.name[0]
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name

    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        return trace[self.name[1], :]



@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(~{paren(self.child)})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Not) and self.child == other.child
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        return np.logical_not(self.child.eval_trace(trace))



@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, And) and ((self.left == other.left and self.right == other.right) or (self.left == other.right and self.right == other.left))
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_and(L, R)



@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Or) and ((self.left == other.left and self.right == other.right) or (self.left == other.right and self.right == other.left))
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_or(L, R)



@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Implies) and self.left == other.left and self.right == other.right
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_or(np.logical_not(L), R)



# Temporal binary
@dataclass(frozen=True)
class Until(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Until) and self.left == other.left and self.right == other.right
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        T = trace.shape[1]
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        out = np.zeros(T, dtype=bool)

        for t in range(T-1, -1, -1):
            if t == T-1:
                out[t] = R[t]
            else:
                out[t] = R[t] or (L[t] and out[t + 1])
        
        return out



# Temporal unary
@dataclass(frozen=True)
class Next(Formula):
    """
    A strong implementation of the Next operator for finite traces.
    """
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(X {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Next) and self.child == other.child
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        sub = self.child.eval_trace(trace)
        return np.append(sub[1:], False)



@dataclass(frozen=True)
class Eventually(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(E {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Eventually) and self.child == other.child
    
    def eval_trace(self, trace) -> np.ndarray:
        T = trace.shape[1]
        sub = self.child.eval_trace(trace)
        out = np.zeros(T, dtype=bool)

        for t in range(T-1, -1, -1):
            if t == T-1:
                out[t] = sub[t]
            else:
                out[t] = out[t+1] or sub[t]

        return out
    


@dataclass(frozen=True)
class Globally(Formula):
    child: Formula

    def atoms(self) -> Set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(G {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Globally) and self.child == other.child
    
    def eval_trace(self, trace) -> np.ndarray:
        T = trace.shape[1]
        sub = self.child.eval_trace(trace)
        out = np.zeros(T, dtype=bool)

        for t in range(T-1, -1, -1):
            if t == T-1:
                out[t] = sub[t]
            else:
                out[t] = out[t+1] and sub[t]

        return out



# helper
def paren(f: Formula) -> str:
    s = str(f)
    if s.startswith('(') and s.endswith(')'):
        return s
    return f"({s})"


# ------------------------- Random formula generator -------------------------

# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'E', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS


# TODO: Figure out if we want a force_tree argument or not!
def random_formula(p_leaf: float = 0.3,
                   max_depth: int = 6,
                   n_atoms: int = 5,
                   seed: Optional[int] = None) -> Formula:
    """Generate a random formula.

    - p_leaf: probability to create an atomic proposition at a *non-root* node.
    - max_depth: maximum recursion depth (root at depth 0). When depth >= max_depth, we force a leaf.
    - n_atoms: number of distinct atomic proposition names (p0..p{n_atoms-1}).
    """
    rng = np.random.default_rng(seed)
    atoms = [(f"p{i}",i) for i in range(n_atoms)]

    def gen(depth: int) -> Formula:
        # If we're at max depth -> force leaf
        if depth >= max_depth:
            return Atom(atoms[rng.integers(len(atoms))])

        make_leaf = rng.random() < p_leaf

        if make_leaf:
            return Atom(atoms[rng.integers(len(atoms))])

        # Otherwise pick an operator uniformly
        op = rng.choice(_ALL_OPS)
        if op in _UNARY_OPS:
            # unary
            child : Formula = gen(depth + 1)

            # Avoid redundant unary nesting (e.g., G(G φ))
            while (op == 'G' and isinstance(child, Globally)) or \
                  (op == 'NOT' and isinstance(child, Not)):
                child = gen(depth + 1)

            if op == 'NOT':
                return Not(child)
            if op == 'X':
                return Next(child)
            if op == 'E':
                return Eventually(child)
            if op == 'G':
                return Globally(child)
        else:
            # binary
            left : Formula = gen(depth + 1)
            right : Formula = gen(depth + 1)

            while left == right:
                right = gen(depth + 1)

            if op == 'AND':
                return And(left, right)
            if op == 'OR':
                return Or(left, right)
            if op == 'IMPLIES':
                return Implies(left, right)
            if op == 'U':
                return Until(left, right)

    return gen(0)