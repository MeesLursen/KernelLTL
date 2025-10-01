from __future__ import annotations
from dataclasses import dataclass
import torch


# ------------------------- formula classes -------------------------

@dataclass(frozen=True)
class Formula:
    def atoms(self) -> set[tuple]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
    
    def __eq__(self,other) -> bool:
        raise NotImplementedError
    
    def eval_trace(self, trace : torch.Tensor) -> bool:
        raise NotImplementedError



@dataclass(frozen=True)
class Atom(Formula):
    name: tuple

    def atoms(self) -> set[tuple]:
        return {self.name}

    def __str__(self) -> str:
        return self.name[0]
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name

    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        return trace[self.name[1], :]



@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(~{paren(self.child)})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Not) and self.child == other.child
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        return torch.logical_not(self.child.eval_trace(trace))



@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, And) and ((self.left == other.left and self.right == other.right) or (self.left == other.right and self.right == other.left))
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return torch.logical_and(L, R)



@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Or) and ((self.left == other.left and self.right == other.right) or (self.left == other.right and self.right == other.left))
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return torch.logical_or(L, R)



@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Implies) and self.left == other.left and self.right == other.right
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return torch.logical_or(torch.logical_not(L), R)



# Temporal unary
@dataclass(frozen=True)
class Next(Formula):
    """
    A strong implementation of the Next operator for finite traces.
    """
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(X {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Next) and self.child == other.child
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        sub = self.child.eval_trace(trace)
        out = torch.zeros_like(sub)
        out[:-1] = sub[1:]
        return out



@dataclass(frozen=True)
class Eventually(Formula):
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(F {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Eventually) and self.child == other.child
    
    def eval_trace(self, trace) -> torch.Tensor:
        T = trace.size(dim=2)
        sub = self.child.eval_trace(trace)
        out = torch.zeros(T)

        out[-1] = sub[-1]
        for t in range(T-2, -1, -1):
            out[t] = out[t+1] or sub[t]

        return out
    


@dataclass(frozen=True)
class Globally(Formula):
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(G {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Globally) and self.child == other.child
    
    def eval_trace(self, trace) -> torch.Tensor:
        T = trace.shape[1]
        sub = self.child.eval_trace(trace)
        out = torch.zeros(T)

        out[-1] = sub[-1]
        for t in range(T-2, -1, -1):
            out[t] = out[t+1] and sub[t]

        return out



# Temporal binary
@dataclass(frozen=True)
class Until(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> set[tuple]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Until) and self.left == other.left and self.right == other.right
    
    def eval_trace(self, trace : torch.Tensor) -> torch.Tensor:
        T = trace.shape[1]
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        out = torch.zeros(T)
        
        out[-1] = R[-1]
        for t in range(T-2, -1, -1):
            out[t] = R[t] or (L[t] and out[t + 1])
        
        return out



# helper
def paren(f: Formula) -> str:
    s = str(f)
    if s.startswith('(') and s.endswith(')'):
        return s
    return f"({s})"



# ------------------------- vectorized batch evaluator -------------------------
def eval_traces_batch_torch(formula: Formula, traces_batch: torch.Tensor) -> torch.Tensor:
    """
    Evaluate formula on a batch of traces.

    - formula: AST formula (Atom/Not/And/Or/Next/Eventually/Globally/Until)
    - traces_batch: torch.Tensor, shape (batch_size=B, n_ap, T), dtype=torch.uint8,
    Returns:
    - out: torch.Tensor, shape (B, T), dtype=torch.uint8,
            out[i, t] == True iff trace i suffix at time t satisfies formula.
    """
    T = traces_batch.shape[2]

    # atoms
    if isinstance(formula, Atom):
        idx = formula.name[1]
        return traces_batch[:, idx, :]

    # boolean connectives
    if isinstance(formula, Not):
        child = eval_traces_batch_torch(formula.child, traces_batch)  # (B,T)
        return torch.logical_not(child)

    if isinstance(formula, And):
        L = eval_traces_batch_torch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch_torch(formula.right, traces_batch)  # (B,T)
        return torch.logical_and(L, R)

    if isinstance(formula, Or):
        L = eval_traces_batch_torch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch_torch(formula.right, traces_batch)  # (B,T)
        return torch.logical_or(L, R)
    
    if isinstance(formula, Implies):
        L = eval_traces_batch_torch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch_torch(formula.right, traces_batch)  # (B,T)
        return torch.logical_or(torch.logical_not(L), R)


    # temporal-unary
    if isinstance(formula, Next):
        child = eval_traces_batch_torch(formula.child, traces_batch)  # (B,T)
        out = torch.zeros_like(child)  # (B,T)
        if T >= 2:
            out[:, :-1] = child[:, 1:]
        return out

    if isinstance(formula, Eventually):
        child = eval_traces_batch_torch(formula.child, traces_batch)  # (B,T)
        rev = torch.flip(child, [1])  # (B,T)
        cum = torch.cumprod(rev, dim=1)
        out = torch.flip(cum, [1])
        return out

    if isinstance(formula, Globally):
        child = eval_traces_batch_torch(formula.child, traces_batch)  # (B,T)
        rev = torch.flip(child, [1])  # (B,T)
        cum = torch.cummax(rev, dim=1).values
        out = torch.flip(cum, [1])
        return out

    # temporal-2ary
    if isinstance(formula, Until):
        L = eval_traces_batch_torch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch_torch(formula.right, traces_batch)  # (B,T)
        out = torch.empty_like(R)  # (B,T)
        out[:, -1] = R[:, -1]
        for t in range(T-2, -1, -1):
            out[:, t] = torch.logical_or(R[:, t], torch.logical_and(L[:, t], out[:, t + 1]))
        return out

    raise NotImplementedError(f"Unknown formula type: {type(formula)}")



# ------------------------- random formula generator -------------------------
# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'F', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS


def sample_formulas_torch(n_formula: int,
                    p_leaf: float,
                    max_depth: int,
                    n_ap: int,
                    force_tree: bool,
                    rng: torch.Generator,
                    device: str) -> Formula:
    """Generate a random formula.
    - n_formula: Specifies the number of sampled formulae.
    - p_leaf: probability to create an atomic proposition at a *non-root* node.
    - max_depth: maximum recursion depth (root at depth 0). When depth >= max_depth, we force a leaf.
    - n_ap: maximum number of distinct atomic proposition names (p0..p{n_ap-1}).
    - force_tree: specifies whether the root is forced to be an operator.
    - rng: specifies the random number generator used, for reproducibility.
    Returns:
    - ls: a list of formulae
    """


    atoms = [(f"p{i}",i) for i in range(n_ap)]

    def gen(depth: int, root_must_be_operator: bool = False) -> Formula:
        # If we're at max depth -> force leaf
        if depth >= max_depth:
            return Atom(atoms[torch.randint(0, len(atoms), (), generator=rng, device=device).item()])
        
        if depth == 0 and root_must_be_operator:
            make_leaf = False
        else:
            make_leaf = rng.random() < p_leaf

        if make_leaf:
            return Atom(atoms[torch.randint(0, len(atoms), (), generator=rng, device=device).item()])

        # Otherwise pick an operator uniformly
        op = _ALL_OPS[torch.randint(0, len(atoms), (), generator=rng, device=device).item()]
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
            if op == 'F':
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

    ls = []
    for _ in range(n_formula):
        formula = gen(0, root_must_be_operator = force_tree)
        ls.append(formula)
    
    return ls



# ------------------------- random traces generator -------------------------
def sample_traces_torch(n_traces: int, n_ap:int, trace_length:int, rng: torch.Generator, device: str) -> torch.Tensor:
    """
    - n_traces: specifies the number of traces sampled uniformly at random (each trace is shape (n_ap, T), with values in {False,True}).
    - n_ap: specifies the number of atomic propositions in each trace.
    - trace_length: specifies the length of each of the sampled traces.
    - rng: specifies the random number generator used, for reproducibility.
    Returns:
    - traces: ndarray of shape (n_traces, n_ap, trace_length).
    """

    traces = torch.randint(0,2, size=(n_traces, n_ap, trace_length), generator=rng, dtype=torch.uint8, device = device)

    return traces