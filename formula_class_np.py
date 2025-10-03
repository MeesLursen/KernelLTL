from dataclasses import dataclass
import numpy as np


# ------------------------- formula classes -------------------------
@dataclass(frozen=True)
class Formula:
    def atoms(self) -> set[tuple]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
    
    def __eq__(self,other) -> bool:
        raise NotImplementedError
    
    def eval_trace(self, trace : np.ndarray) -> bool:
        raise NotImplementedError



@dataclass(frozen=True)
class Atom(Formula):
    name: int

    def atoms(self) -> set[int]:
        return {self.name}

    def __str__(self) -> str:
        return f'p_{self.name}'
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name

    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        return trace[self.name, :]



@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(~ {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Not) and self.child == other.child
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        return np.logical_not(self.child.eval_trace(trace))



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
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_and(L, R)



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
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_or(L, R)



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
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        return np.logical_or(np.logical_not(L), R)



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
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        sub = self.child.eval_trace(trace)
        return np.append(sub[1:], False)



@dataclass(frozen=True)
class Eventually(Formula):
    child: Formula

    def atoms(self) -> set[tuple]:
        return self.child.atoms()

    def __str__(self) -> str:
        return f"(F {self.child})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Eventually) and self.child == other.child
    
    def eval_trace(self, trace) -> np.ndarray:
        T = trace.shape[1]
        sub = self.child.eval_trace(trace)
        out = np.zeros(T, dtype=bool)

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
    
    def eval_trace(self, trace) -> np.ndarray:
        T = trace.shape[1]
        sub = self.child.eval_trace(trace)
        out = np.zeros(T, dtype=bool)

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
    
    def eval_trace(self, trace : np.ndarray) -> np.ndarray:
        T = trace.shape[1]
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        out = np.zeros(T, dtype=bool)
        
        out[-1] = R[-1]
        for t in range(T-2, -1, -1):
            out[t] = R[t] or (L[t] and out[t + 1])
        
        return out



# ------------------------- vectorized batch evaluator -------------------------
def eval_traces_batch_np(formula: Formula, traces_batch: np.ndarray) -> np.ndarray:
    """
    Evaluate formula on a batch of traces.

    - formula: AST formula (Atom/Not/And/Or/Next/Eventually/Globally/Until)
    - traces_batch: np.ndarray, shape (batch_size=B, n_ap, T), dtype=bool
    Returns:
    - out: np.ndarray, shape (B, T), dtype=bool,
            out[i, t] == True iff trace i suffix at time t satisfies formula.
    """
    _, _, T = traces_batch.shape

    # atoms
    if isinstance(formula, Atom):
        return traces_batch[:, formula.name, :]

    # boolean connectives
    if isinstance(formula, Not):
        child = eval_traces_batch_np(formula.child, traces_batch)
        return np.logical_not(child)

    if isinstance(formula, And):
        L = eval_traces_batch_np(formula.left, traces_batch)
        R = eval_traces_batch_np(formula.right, traces_batch)
        return np.logical_and(L, R)

    if isinstance(formula, Or):
        L = eval_traces_batch_np(formula.left, traces_batch)
        R = eval_traces_batch_np(formula.right, traces_batch)
        return np.logical_or(L, R)
    
    if isinstance(formula, Implies):
        L = eval_traces_batch_np(formula.left, traces_batch)
        R = eval_traces_batch_np(formula.right, traces_batch)
        return np.logical_or(np.logical_not(L), R)


    # temporal-unary
    if isinstance(formula, Next):
        child = eval_traces_batch_np(formula.child, traces_batch)  # (B,T)
        out = np.zeros_like(child)
        if T >= 2:
            out[:, :-1] = child[:, 1:]
        return out

    if isinstance(formula, Eventually):
        child = eval_traces_batch_np(formula.child, traces_batch)  # (B,T)
        rev = child[:, ::-1]
        cum = np.logical_or.accumulate(rev, axis=1)
        out = cum[:, ::-1]
        return out

    if isinstance(formula, Globally):
        child = eval_traces_batch_np(formula.child, traces_batch)  # (B,T)
        rev = child[:, ::-1]
        cum = np.logical_and.accumulate(rev, axis=1)
        out = cum[:, ::-1]
        return out

    # temporal-2ary
    if isinstance(formula, Until):
        L = eval_traces_batch_np(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch_np(formula.right, traces_batch)  # (B,T)
        out = np.empty_like(R)
        out[:, -1] = R[:, -1]
        for t in range(T-2, -1, -1):
            out[:, t] = np.logical_or(R[:, t], np.logical_and(L[:, t], out[:, t + 1]))
        return out

    raise NotImplementedError(f"Unknown formula type: {type(formula)}")



# ------------------------- random formula generator -------------------------
# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'F', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS


def sample_formulas_np(n_formula: int,
                    p_leaf: float,
                    max_depth: int,
                    n_ap: int,
                    force_tree: bool,
                    rng: np.random.Generator) -> Formula:
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

    atoms = list(range(n_ap))

    def gen(depth: int, root_must_be_operator: bool = False) -> Formula:
        # If we're at max depth -> force leaf
        if depth >= max_depth:
            return Atom(atoms[rng.integers(len(atoms))])
        
        if depth == 0 and root_must_be_operator:
            make_leaf = False
        else:
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
def sample_traces_np(n_traces: int, n_ap:int, trace_length:int, rng: np.random.Generator) -> np.ndarray:
    """
    - n_traces: specifies the number of traces sampled uniformly at random (each trace is shape (n_ap, T), with values in {False,True}).
    - n_ap: specifies the number of atomic propositions in each trace.
    - trace_length: specifies the length of each of the sampled traces.
    - rng: specifies the random number generator used, for reproducibility.
    Returns:
    - traces: ndarray of shape (n_traces, n_ap, trace_length).
    """

    traces = rng.integers(0, 2, size=(n_traces, n_ap, trace_length), dtype=np.uint8).astype(bool)

    return traces