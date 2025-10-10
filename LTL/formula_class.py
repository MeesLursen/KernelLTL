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
    
    def eval_trace(self, trace: torch.Tensor) -> bool:
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

    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
        T = trace.size(dim=1)
        sub = self.child.eval_trace(trace)
        out = torch.empty_like(sub)

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
    
    def eval_trace(self, trace: torch.Tensor) -> torch.Tensor:
        T = trace.size(dim=1)
        sub = self.child.eval_trace(trace)
        out = torch.empty_like(sub)

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
        T = trace.size(dim=1)
        L = self.left.eval_trace(trace)
        R = self.right.eval_trace(trace)
        out = torch.zeros(T)
        
        out[-1] = R[-1]
        for t in range(T-2, -1, -1):
            out[t] = R[t] or (L[t] and out[t + 1])
        
        return out



# ------------------------- vectorized batch evaluator -------------------------
def eval_traces_batch(formula: Formula, traces_batch: torch.Tensor) -> torch.Tensor:
    """
    Evaluate formula on a batch of traces.

    - formula: AST formula (Atom/Not/And/Or/Next/Eventually/Globally/Until)
    - traces_batch: torch.Tensor, shape (batch_size=B, n_ap, T), dtype=torch.bool,
    Returns:
    - out: torch.Tensor, shape (B, T), dtype=torch.bool,
            out[i, t] == True iff trace i suffix at time t satisfies formula.
    """
    T = traces_batch.size(dim=2)

    # atoms
    if isinstance(formula, Atom):
        return traces_batch[:, formula.name, :]

    # boolean connectives
    if isinstance(formula, Not):
        child = eval_traces_batch(formula.child, traces_batch)  # (B,T)
        return torch.logical_not(child)

    if isinstance(formula, And):
        L = eval_traces_batch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch(formula.right, traces_batch)  # (B,T)
        return torch.logical_and(L, R)

    if isinstance(formula, Or):
        L = eval_traces_batch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch(formula.right, traces_batch)  # (B,T)
        return torch.logical_or(L, R)
    
    if isinstance(formula, Implies):
        L = eval_traces_batch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch(formula.right, traces_batch)  # (B,T)
        return torch.logical_or(torch.logical_not(L), R)


    # temporal-unary
    if isinstance(formula, Next):
        child = eval_traces_batch(formula.child, traces_batch)  # (B,T)
        out = torch.zeros_like(child)  # (B,T)
        if T >= 2:
            out[:, :-1] = child[:, 1:]
        return out

    if isinstance(formula, Globally):
        child = eval_traces_batch(formula.child, traces_batch)  # (B,T)
        rev = torch.flip(child, [1]).to(torch.uint8)  # (B,T)
        cum = torch.cumprod(rev, dim=1, dtype=torch.uint8)
        out = torch.flip(cum, [1]).to(torch.bool)
        return out

    if isinstance(formula, Eventually):
        child = eval_traces_batch(formula.child, traces_batch)  # (B,T)
        rev = torch.flip(child, [1]).to(torch.uint8)  # (B,T)
        cum = torch.cummax(rev, dim=1).values
        out = torch.flip(cum, [1]).to(torch.bool)
        return out

    # temporal-2ary
    if isinstance(formula, Until):
        L = eval_traces_batch(formula.left, traces_batch)  # (B,T)
        R = eval_traces_batch(formula.right, traces_batch)  # (B,T)
        out = torch.empty_like(R)  # (B,T)
        out[:, -1] = R[:, -1]
        for t in range(T-2, -1, -1):
            out[:, t] = torch.logical_or(R[:, t], torch.logical_and(L[:, t], out[:, t + 1]))
        return out

    raise NotImplementedError(f"Unknown formula type: {type(formula)}")