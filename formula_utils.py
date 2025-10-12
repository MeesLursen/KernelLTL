from typing import List
import torch
from formula_class import Atom, Not, And, Or, Implies, Next, Eventually, Globally, Until, Formula
from kernel_class import LTLKernel

# ------------------------- random formula generator -------------------------
# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'F', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS


def sample_formulas(n_formula: int,
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

    atoms = list(range(n_ap))

    def gen(depth: int, root_must_be_operator: bool = False) -> Formula:
        # If we're at max depth -> force leaf
        if depth >= max_depth:
            return Atom(atoms[torch.randint(0, len(atoms), (), generator=rng, device=device).item()])
        
        if depth == 0 and root_must_be_operator:
            make_leaf = False
        else:
            make_leaf = torch.rand((),generator=rng, device=device).item() < p_leaf

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
def sample_traces(n_traces: int, n_ap:int, trace_length:int, rng: torch.Generator, device: str) -> torch.Tensor:
    """
    - n_traces: specifies the number of traces sampled uniformly at random (each trace is shape (n_ap, T), with values in {False,True}).
    - n_ap: specifies the number of atomic propositions in each trace.
    - trace_length: specifies the length of each of the sampled traces.
    - rng: specifies the random number generator used, for reproducibility.
    Returns:
    - traces: Tensor of shape (n_traces, n_ap, trace_length).
    """

    traces = torch.randint(0,2, size=(n_traces, n_ap, trace_length), generator=rng, dtype=torch.bool, device = device)

    return traces



# ------------------------- formula string parser -------------------------
# helper functions
def _simple_tokenize(s: str) -> List[str]:
    # ensure parentheses separated, then split on whitespace
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    toks = [t for t in s.strip().split() if t != '']
    return toks



class ParseError(Exception):
    pass



def parse_formula_from_tokens(tokens: List[str], pos: int = 0):
    """Recursive parser that returns (Formula, new_pos)"""
    if pos >= len(tokens):
        raise ParseError("Unexpected end of tokens")

    tok = tokens[pos]
    # atom
    if tok.startswith('p_'):
        try:
            idx = int(tok.split('_', 1)[1])
        except Exception:
            raise ParseError(f"Invalid atom token: {tok}")
        return Atom(idx), pos + 1

    # parenthesized expression
    if tok == '(':
        # peek next
        if pos + 1 >= len(tokens):
            raise ParseError('Unexpected end after (')
        next_tok = tokens[pos + 1]
        # unary operators: ~, X, F, G
        if next_tok in ('~', 'X', 'F', 'G'):
            op = next_tok
            child, new_pos = parse_formula_from_tokens(tokens, pos + 2)
            if new_pos >= len(tokens) or tokens[new_pos] != ')':
                raise ParseError('Expected ) after unary')
            if op == '~':
                return Not(child), new_pos + 1
            if op == 'X':
                return Next(child), new_pos + 1
            if op == 'F':
                return Eventually(child), new_pos + 1
            if op == 'G':
                return Globally(child), new_pos + 1

        # otherwise binary: ( left OP right )
        left, p = parse_formula_from_tokens(tokens, pos + 1)
        if p >= len(tokens):
            raise ParseError('Unexpected end after left expr')
        op = tokens[p]
        right, p2 = parse_formula_from_tokens(tokens, p + 1)
        if p2 >= len(tokens) or tokens[p2] != ')':
            raise ParseError('Expected ) after binary')

        if op == 'AND':
            return And(left, right), p2 + 1
        if op == 'OR':
            return Or(left, right), p2 + 1
        if op in ('->', 'IMPLIES'):
            return Implies(left, right), p2 + 1
        if op == 'U' or op == 'UNTIL':
            return Until(left, right), p2 + 1

        raise ParseError(f'Unknown binary operator: {op}')

    raise ParseError(f'Unexpected token: {tok}')



def str_to_formula(s: str) -> Formula:
    toks = _simple_tokenize(s)
    if len(toks) == 0:
        raise ParseError('Empty string')
    f, pos = parse_formula_from_tokens(toks, 0)
    if pos != len(toks):
        raise ParseError('Extra tokens after parsing')
    return f



def is_valid_formula(s: str) -> bool:
    try:
        _ = str_to_formula(s)
        return True
    except Exception:
        return False
