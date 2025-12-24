from typing import List
import torch
from formula_class import Atom, Not, And, Or, Implies, Next, Eventually, Globally, Until, Formula

# ------------------------- random formula generator -------------------------
# Operator classes and arities. We'll sample uniformly among these names.
_UNARY_OPS = ['NOT', 'X', 'F', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS



def sample_formulas(n_formula: int,
                    p_leaf_range: tuple[float, float],
                    max_depth: int,
                    n_ap: int,
                    force_tree: bool,
                    rng: torch.Generator,
                    device: str) -> list[Formula]:
    """Generate a random formula.
    - n_formula: Specifies the number of sampled formulae.
    - p_leaf_range: probability to create an atomic proposition at a node.
    - max_depth: maximum recursion depth (root at depth 0). When depth >= max_depth, we force a leaf.
    - n_ap: maximum number of distinct atomic proposition names (p0..p{n_ap-1}).
    - force_tree: specifies whether the root is forced to be an operator.
    - rng: specifies the random number generator used, for reproducibility.
    Returns:
    - ls: a list of formulae
    """

    atoms = list(range(n_ap))

    def gen(depth: int, p_leaf: float, root_must_be_operator: bool = False) -> Formula:
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
        op = _ALL_OPS[torch.randint(0, len(_ALL_OPS), (), generator=rng, device=device).item()]
        if op in _UNARY_OPS:
            # unary
            child : Formula = gen(depth + 1, p_leaf)

            # Avoid redundant unary nesting (e.g., G(G φ))
            while (op == 'G' and isinstance(child, Globally)) or \
                  (op == 'F' and isinstance(child, Eventually)) or \
                  (op == 'NOT' and isinstance(child, Not)):
                child = gen(depth + 1, p_leaf)

            if op == 'NOT':
                return Not(child)
            if op == 'X':
                return Next(child)
            if op == 'F':
                return Eventually(child)
            if op == 'G':
                return Globally(child)
        else:
            p_leaf_left = p_leaf_range[0] + torch.rand((), generator=rng, device=device).item() * (p_leaf_range[1] - p_leaf_range[0])
            p_leaf_right = p_leaf_range[0] + torch.rand((), generator=rng, device=device).item() * (p_leaf_range[1] - p_leaf_range[0])
            # binary
            left : Formula = gen(depth + 1, p_leaf_left)
            right : Formula = gen(depth + 1, p_leaf_right)

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
        p_leaf = p_leaf_range[0] + torch.rand((), generator=rng, device=device).item() * (p_leaf_range[1] - p_leaf_range[0])
        formula = gen(0, p_leaf=p_leaf, root_must_be_operator = force_tree)
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

    baseline_zeros  = torch.zeros(size=(1, n_ap, trace_length), dtype=torch.bool,device=device)
    baseline_ones   = torch.ones(size=(1, n_ap, trace_length), dtype=torch.bool,device=device)
    sampled_traces  = torch.randint(0,2, size=(n_traces, n_ap, trace_length), generator=rng, dtype=torch.bool, device = device)
    traces = torch.cat((baseline_zeros, baseline_ones, sampled_traces), dim=0)
    return traces


def sample_traces_correlated(n_traces: int,
                             n_ap: int,
                             trace_length: int,
                             rng: torch.Generator,
                             device: str,
                             low_variance_ratio: float = 0.5,
                             low_var_switch_prob: float = 0.1) -> torch.Tensor:
    """Sample traces from a mixture of high-variance Bernoulli and low-variance correlated processes.

    Args:
        n_traces: Number of *additional* random traces (excluding the added all-zero/all-one baselines).
        n_ap: Number of atomic propositions per trace.
        trace_length: Length of each trace.
        rng: Torch RNG for reproducibility.
        device: Target device for the returned tensor.
        low_variance_ratio: Fraction of traces generated with strong temporal correlation (in [0,1]).
        low_var_switch_prob: Probability of flipping a proposition at each time-step for the correlated traces.

    Returns:
        Tensor of unique traces with shape (<= n_traces + 2, n_ap, trace_length).
    """

    n_low = int(round(low_variance_ratio * n_traces))
    n_high = max(0, n_traces - n_low)

    def _sample_low_variance(count: int) -> torch.Tensor:
        if count == 0:
            return torch.empty((0, n_ap, trace_length), dtype=torch.bool, device=device)

        traces = torch.empty((count, n_ap, trace_length), dtype=torch.bool, device=device)
        current = torch.randint(0, 2, (count, n_ap), generator=rng, dtype=torch.bool, device=device)
        traces[:, :, 0] = current

        for t in range(1, trace_length):
            flip_mask = torch.rand((count, n_ap), generator=rng, device=device) < low_var_switch_prob
            current = torch.where(flip_mask, torch.logical_not(current), current)
            traces[:, :, t] = current

        return traces

    def _sample_high_variance(count: int) -> torch.Tensor:
        if count == 0:
            return torch.empty((0, n_ap, trace_length), dtype=torch.bool, device=device)
        return torch.randint(0, 2, (count, n_ap, trace_length), generator=rng, dtype=torch.bool, device=device)

    baseline_zeros = torch.zeros((1, n_ap, trace_length), dtype=torch.bool, device=device)
    baseline_ones = torch.ones((1, n_ap, trace_length), dtype=torch.bool, device=device)
    low_variance_traces = _sample_low_variance(n_low)
    high_variance_traces = _sample_high_variance(n_high)

    traces = torch.cat((baseline_zeros, baseline_ones, low_variance_traces, high_variance_traces), dim=0)
    flat = traces.reshape(traces.size(0), -1)
    unique_flat = torch.unique(flat, dim=0)
    unique_traces = unique_flat.reshape(-1, n_ap, trace_length)
    return unique_traces



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
