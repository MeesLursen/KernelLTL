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
                right = gen(depth + 1, p_leaf_right)

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



# ------------------------- formula mutation utilities -------------------------
def _children(formula: Formula) -> list[Formula]:
    if isinstance(formula, (Atom,)):
        return []
    if isinstance(formula, (Not, Next, Eventually, Globally)):
        return [formula.child]
    if isinstance(formula, (And, Or, Implies, Until)):
        return [formula.left, formula.right]
    raise TypeError(f"Unsupported formula node type: {type(formula)}")


def _rebuild_with_children(formula: Formula, children: list[Formula]) -> Formula:
    if isinstance(formula, Atom):
        if len(children) != 0:
            raise ValueError("Atom expects zero children")
        return formula
    if isinstance(formula, Not):
        if len(children) != 1:
            raise ValueError("Not expects one child")
        return Not(children[0])
    if isinstance(formula, Next):
        if len(children) != 1:
            raise ValueError("Next expects one child")
        return Next(children[0])
    if isinstance(formula, Eventually):
        if len(children) != 1:
            raise ValueError("Eventually expects one child")
        return Eventually(children[0])
    if isinstance(formula, Globally):
        if len(children) != 1:
            raise ValueError("Globally expects one child")
        return Globally(children[0])
    if isinstance(formula, And):
        if len(children) != 2:
            raise ValueError("And expects two children")
        return And(children[0], children[1])
    if isinstance(formula, Or):
        if len(children) != 2:
            raise ValueError("Or expects two children")
        return Or(children[0], children[1])
    if isinstance(formula, Implies):
        if len(children) != 2:
            raise ValueError("Implies expects two children")
        return Implies(children[0], children[1])
    if isinstance(formula, Until):
        if len(children) != 2:
            raise ValueError("Until expects two children")
        return Until(children[0], children[1])
    raise TypeError(f"Unsupported formula node type: {type(formula)}")


def _iter_paths(formula: Formula, prefix: tuple[int, ...] = ()):
    yield prefix
    for idx, child in enumerate(_children(formula)):
        yield from _iter_paths(child, prefix + (idx,))


def _subformula_at_path(formula: Formula, path: tuple[int, ...]) -> Formula:
    node = formula
    for child_idx in path:
        kids = _children(node)
        if child_idx < 0 or child_idx >= len(kids):
            raise IndexError(f"Invalid child index {child_idx} in path {path}")
        node = kids[child_idx]
    return node


def _replace_subformula(formula: Formula, path: tuple[int, ...], replacement: Formula) -> Formula:
    if len(path) == 0:
        return replacement
    child_idx = path[0]
    kids = _children(formula)
    if child_idx < 0 or child_idx >= len(kids):
        raise IndexError(f"Invalid child index {child_idx} in path {path}")
    new_children = list(kids)
    new_children[child_idx] = _replace_subformula(kids[child_idx], path[1:], replacement)
    return _rebuild_with_children(formula, new_children)


def _common_factor_in_and(lhs: And, rhs: And) -> tuple[Formula, Formula, Formula] | None:
    lhs_pairs = [(lhs.left, lhs.right), (lhs.right, lhs.left)]
    rhs_pairs = [(rhs.left, rhs.right), (rhs.right, rhs.left)]
    for shared_l, other_l in lhs_pairs:
        for shared_r, other_r in rhs_pairs:
            if shared_l == shared_r:
                return shared_l, other_l, other_r
    return None


def _common_factor_in_or(lhs: Or, rhs: Or) -> tuple[Formula, Formula, Formula] | None:
    lhs_pairs = [(lhs.left, lhs.right), (lhs.right, lhs.left)]
    rhs_pairs = [(rhs.left, rhs.right), (rhs.right, rhs.left)]
    for shared_l, other_l in lhs_pairs:
        for shared_r, other_r in rhs_pairs:
            if shared_l == shared_r:
                return shared_l, other_l, other_r
    return None


def _local_semantic_equivalent_rewrites(formula: Formula) -> list[Formula]:
    rewrites: list[Formula] = []

    if isinstance(formula, Not):
        if isinstance(formula.child, Not):
            rewrites.append(formula.child.child)
        if isinstance(formula.child, And):
            if not isinstance(formula.child.left, Not):
                if not isinstance(formula.child.right, Not):
                    rewrites.append(Or(Not(formula.child.left), Not(formula.child.right)))
                else:
                    rewrites.append(Or(Not(formula.child.left), formula.child.right.child))
            if isinstance(formula.child.left, Not):
                if not isinstance(formula.child.right, Not):
                    rewrites.append(Or(formula.child.left.child, Not(formula.child.right)))
                else:
                    rewrites.append(Or(formula.child.left.child, formula.child.right.child))
        if isinstance(formula.child, Or):
            if not isinstance(formula.child.left, Not):
                if not isinstance(formula.child.right, Not):
                    rewrites.append(And(Not(formula.child.left), Not(formula.child.right)))
                if isinstance(formula.child.right, Not):
                    rewrites.append(And(Not(formula.child.left), formula.child.right.child))
            if isinstance(formula.child.left, Not):
                if not isinstance(formula.child.right, Not):
                    rewrites.append(And(formula.child.left.child, Not(formula.child.right)))
                if isinstance(formula.child.right, Not):
                    rewrites.append(And(formula.child.left.child, formula.child.right.child))
        if isinstance(formula.child, Eventually):
            rewrites.append(Globally(Not(formula.child.child)))
        if isinstance(formula.child, Globally):
            rewrites.append(Eventually(Not(formula.child.child)))

    if isinstance(formula, And):
        if formula.left != formula.right:
            rewrites.append(And(formula.right, formula.left))

        # Distributivity: a AND (b OR c) -> (a AND b) OR (a AND c)
        if isinstance(formula.left, Or):
            rewrites.append(Or(And(formula.left.left, formula.right), And(formula.left.right, formula.right)))
        if isinstance(formula.right, Or):
            rewrites.append(Or(And(formula.left, formula.right.left), And(formula.left, formula.right.right)))

        # Reverse De Morgan: (~a AND ~b) -> ~(a OR b)
        if isinstance(formula.left, Not) and isinstance(formula.right, Not):
            rewrites.append(Not(Or(formula.left.child, formula.right.child)))

        # Factoring: (a OR b) AND (a OR c) -> a OR (b AND c)
        if isinstance(formula.left, Or) and isinstance(formula.right, Or):
            shared = _common_factor_in_or(formula.left, formula.right)
            if shared is not None:
                common, rem_l, rem_r = shared
                rewrites.append(Or(common, And(rem_l, rem_r)))

    if isinstance(formula, Or):
        if formula.left != formula.right:
            rewrites.append(Or(formula.right, formula.left))

        # Distributivity: a OR (b AND c) -> (a OR b) AND (a OR c)
        if isinstance(formula.left, And):
            rewrites.append(And(Or(formula.left.left, formula.right), Or(formula.left.right, formula.right)))
        if isinstance(formula.right, And):
            rewrites.append(And(Or(formula.left, formula.right.left), Or(formula.left, formula.right.right)))

        # Reverse De Morgan: (~a OR ~b) -> ~(a AND b)
        if isinstance(formula.left, Not) and isinstance(formula.right, Not):
            rewrites.append(Not(And(formula.left.child, formula.right.child)))

        # Implication elimination/inversion.
        if isinstance(formula.left, Not):
            rewrites.append(Implies(formula.left.child, formula.right))
        if isinstance(formula.right, Not):
            rewrites.append(Implies(formula.right.child, formula.left))
        if not isinstance(formula.left, Not):
            rewrites.append(Implies(Not(formula.left), formula.right))
        if not isinstance(formula.right, Not):
            rewrites.append(Implies(Not(formula.right), formula.left))

        # Factoring: (a AND b) OR (a AND c) -> a AND (b OR c)
        if isinstance(formula.left, And) and isinstance(formula.right, And):
            shared = _common_factor_in_and(formula.left, formula.right)
            if shared is not None:
                common, rem_l, rem_r = shared
                rewrites.append(And(common, Or(rem_l, rem_r)))

    if isinstance(formula, Implies):
        if not isinstance(formula.left, Not):
            # Material Implication
            rewrites.append(Or(Not(formula.left), formula.right))
            if not isinstance(formula.right, Not):
                # Contraposition: (a -> b) <-> (~b -> ~a)
                rewrites.append(Implies(Not(formula.right), Not(formula.left)))
            if isinstance(formula.right, Not):
                rewrites.append(Implies(formula.right.child, Not(formula.left)))
        if isinstance(formula.left, Not):
            rewrites.append(Or(formula.left.child, formula.right))
            if not isinstance(formula.right, Not):
                rewrites.append(Implies(Not(formula.right), formula.left.child))
            if isinstance(formula.right, Not):
                rewrites.append(Implies(formula.right.child, formula.left.child))

    if isinstance(formula, Next):
        if isinstance(formula.child, And):
            rewrites.append(And(Next(formula.child.left), Next(formula.child.right)))
        if isinstance(formula.child, Or):
            rewrites.append(Or(Next(formula.child.left), Next(formula.child.right)))
        if isinstance(formula.child, Eventually):
            rewrites.append(Eventually(Next(formula.child.child)))
        if isinstance(formula.child, Until):
            rewrites.append(Until(Next(formula.child.left), Next(formula.child.right))) # TODO: suspicious rewrite because of strong next semantics

    if isinstance(formula, Eventually):
        if isinstance(formula.child, Not):
            # F(~phi) <-> ~(G(phi))
            rewrites.append(Not(Globally(formula.child.child)))
        if not isinstance(formula.child, Not):
            # F(phi) <-> ~(G(~phi))
            rewrites.append(Not(Globally(Not(formula.child))))
        if isinstance(formula.child, Next):
            # F(X phi) <-> X(F phi)
            rewrites.append(Next(Eventually(formula.child.child)))

    if isinstance(formula, Globally):
        # G(phi) <-> ~(F(~phi))
        if isinstance(formula.child, Not):
            # G(~phi) <-> ~(F(phi))
            rewrites.append(Not(Eventually(formula.child.child)))
        if not isinstance(formula.child, Not):
            rewrites.append(Not(Eventually(Not(formula.child))))

    if isinstance(formula, Until):
        rewrites.append(Or(formula.right, And(formula.left, Next(formula))))

        if isinstance(formula.right, Or):
            rewrites.append(Or(Until(formula.left, formula.right.left), Until(formula.left, formula.right.right))) # TODO: suspicious rewrite because of strong next semantics
        if isinstance(formula.left, And):
            rewrites.append(And(Until(formula.left.left, formula.right), Until(formula.left.right, formula.right))) # TODO: suspicious rewrite because of strong next semantics
        if isinstance(formula.left, Next) and isinstance(formula.right, Next):
            rewrites.append(Next(Until(formula.left.child, formula.right.child))) # TODO: suspicious rewrite because of strong next semantics

    unique: list[Formula] = []
    seen: set[str] = set()
    for candidate in rewrites:
        cand_str = str(candidate)
        if cand_str in seen:
            continue
        seen.add(cand_str)
        unique.append(candidate)
    return unique


def list_semantically_equivalent_transformations(formula: Formula) -> list[Formula]:
    """Enumerate one-step semantic-equivalent rewrites of a formula.

    The list contains full-formula variants obtained by applying one local rewrite
    at exactly one AST node.
    """

    transformed_formulas: list[Formula] = []
    seen: set[str] = {str(formula)}

    for path in _iter_paths(formula):
        local = _subformula_at_path(formula, path)
        for local_rewrite in _local_semantic_equivalent_rewrites(local):
            if local_rewrite == local:
                continue
            candidate = _replace_subformula(formula, path, local_rewrite)
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            transformed_formulas.append(candidate)

    return transformed_formulas


def sample_random_semantically_equivalent_transformation(
    formula: Formula,
    num_samples: int,
    rng: torch.Generator,
    device: str = "cpu",
) -> List[Formula]:
    """Sample semantic-equivalent one-step rewrites uniformly without replacement."""

    candidates = list_semantically_equivalent_transformations(formula)
    if num_samples <= 0 or len(candidates) == 0:
        return []
    draw_count = min(int(num_samples), len(candidates))
    ids = torch.randperm(len(candidates), generator=rng)[:draw_count].tolist()
    return [candidates[idx] for idx in ids]


def list_negation_insertions(formula: Formula) -> list[Formula]:
    """Enumerate unique one-step formulas formed by inserting one negation at one AST node."""

    candidates: list[Formula] = []
    seen: set[str] = set()
    for path in _iter_paths(formula):
        selected_subformula = _subformula_at_path(formula, path)
        candidate = _replace_subformula(formula, path, Not(selected_subformula))
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        candidates.append(candidate)
    return candidates


def add_random_negation(
    formula: Formula,
    rng: torch.Generator,
    device: str = "cpu",
) -> Formula:
    """Insert a negation at a uniformly sampled AST node."""

    candidates = list_negation_insertions(formula)
    if len(candidates) == 0:
        return Not(formula)
    idx = int(torch.randint(0, len(candidates), (), generator=rng, device=device).item())
    return candidates[idx]



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


# ------------------------- tree edit distance -------------------------


def _node_label(formula: Formula) -> str:
    """Relabel-cost label for tree_edit_distance: atoms carry their proposition and
    operators their class, so ``p_0`` vs ``p_1`` and ``And`` vs ``Or`` are both unit-cost
    relabels."""
    if isinstance(formula, Atom):
        return f"p_{formula.name}"
    return type(formula).__name__


def _formula_to_tuple(formula: Formula) -> tuple:
    """Nested ``(label, children-forest)`` view of an AST, for tree_edit_distance."""
    return (_node_label(formula), tuple(_formula_to_tuple(c) for c in _children(formula)))


def tree_edit_distance(a: Formula, b: Formula, *, normalize: bool = True) -> float:
    """Unit-cost ordered tree-edit distance between two formula ASTs.

    Each node insert / delete / relabel costs 1; a relabel is free only when labels match
    (same operator class, or same atom). Computed by the standard ordered-forest recursion
    with per-call memoisation -- ample for formulae of depth <= 5. With ``normalize`` the
    distance is divided by the total node count ``|a| + |b|``, giving a value in [0, 1]
    (0 = identical trees, 1 = fully disjoint).

    This is the size-unbiased pairwise distance used for the correct-only equivalence-class
    spread (Experiment Design ``sec:rq2_mechanism_syntax``; ``flexibility_metrics.py``). A
    commutative swap costs two relabels, not a maximal change -- the property BLEU-4 lacked.
    """
    def _forest_size(forest: tuple) -> int:
        return sum(1 + _forest_size(kids) for (_, kids) in forest)

    memo: dict[tuple, int] = {}

    def _fd(F: tuple, G: tuple) -> int:
        key = (F, G)
        if key in memo:
            return memo[key]
        if not F and not G:
            r = 0
        elif not G:
            r = _forest_size(F)
        elif not F:
            r = _forest_size(G)
        else:
            (ls, cs), (lt, ct) = F[-1], G[-1]            # rightmost trees
            F1, G1 = F[:-1], G[:-1]
            delete = _fd(F1 + cs, G) + 1                 # drop a's node, promote its children
            insert = _fd(F, G1 + ct) + 1                 # add b's node
            match = _fd(F1, G1) + _fd(cs, ct) + (0 if ls == lt else 1)
            r = min(delete, insert, match)
        memo[key] = r
        return r

    ta, tb = (_formula_to_tuple(a),), (_formula_to_tuple(b),)
    dist = _fd(ta, tb)
    if normalize:
        n = _forest_size(ta) + _forest_size(tb)
        return dist / n if n else 0.0
    return float(dist)
