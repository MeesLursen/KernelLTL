"""Formula helpers shared by the analyses: evaluation on the trace sample and syntax.

Evaluation goes through ``formula_class.eval_traces_batch``, the evaluator that
built the kernel and scored the validation passes, so satisfaction vectors computed
here are the ones the stored data were computed with.
"""

from __future__ import annotations

import torch

from formula_class import (And, Eventually, Formula, Globally, Implies, Next, Not, Or,
                           Until, eval_traces_batch)

OPERATORS = (("not", Not), ("next", Next), ("eventually", Eventually), ("globally", Globally),
             ("and", And), ("or", Or), ("implies", Implies), ("until", Until))
OPERATOR_NAMES = tuple(name for name, _ in OPERATORS)


def node_types(formula: Formula) -> set[type]:
    stack, seen = [formula], set()
    while stack:
        f = stack.pop()
        seen.add(type(f))
        if isinstance(f, (Not, Next, Eventually, Globally)):
            stack.append(f.child)
        elif isinstance(f, (And, Or, Implies, Until)):
            stack.extend((f.left, f.right))
    return seen


def operator_presence(formula: Formula) -> dict[str, int]:
    """1 for every operator that occurs anywhere in the formula, 0 otherwise."""
    types = node_types(formula)
    return {name: int(cls in types) for name, cls in OPERATORS}


def satisfaction(formula: Formula, traces: torch.Tensor, *, time_index: int = 0,
                 chunk: int | None = None) -> torch.Tensor:
    """Boolean satisfaction vector (N,) of the formula at ``time_index``, on traces' device."""
    if chunk is None or chunk >= traces.shape[0]:
        return eval_traces_batch(formula, traces)[:, time_index]
    out = torch.empty(traces.shape[0], dtype=torch.bool, device=traces.device)
    for s in range(0, traces.shape[0], chunk):
        out[s:s + chunk] = eval_traces_batch(formula, traces[s:s + chunk])[:, time_index]
    return out
