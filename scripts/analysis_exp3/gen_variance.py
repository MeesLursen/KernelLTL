"""Generated-formula variance for the Experiment 1 greedy records (Test 1 input).

The greedy records store the generated string but not its satisfaction vector.
Test 1 needs the generated formula's base rate and variance, so this module
parses each generated string and evaluates it over the kernel's trace sample,
exactly as ``validation_utils`` did when scoring.

Two things keep this cheap and exact:

  * a semantically equivalent generation has, by definition of the scoring,
    the *same* satisfaction vector as its target, hence the same base rate and
    variance -- those rows are copied from the features table, not re-evaluated;
  * invalid generations have no satisfaction vector and are NaN.

Only the misses are evaluated. On CPU that is ~1.5k formulae over 500k traces;
pass ``--device cuda`` to the driver where available.

Before anything is evaluated, ``verify_trace_sample`` recomputes the base rate
of a sample of *target* strings on the supplied traces and compares it with the
``p`` column of the features table. A mismatch means the traces are not the
sample the kernel (and the features) were built on, and the run aborts: a
generated variance computed on the wrong traces is not a variance error, it is
noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from formula_class import eval_traces_batch
from formula_utils import ParseError, str_to_formula


def load_traces(path: Path, *, device: str = "cpu") -> torch.Tensor:
    """``traces.pt`` -- either the file itself or the kernel directory holding it."""
    p = Path(path)
    if p.is_dir():
        p = p / "traces.pt"
    traces = torch.load(p, map_location="cpu")
    if not isinstance(traces, torch.Tensor) or traces.ndim != 3 or traces.dtype != torch.bool:
        raise ValueError(f"{p}: expected a bool tensor of shape (N, AP, T)")
    return traces.to(device)


@torch.no_grad()
def base_rate_of(formula_str: str, traces: torch.Tensor, *, time_index: int, batch_size: int) -> float:
    """Empirical base rate of ``formula_str`` over ``traces`` at ``time_index``.

    Raises ``ParseError`` for an unparseable string.
    """
    formula = str_to_formula(formula_str)
    N = traces.size(0)
    total = 0
    for j in range(0, N, batch_size):
        sats = eval_traces_batch(formula, traces[j:j + batch_size])[:, time_index]
        total += int(sats.sum().item())
    return total / N


def verify_trace_sample(
    features: pd.DataFrame, greedy: pd.DataFrame, traces: torch.Tensor, *,
    time_index: int, batch_size: int, n_sample: int, seed: int, atol: float = 1e-6,
) -> dict:
    """Gate: the supplied traces reproduce the features' base rates for sampled targets.

    ``atol`` allows for the features' ``p`` being a float32 mean (compute_features.py);
    a wrong trace sample disagrees at the 1e-2 level, so 1e-6 separates the cases.
    """
    if n_sample <= 0:
        return {"checked": 0, "max_abs_diff": None, "passed": True, "skipped": True}
    rng = np.random.default_rng(seed)
    merged = features[["formula_id", "p"]].merge(
        greedy[["formula_id", "target_formula_str"]], on="formula_id", validate="one_to_one")
    rows = merged.iloc[rng.choice(len(merged), size=min(n_sample, len(merged)), replace=False)]
    diffs = []
    for _, r in rows.iterrows():
        p_hat = base_rate_of(r["target_formula_str"], traces, time_index=time_index, batch_size=batch_size)
        diffs.append(abs(p_hat - float(r["p"])))
    max_diff = max(diffs)
    passed = max_diff <= atol
    if not passed:
        raise RuntimeError(
            f"trace-sample gate failed: max |p_recomputed - p_features| = {max_diff:.3e} "
            f"over {len(diffs)} targets (atol {atol:g}). These traces are not the kernel's sample."
        )
    return {"checked": len(diffs), "max_abs_diff": max_diff, "passed": True, "skipped": False}


def add_generated_variance(
    greedy: pd.DataFrame, features: pd.DataFrame, traces: torch.Tensor, *,
    time_index: int, batch_size: int, log=None,
) -> pd.DataFrame:
    """Return ``greedy`` joined to the features with generated base rate / variance columns.

    Adds ``target_base_rate``, ``target_variance``, ``generated_base_rate``,
    ``generated_variance`` and ``delta_variance = generated - target``. Hits copy the
    target values; invalid rows are NaN; misses are evaluated on ``traces``.
    """
    df = greedy.merge(
        features[["formula_id", "p", "variance"]], on="formula_id", validate="one_to_one")
    df = df.rename(columns={"p": "target_base_rate", "variance": "target_variance"})

    gen_p = np.full(len(df), np.nan, dtype=np.float64)
    hit = df["is_semantic_equivalent"].to_numpy(dtype=bool)
    invalid = df["is_invalid"].to_numpy(dtype=bool)
    gen_p[hit] = df.loc[hit, "target_base_rate"].to_numpy(dtype=np.float64)

    todo = np.where(~hit & ~invalid)[0]
    if log is not None:
        log(f"evaluating {len(todo)} generated formulae on {traces.size(0)} traces "
            f"({int(hit.sum())} hits copied, {int(invalid.sum())} invalid skipped)")
    for k, i in enumerate(todo):
        s = df.at[i, "generated_formula_str"]
        try:
            gen_p[i] = base_rate_of(s, traces, time_index=time_index, batch_size=batch_size)
        except ParseError as exc:
            # The record says valid; the parser disagrees. Do not paper over it.
            raise RuntimeError(f"formula_id {df.at[i, 'formula_id']}: recorded as valid but "
                               f"unparseable: {s!r}") from exc
        if log is not None and (k + 1) % 200 == 0:
            log(f"  {k + 1}/{len(todo)}")

    df["generated_base_rate"] = gen_p
    df["generated_variance"] = gen_p * (1.0 - gen_p)
    df["delta_variance"] = df["generated_variance"] - df["target_variance"]
    # Exactness check on the copied hits: zero error by construction.
    assert np.all(np.abs(df.loc[hit, "delta_variance"].to_numpy()) < 1e-12)
    return df


def prior_variance(zero_greedy: pd.DataFrame, traces: torch.Tensor, *,
                   time_index: int, batch_size: int) -> dict:
    """V_0: the variance of what the decoder emits under the zero embedding.

    The zero-ablation run emits one fixed string for every target (Experiment 1);
    this returns that string's base rate and variance, plus how many distinct strings
    the run actually contained, so a non-degenerate prior is reported rather than
    assumed.
    """
    valid = zero_greedy.loc[~zero_greedy["is_invalid"], "generated_formula_str"]
    counts = valid.value_counts()
    if counts.empty:
        return {"n_distinct": 0, "modal_formula": None, "modal_share": 0.0, "p0": math.nan, "V0": math.nan}
    modal = str(counts.index[0])
    p0 = base_rate_of(modal, traces, time_index=time_index, batch_size=batch_size)
    return {
        "n_distinct": int(len(counts)),
        "modal_formula": modal,
        "modal_share": float(counts.iloc[0] / len(zero_greedy)),
        "p0": p0,
        "V0": p0 * (1.0 - p0),
    }
