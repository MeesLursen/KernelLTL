"""Experiment 2 analysis frame: features + formulas + greedy outcomes.

Builds the per-target frame the Part II analyses run on, and derives the
model covariates. The derivation (``derive_covariates``) is a pure function
of the frame it is handed: the bootstrap re-runs it on every resample, so the
percentile intervals absorb the estimation uncertainty of the binned
conditional means and SDs (the generated-regressor fix) and of the
low-faithfulness quantile cut.

Covariates:
  log_norm    log ||emb(phi)||
  norm_resid  log_norm - binned E[log_norm | variance]   (FWL residual)
  u           norm_resid / within-bin SD                 (studentized; the
              model covariate: "how many local SDs below variance-matched
              peers is this target registered")
  z_variance  globally z-scored variance
  low_faith   1[relational_faithfulness < sample q_{faith_tail}]

Hard gates (raise ValueError):
  * feature/greedy row counts match the dataset metadata size
  * identical formula_id sets across features and greedy
  * variance > 0 everywhere
  * depth parsed from the formula string == target_depth recorded at scoring
    time (cross-checks the parser against the generation pipeline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPERATORS = ("AND", "OR", "->", "~", "X", "F", "G", "U")
UNARY = {"~", "X", "F", "G"}
BINARY = {"AND", "OR", "->", "U"}

DEFAULT_N_BINS = 50
DEFAULT_FAITH_TAIL = 0.05


# --------------------------- formula parsing ------------------------------- #

def tokenize(formula: str) -> list[str]:
    return formula.replace("(", " ( ").replace(")", " ) ").split()


def _parse(tokens: list[str], i: int) -> tuple[int, int]:
    """Return (depth, next_index) for the subformula starting at ``i``.

    Grammar (fully parenthesised infix): atom | (op A) | (A op B).
    """
    if i >= len(tokens):
        raise ValueError("unexpected end of formula")
    if tokens[i] != "(":
        if tokens[i] in UNARY or tokens[i] in BINARY or tokens[i] == ")":
            raise ValueError(f"expected atom at token {i}: {tokens[i]!r}")
        return 0, i + 1
    i += 1
    if tokens[i] in UNARY:
        d, i = _parse(tokens, i + 1)
        if tokens[i] != ")":
            raise ValueError(f"expected ')' at token {i}")
        return d + 1, i + 1
    d1, i = _parse(tokens, i)
    if tokens[i] not in BINARY:
        raise ValueError(f"expected binary operator at token {i}: {tokens[i]!r}")
    d2, i = _parse(tokens, i + 1)
    if tokens[i] != ")":
        raise ValueError(f"expected ')' at token {i}")
    return max(d1, d2) + 1, i + 1


def parse_depth(formula: str) -> int:
    tokens = tokenize(formula)
    depth, nxt = _parse(tokens, 0)
    if nxt != len(tokens):
        raise ValueError(f"trailing tokens in formula: {formula!r}")
    return depth


def load_formulas(path: Path) -> list[str]:
    """Validation formulas: plain strings, one per line, row index = formula_id."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# ----------------------------- frame build --------------------------------- #

def build_frame(features: pd.DataFrame, greedy: pd.DataFrame,
                formulas: list[str], *, expected_n: int | None) -> pd.DataFrame:
    n = len(features)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"features: {n} rows, expected {expected_n} targets")
    if len(formulas) != n:
        raise ValueError(f"formulas: {len(formulas)} lines != {n} feature rows")
    if features["formula_id"].duplicated().any():
        raise ValueError("features: duplicated formula_id")
    if set(features["formula_id"]) != set(greedy["formula_id"]):
        raise ValueError("formula_id sets differ between features and greedy")
    if (features["variance"] <= 0).any():
        raise ValueError("features: non-positive variance rows present")

    df = features.merge(
        greedy[["formula_id", "target_depth", "is_semantic_equivalent",
                "semantic_distance"]],
        on="formula_id", validate="one_to_one")
    df = df.sort_values("formula_id").reset_index(drop=True)

    toks = [tokenize(formulas[int(i)]) for i in df["formula_id"]]
    df["depth"] = [parse_depth(formulas[int(i)]) for i in df["formula_id"]]
    mismatch = df["depth"] != df["target_depth"]
    if mismatch.any():
        bad = df.loc[mismatch, "formula_id"].head(5).tolist()
        raise ValueError(
            f"parsed depth != recorded target_depth for {int(mismatch.sum())} "
            f"targets (e.g. ids {bad}) -- parser and pipeline disagree")

    for op in OPERATORS:
        df[f"has_{op}"] = np.fromiter((float(op in t) for t in toks),
                                      dtype=np.float64, count=len(toks))
    df["correct"] = df["is_semantic_equivalent"].astype(int)
    df["sem_dist"] = df["semantic_distance"].astype(float)
    return df.drop(columns=["is_semantic_equivalent", "semantic_distance"])


# --------------------------- covariate derivation --------------------------- #

def derive_covariates(df: pd.DataFrame, *, n_bins: int = DEFAULT_N_BINS,
                      faith_tail: float = DEFAULT_FAITH_TAIL) -> pd.DataFrame:
    """Derive u, z_variance, low_faith. Pure function; re-run per resample."""
    out = df.copy()
    out["log_norm"] = np.log(out["emb_norm"])
    out["vbin"] = pd.qcut(out["variance"], n_bins, labels=False, duplicates="drop")
    grp = out.groupby("vbin")["log_norm"]
    out["norm_resid"] = out["log_norm"] - grp.transform("mean")
    sd = out.groupby("vbin")["norm_resid"].transform("std")
    # A degenerate bin (zero spread) contributes no within-bin contrast: u = 0.
    out["u"] = np.where(sd > 0, out["norm_resid"] / sd, 0.0)
    out["z_variance"] = ((out["variance"] - out["variance"].mean())
                         / out["variance"].std())
    faith_cut = out["relational_faithfulness"].quantile(faith_tail)
    out["low_faith"] = (out["relational_faithfulness"] < faith_cut).astype(float)
    return out
