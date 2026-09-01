"""Experiment 2 analysis frame: features + formulas + greedy outcomes.

Builds the per-target frame the Part II analyses run on, and derives the
model covariates. The derivation (``derive_covariates``) is a pure function
of the frame it is handed: the bootstrap re-runs it on every resample, so the
percentile intervals absorb the estimation uncertainty of the binned
conditional means and SDs (the generated-regressor fix) and of the Fisher-z
faithfulness standardisation.

Covariates:
  log10_norm  log10 ||emb(phi)||                         (BASE 10 throughout:
              u is a ratio of logs and so is exactly base-invariant, as are
              all coefficients; the base is visible only in the logged columns
              reported by descriptives and plotted on the transform figures.
              Base 10 makes those read in DECADES -- the same unit as the
              log-scaled variance axis they sit beside -- so a span in norm
              and a span in variance can be compared without arithmetic.)
  norm_resid  log10_norm - binned E[log10_norm | variance]   (FWL residual)
  u           norm_resid / within-bin SD                 (studentized; the
              "how many local SDs below variance-matched peers is this target
              registered" quantity. NOT a model term: linearity in the logit is
              rejected for u and a quadratic does not repair it, so u enters
              the models as decile indicators -- see u_d1..u_d9.)
  u_d1..u_d9  indicators for u's 2nd..10th decile, 1st decile as reference.
              This is how u ENTERS EVERY MODEL. Nine indicators rather than
              ten because the depth block is cell-mean coded (no shared
              intercept) and so already spans the constant. Derived here, not
              in models.py, so the bootstrap re-cuts them on every resample
              along with u itself (S4).
  u_sq        u**2                    NOT a model term. Kept solely so the
              specification search can fit the rejected quadratic-in-u form
              and report why it was rejected.
  z_variance  globally z-scored variance
  z_variance_sq
              z_variance**2           A PRIMARY model term. Linearity in the
              logit is rejected for V and the quadratic repairs it at decile
              resolution, so V enters every model as z_variance + z_variance_sq.
  z_faith     z-scored Fisher-z (atanh) of relational_faithfulness; the
              continuous faithfulness covariate for the F-branch and M3.
              Fisher-z is the standard variance stabiliser for correlations
              and decompresses the near-1 bulk so the coefficient is not
              driven purely by the low tail; |rho| is clipped at FAITH_CLIP
              to guard the atanh pole (affects only exact-|1| values).

Hard gates (raise ValueError):
  * feature/greedy row counts match the dataset metadata size
  * identical formula_id sets across features and greedy
  * variance > 0 everywhere
  * relational_faithfulness finite and inside [-1, 1] everywhere
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
FAITH_CLIP = 1.0 - 1e-6

# u enters the models as decile indicators; 10 cuts -> 9 columns (see
# derive_covariates). Chosen by the specification search, not by convention:
# linear and quadratic forms are both rejected, deciles are not, and a 20-bin
# cut finds no further structure (p = 0.44).
U_DECILES = 10
U_DEC_COLS = [f"u_d{k}" for k in range(1, U_DECILES)]


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
    faith = features["relational_faithfulness"]
    if not np.isfinite(faith).all():
        raise ValueError("features: non-finite relational_faithfulness present")
    if (faith.abs() > 1.0).any():
        raise ValueError("features: relational_faithfulness outside [-1, 1]")

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

def derive_covariates(df: pd.DataFrame, *,
                      n_bins: int = DEFAULT_N_BINS) -> pd.DataFrame:
    """Derive u, z_variance, z_faith. Pure function; re-run per resample."""
    out = df.copy()
    # Base 10: u, its deciles and every coefficient are invariant (u is a ratio
    # of logs), so this is a reporting choice only -- it buys decades as the
    # unit of the logged columns and figure axes. See the module docstring.
    out["log10_norm"] = np.log10(out["emb_norm"])
    out["vbin"] = pd.qcut(out["variance"], n_bins, labels=False, duplicates="drop")
    grp = out.groupby("vbin")["log10_norm"]
    out["norm_resid"] = out["log10_norm"] - grp.transform("mean")
    sd = out.groupby("vbin")["norm_resid"].transform("std")
    # A degenerate bin (zero spread) contributes no within-bin contrast: u = 0.
    out["u"] = np.where(sd > 0, out["norm_resid"] / sd, 0.0)
    out["u_sq"] = out["u"] ** 2            # specification search only, not a term
    # u's model representation. Re-cut here rather than in models.py so the
    # bootstrap re-derives the cut points on every resample, exactly as it
    # re-derives u (S4). Decile 0 is the reference and gets no column.
    udec = pd.qcut(out["u"], U_DECILES, labels=False, duplicates="drop")
    for k in range(1, U_DECILES):
        out[f"u_d{k}"] = (udec == k).astype(np.float64)
    out["z_variance"] = ((out["variance"] - out["variance"].mean())
                         / out["variance"].std())
    out["z_variance_sq"] = out["z_variance"] ** 2   # PRIMARY term (see docstring)
    fz = np.arctanh(out["relational_faithfulness"].clip(-FAITH_CLIP, FAITH_CLIP))
    out["z_faith"] = (fz - fz.mean()) / fz.std()
    return out
