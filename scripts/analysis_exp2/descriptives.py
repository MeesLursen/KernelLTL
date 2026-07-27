"""Stage A of Experiment 2: model-free audit of the representation.

Every function here consumes only dataset/kernel-derived quantities -- no
model outcome enters (the shuffle-null check reads the *shuffle-ablation*
run, whose whole point is that the conditioning is destroyed). All outputs
are descriptive or diagnostic: population-level statements about the 4,000
validation targets, no inference attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frame import OPERATORS

TERC_LABELS = ("low", "mid", "high")


def _terciles(s: pd.Series) -> pd.Series:
    return pd.qcut(s, 3, labels=TERC_LABELS)


def occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Variance x absolute-norm tercile counts (motivation exhibit)."""
    t = pd.DataFrame({"variance_terc": _terciles(df["variance"]),
                      "norm_terc": _terciles(df["emb_norm"])})
    rows = (t.groupby(["variance_terc", "norm_terc"], observed=False)
            .size().reset_index(name="n"))
    return rows


def norm_variance_curve(df: pd.DataFrame) -> pd.DataFrame:
    """The binned conditional E[log_norm | variance] that u residualises against."""
    g = df.groupby("vbin")
    out = pd.DataFrame({
        "vbin": sorted(df["vbin"].unique()),
    }).set_index("vbin")
    out["n"] = g.size()
    out["mean_variance"] = g["variance"].mean()
    out["mean_log_norm"] = g["log_norm"].mean()
    out["sd_log_norm"] = g["log_norm"].std()
    return out.reset_index()


def faithfulness_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution + leverage concentration of relational faithfulness."""
    f = df["relational_faithfulness"].to_numpy()
    z = (f - f.mean()) / f.std()
    ss = z ** 2
    order = np.argsort(f)
    rows: list[tuple[str, float]] = [("mean", f.mean()), ("sd", f.std())]
    for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        rows.append((f"p{q}", float(np.percentile(f, q))))
    rows.append(("frac_in_085_100", float(((f >= 0.85) & (f <= 1.0)).mean())))
    rows.append(("frac_below_075", float((f < 0.75).mean())))
    for q in (0.5, 1.0, 2.0, 5.0):
        k = int(len(f) * q / 100)
        rows.append((f"var_share_bottom_{q}pct",
                     float(ss[order[:k]].sum() / ss.sum())))
    return pd.DataFrame(rows, columns=["stat", "value"])


def faith_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Mean faithfulness over variance x u tercile cells (H2a's cell exhibit)."""
    t = pd.DataFrame({"variance_terc": _terciles(df["variance"]),
                      "u_terc": _terciles(df["u"]),
                      "faith": df["relational_faithfulness"]})
    g = t.groupby(["variance_terc", "u_terc"], observed=False)["faith"]
    out = g.agg(mean_faith="mean", n="size").reset_index()
    return out


def design_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """How operator-determined is u? Certifies that M2 is identified.

    Reported in the methods text, not results: computed without outcome data.
    """
    y = df["u"].to_numpy()
    H = df[[f"has_{op}" for op in OPERATORS]].to_numpy()

    X1 = np.column_stack([np.ones(len(y)), H])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r2_dummies = 1.0 - (y - X1 @ beta).var() / y.var()

    pattern = df[[f"has_{op}" for op in OPERATORS]].astype(int).astype(str).agg("".join, axis=1)
    within = y - pd.Series(y).groupby(pattern.values).transform("mean").to_numpy()
    r2_pattern = 1.0 - within.var() / y.var()
    sizes = pattern.value_counts()

    return pd.DataFrame([
        ("r2_u_on_has_op", r2_dummies),
        ("r2_u_on_pattern_fe", r2_pattern),
        ("n_patterns", float(sizes.size)),
        ("median_pattern_size", float(sizes.median())),
        ("within_pattern_sd", float(np.sqrt(within.var()))),
        ("total_sd", float(y.std())),
    ], columns=["stat", "value"])


def operator_signature(df: pd.DataFrame) -> pd.DataFrame:
    """Mean u by operator presence (the covariate-side geometry-operator bridge)."""
    rows = []
    for op in OPERATORS:
        present = df.loc[df[f"has_{op}"] == 1, "u"]
        absent = df.loc[df[f"has_{op}"] == 0, "u"]
        delta = present.mean() - absent.mean()
        se = float(np.sqrt(present.var() / len(present) + absent.var() / len(absent)))
        rows.append({"operator": op,
                     "prevalence": float(df[f"has_{op}"].mean()),
                     "mean_u_present": float(present.mean()),
                     "mean_u_absent": float(absent.mean()),
                     "delta_u": float(delta),
                     "t_stat": float(delta / se) if se > 0 else np.nan})
    return pd.DataFrame(rows)


def shuffle_null(df: pd.DataFrame, shuffle_greedy: pd.DataFrame) -> pd.DataFrame:
    """Chance-level equivalence rate under shuffled embeddings.

    Falsification check on target guessability: a rate ~0, flat in variance,
    kills the concern that low-variance targets are mechanically easier to
    hit. Variance-patterned nonzero rates would flag exactly that confound.
    """
    m = df[["formula_id", "variance"]].merge(
        shuffle_greedy[["formula_id", "is_semantic_equivalent"]],
        on="formula_id", validate="one_to_one")
    m["variance_terc"] = _terciles(m["variance"])
    rows = [{"slice": "overall",
             "n": len(m),
             "equiv_rate": float(m["is_semantic_equivalent"].mean())}]
    for terc, sub in m.groupby("variance_terc", observed=False):
        rows.append({"slice": f"variance_{terc}",
                     "n": len(sub),
                     "equiv_rate": float(sub["is_semantic_equivalent"].mean())})
    return pd.DataFrame(rows)
