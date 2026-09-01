"""Experiment 2 analysis driver: cached features + validation JSONL -> tables.

Consumes the feature extraction of compute_features.py, the validation
formulas, and the greedy per-generation records of the conditioned run
(plus, optionally, the shuffle-ablation run for the guessability null).
Emits tidy CSV tables -- everything the local data-viz layer and the thesis
need, and nothing else. No plotting here.

Outputs (under --output-dir):
    spec_search.csv       the linearity ladder that CHOSE the    (Stage B; read
                          specification: every candidate form     BEFORE any
                          against decile and 20-bin references    estimand)
    spec_curves.csv       fitted linear and quadratic response   (Stage B; the
                          curves at the D + S base, for overlay   lines on the
                          on the decile points                    spec figure)
    adequacy.csv          AUC by nested block (both orders),     (Stage B)
                          Pregibon link test, dfbeta influence
    occupancy.csv         decile cross-tabs: variance x norm,    (motivation +
                          z_variance x z_faith                    positivity)
    norm_variance.csv     binned E[log10_norm | variance] curve  (Stage A)
    norm_variance_stats.csv   ridge fit, VIF pair, C1 edge case  (Stage A)
    covariates.csv        shape + leverage for every covariate,  (Stage A)
                          before and after its transform
    faith_by_variance.csv C4's reliability channel by stratum    (Stage A)
    diagnostic.csv        R2 of V, u, F on operator features     (Stage A)
    op_signature.csv      joint per-operator shift in V, u, F    (Stage A bridge)
    depth_op_mix.csv      operator prevalence per depth cell     (Stage A)
    shuffle_null.csv      chance-level equivalence rates         (check; optional)
    m_ladder.csv          the rung lattice M0/M1/M2/M3u/M3F/M4   (Stage B/C)
    marginal_effects.csv  +1 SD probability-scale effect of F    (Stage B/C)
                          per rung + its attenuation step
    curve_z_variance.csv  variance-decile correctness at each    (Q1's ESTIMAND,
    curve_u.csv           step of the covariate's curve           Q3's ESTIMAND,
    curve_z_faith.csv     sequence: raw, D+S, then the other      Q2 companion)
                          geometry blocks one at a time
    operators.csv         M1: per-operator joint contrasts       (Stage C; Q4)
    depth_curve.csv       raw + operator-standardised            (descriptive)
                          correctness per depth
    checks.csv            gate outcomes
    manifest.json         inputs, frozen constants, tier map, estimand family

THE SPECIFICATION IS A RESULT. Linearity in the logit is rejected for V and u
and holds for F, so V enters as a quadratic and u as decile indicators
everywhere. ``spec_search.csv`` reports every candidate form against two
flexible references and is what selected them; the rejected linear fits are
tabulated beside it so the search is auditable rather than asserted. See the
``models`` module docstring for the ladder and ASSUMPTIONS.md for the rest.

Because V and u are non-monotone in correctness, NEITHER GETS A SCALAR: an
average marginal effect of a shift measures where the population sits relative
to the optimum rather than the strength of the relationship, and for u under
decile coding it is undefined. Their curves are the estimands. Only F, which
is linear, carries an AME.

Usage::

    python scripts/analysis_exp2/run_exp2.py \
        --features-dir <artifacts>/analysis/exp2/features \
        --run-dir <validation_root>/ce_base \
        --shuffle-run-dir <validation_root>/ce_base_ablation_shuffle \
        --dataset-dir <artifacts>/datasets/validation \
        --output-dir <artifacts>/analysis/exp2/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "analysis_exp1"))
from bootstrap import DEFAULT_B, DEFAULT_SEED, index_matrix              # noqa: E402
from load import load_greedy, read_dataset_size                          # noqa: E402

import descriptives as desc                                              # noqa: E402
import models                                                            # noqa: E402
from frame import (DEFAULT_N_BINS, FAITH_CLIP, OPERATORS,                # noqa: E402
                   build_frame, derive_covariates, load_formulas)

# Stated WITHOUT interventional vocabulary. The dependency structure is here to
# justify which adjustment set each quantity is read at, and the adjustment
# sets follow from constructional facts (C1-C5) rather than from assumptions
# about unmeasured causes: S and D are computed from the formula string before
# any trace is drawn (C5), F is computed downstream of V (C4), u is orthogonal
# to V by construction (C1), and F and u have no directed relation but are
# dependent (C2), so they enter together. Nothing here is an effect claim.
ESTIMANDS = {
    "variance_total": (
        "the variance curve at M2 (curve_z_variance.csv, step DS): adjusted "
        "greedy semantic-equivalence rate by variance decile at matched depth "
        "and operator presence, conditioning on nothing computed downstream of "
        "variance. No scalar: the relationship is non-monotone, so an average "
        "shift effect would report the population's position relative to the "
        "optimum rather than the strength of the relationship."),
    "residual_norm": (
        "the u curve at M4 (curve_u.csv, step DSVF): adjusted rate by u decile "
        "at matched depth, operator presence, variance and faithfulness. No "
        "scalar, for the same reason and because u enters as indicators."),
    "faithfulness": (
        "beta_z_faith at M4 with its +1 SD marginal effect: relational "
        "faithfulness (Fisher-z, z-scored) at matched depth, operator "
        "presence, variance and u. The one geometry covariate whose linearity "
        "survives testing, and so the only one carrying a scalar."),
    "operators": (
        "joint has_op contrasts at M1 (operators.csv): per-operator presence "
        "contrast at matched depth and matched co-occurring operators, with no "
        "geometry term. The geometry covariates are computed downstream of the "
        "operator set and are 12-16% operator-determined, so these contrasts "
        "already contain whatever an operator does by shifting V, u and F: "
        "they are TOTAL, and not additive with the geometry readings."),
}

INFERENCE_NOTE = (
    "Confirmatory family reported in full with 95% percentile-bootstrap CIs; "
    "no multiplicity adjustment (declared exploratory; the family is small "
    "and pre-stated). Attribution trajectories and curves are secondary or "
    "descriptive.")

DESIGN_REVISION = (
    "Estimand family v2 (2026-07-29): rung lattice and four-member "
    "confirmatory family adopted at design stage on identification grounds "
    "(see ASSUMPTIONS.md), before the present pipeline produced numbers; "
    "supersedes the single-primary declaration (beta_u in M1) and the "
    "low-faithfulness tail contrast. Old-suite results were known at "
    "revision time. "
    "Amendment (2026-07-31): both curvature rungs, M2q (V^2) and M3q (u^2), "
    "were specified AFTER inspecting the variance- and u-decile curves "
    "respectively, and were tiered exploratory on that basis. Also at this "
    "revision, log_norm moved from natural log to log10 (decades); u and every "
    "coefficient are exactly base-invariant, so only the logged columns of "
    "norm_variance.csv change. "
    "Amendment v3 (2026-08-06), TWO CHANGES, both after the data were seen and "
    "both disclosed rather than absorbed. "
    "(1) SPECIFICATION. Linearity in the logit is rejected for z_variance "
    "(p = 2.9e-3) and u (p = 8.3e-6) and holds for z_faith (p = 0.41), so the "
    "linear specification is not reported as a result at all: V enters as a "
    "quadratic and u as decile indicators in every rung. The forms were chosen "
    "by spec_search.csv over a D + S base, on shape adequacy and not on any "
    "estimand's value -- and the change made the headline result WEAKER, which "
    "is the evidence that it was not outcome-driven. M2q and M3q are retired "
    "as rungs: they are now rows of the search. The rejected linear fits stay "
    "tabulated so the search can be audited. Intervals are conditional on the "
    "selected specification and do not account for the search (post-selection "
    "inference). "
    "(2) LATTICE ORDER. Syntax now sits at the base (M1 = D + S) rather than "
    "entering as an adjustment step, because C5 makes it computationally prior "
    "and a rung that reads a geometry quantity with the operator set open is "
    "not an estimand anyone would report. The lattice then forks symmetrically "
    "at M2, which also yields something the old chain could not: what u ALONE "
    "does to V (M2 -> M3u, C1 predicts ~nothing) against what F alone does "
    "(M2 -> M3F). The syntax-absorption comparison survives in the CURVE "
    "sequences, which start at raw and pass through D + S -- curves are "
    "descriptive, so their sequence may include steps the lattice does not. "
    "Consequence of (1): V and u lose their scalar readings entirely; their "
    "curves are the estimands. Q4's rung is renamed S -> M1; it is unchanged.")

TIERS = {
    "curve_z_variance.csv:DS": "confirmatory (Q1: the variance curve at M2 IS the estimand -- no scalar, the relationship is non-monotone)",
    "curve_u.csv:DSVF": "confirmatory (Q3: the u curve at M4 IS the estimand -- no scalar)",
    "m_ladder.csv:M4:z_faith": "confirmatory (Q2: the one geometry covariate whose linearity survives, so the only one with a scalar)",
    "marginal_effects.csv:M4:z_faith": "confirmatory (Q2 on the probability scale)",
    "operators.csv": "confirmatory-comparative (Q4: M1 joint operator contrasts, geometry path open; TOTAL contrasts, not additive with the geometry readings; not eight hypotheses)",
    "spec_search.csv": "primary methodological (chose the specification; read before any estimand -- see design_revision)",
    "curve_z_variance.csv:DSu": "secondary (what u alone does to the V curve; C1 predicts ~nothing)",
    "curve_z_variance.csv:DSF": "secondary, A STEP TOO FAR (F is computed downstream of V, so conditioning on it changes what the curve refers to; computed only to price the choice not to read there)",
    "curve_u.csv:DS,DSV": "secondary (u's attenuation sequence)",
    "curve_z_faith.csv": "secondary (F's attenuation sequence; DSVu is the step matching Q2's rung)",
    "m_ladder.csv:M0,M1,M2,M3u,M3F": "secondary (lattice scaffolding; the rungs the estimands are NOT read at)",
    "m_ladder.csv:attenuation": "secondary (F's single attenuation step; probability scale in marginal_effects.csv)",
    "m_ladder.csv:L-M2,L-M4": "rejected (the linear specification, tabulated so spec_search can be audited; NOT a result)",
    "adequacy.csv": "diagnostic (discrimination, link, influence -- the checks the score equations do not already enforce; calibration is omitted because they do)",
    "depth_curve.csv": "descriptive (raw + operator-standardised depth profile)",
    "occupancy.csv": "motivation exhibit (variance x norm) + positivity limit (z_variance x z_faith)",
    "norm_variance.csv": "Stage A descriptive",
    "norm_variance_stats.csv": "Stage A descriptive (why the raw norm cannot enter beside variance)",
    "covariates.csv": "Stage A descriptive (shape + leverage, before and after each transform)",
    "faith_by_variance.csv": "Stage A descriptive (C4's reliability channel)",
    "diagnostic.csv": "design diagnostic (identification + syntax/geometry entanglement)",
    "op_signature.csv": "Stage A descriptive (bridge)",
    "depth_op_mix.csv": "Stage A descriptive",
    "shuffle_null.csv": "falsification check",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features-dir", required=True, type=Path,
                   help="Output dir of compute_features.py (exp2_features.csv).")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Conditioned validation run dir (contains per_sample/).")
    p.add_argument("--shuffle-run-dir", type=Path, default=None,
                   help="Shuffle-ablation run dir for the guessability null.")
    p.add_argument("--dataset-dir", required=True, type=Path,
                   help="Validation dataset dir (formulas.jsonl + metadata.json).")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS,
                   help="Variance-quantile bins for the residualisation.")
    p.add_argument("--curve-bins", type=int, default=10,
                   help="Quantile bins for the descriptive curves.")
    p.add_argument("--bootstrap-samples", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- load + gates ------------------------------------------------------ #
    expected_n = read_dataset_size(args.dataset_dir)
    features = pd.read_csv(args.features_dir / "exp2_features.csv")
    formulas = load_formulas(args.dataset_dir / "formulas.jsonl")
    greedy, checks = load_greedy(args.run_dir, expected_n=expected_n)

    frame_raw = build_frame(features, greedy, formulas, expected_n=expected_n)
    print(f"[exp2] frame: {len(frame_raw)} targets, "
          f"correct rate {frame_raw['correct'].mean():.3f}", file=sys.stderr)

    dfc = derive_covariates(frame_raw, n_bins=args.n_bins)
    depth_levels = sorted(int(d) for d in frame_raw["depth"].unique())

    # ---- Stage A: model-free audit ---------------------------------------- #
    # Two occupancy grids doing two different jobs: variance x norm motivates
    # building u (the design supplies no norm contrast at fixed variance),
    # z_variance x z_faith records a positivity limit on the F readings (high F
    # never co-occurs with low V, and by C4 more data cannot fill that in).
    pd.concat([desc.occupancy(dfc, rows="variance", cols="emb_norm"),
               desc.occupancy(dfc, rows="z_variance", cols="z_faith")],
              ignore_index=True).to_csv(out / "occupancy.csv", index=False)
    desc.norm_variance_curve(dfc).to_csv(out / "norm_variance.csv", index=False)
    desc.norm_variance_stats(dfc).to_csv(out / "norm_variance_stats.csv",
                                         index=False)
    desc.covariate_stats(dfc).to_csv(out / "covariates.csv", index=False)
    desc.faith_by_variance(dfc).to_csv(out / "faith_by_variance.csv", index=False)
    desc.design_diagnostic(dfc).to_csv(out / "diagnostic.csv", index=False)
    desc.operator_signature(dfc).to_csv(out / "op_signature.csv", index=False)
    desc.depth_operator_mix(dfc).to_csv(out / "depth_op_mix.csv", index=False)

    if args.shuffle_run_dir is not None:
        shuffle_greedy, shuffle_checks = load_greedy(args.shuffle_run_dir,
                                                     expected_n=expected_n)
        checks += shuffle_checks
        desc.shuffle_null(dfc, shuffle_greedy).to_csv(out / "shuffle_null.csv",
                                                      index=False)

    # ---- specification search: run FIRST, it chose the models -------------- #
    models.spec_search(dfc).to_csv(out / "spec_search.csv", index=False)
    models.spec_curves(dfc).to_csv(out / "spec_curves.csv", index=False)
    models.model_adequacy(dfc).to_csv(out / "adequacy.csv", index=False)

    # ---- Stage B/C: point fits -------------------------------------------- #
    ladder = models.point_ladder(dfc)
    marginals = models.point_marginals(dfc)
    operators = models.point_operators(dfc)
    depth_curve = models.point_depth_curve(dfc)

    # The rejected linear specification, tabulated beside the search that
    # rejected it. Without this the search cannot be checked; it is never read
    # as a result and is tiered accordingly.
    y_pt = dfc["correct"].to_numpy(dtype=np.float64)
    rejected = []
    for name, terms in models.REJECTED_LINEAR.items():
        res = models._fit(y_pt, models._design(dfc, terms), cov_type="HC1")
        rejected += [{"model": name, "term": t, "estimate": float(res.params[t]),
                      "hc1_se": float(res.bse[t]), "ci_lo": np.nan,
                      "ci_hi": np.nan} for t in res.params.index]

    # Extra per-bin means so a figure can label an axis in interpretable units
    # while the bin positions stay on the scale the models actually use.
    CURVE_EXTRA = {"z_variance": ("variance",), "u": (),
                   "z_faith": ("relational_faithfulness",)}
    curves = {}
    for col, seq in models.CURVE_SEQ.items():
        frame = models.curve_descriptives(dfc, args.curve_bins, col=col,
                                          extra_means=CURVE_EXTRA[col])
        bins_np = frame["bin"].to_numpy()
        for step, adjust in seq:
            frame[f"adj_{step}"] = models.curve_rates(
                dfc, args.curve_bins, col=col, adjust=adjust)[bins_np]
        frame["primary_step"] = models.CURVE_PRIMARY[col]
        curves[col] = frame

    # ---- bootstrap (whole pipeline inside every resample) ------------------ #
    idx = index_matrix(len(frame_raw), b=args.bootstrap_samples, seed=args.seed)
    boot = models.bootstrap(frame_raw, idx, n_bins=args.n_bins,
                            curve_bins=args.curve_bins,
                            depth_levels=depth_levels)

    def _ci_cols(key: str) -> tuple[float, float]:
        lo, hi = models.ci(boot[key])
        return float(lo), float(hi)

    # ---- m_ladder: rung coefficients + attenuation trajectories ------------ #
    ladder["ci_lo"] = np.nan
    ladder["ci_hi"] = np.nan
    for model_name, terms in models.REPORTED_COEFS.items():
        for term in terms:
            lo, hi = _ci_cols(f"{model_name}_{term}")
            sel = (ladder["model"] == model_name) & (ladder["term"] == term)
            ladder.loc[sel, ["ci_lo", "ci_hi"]] = (lo, hi)

    def _coef(model_name: str, term: str) -> float:
        sel = ladder.query("model == @model_name and term == @term")
        return float(sel["estimate"].iloc[0])

    att_rows = []
    for term, steps in models.TRAJECTORIES.items():
        for a, b in steps:
            delta, ratio = models.attenuation(_coef(a, term), _coef(b, term))
            for stat, est in (("delta", delta), ("ratio", ratio)):
                lo, hi = _ci_cols(f"att_{term}_{a}_{b}_{stat}")
                att_rows.append({"model": f"attenuation_{a}_{b}",
                                 "term": f"{stat}_{term}",
                                 "estimate": est, "hc1_se": np.nan,
                                 "ci_lo": lo, "ci_hi": hi})
    ladder = pd.concat([ladder, pd.DataFrame(att_rows), pd.DataFrame(rejected)],
                       ignore_index=True)
    ladder.to_csv(out / "m_ladder.csv", index=False)

    # ---- marginal effects: per-rung AMEs + AME-scale trajectories ---------- #
    marginals["ci_lo"] = np.nan
    marginals["ci_hi"] = np.nan
    for model_name, terms in models.AME_TERMS.items():
        for term in terms:
            lo, hi = _ci_cols(f"ame_{model_name}_{term}")
            sel = (marginals["model"] == model_name) & (marginals["term"] == term)
            marginals.loc[sel, ["ci_lo", "ci_hi"]] = (lo, hi)

    def _ame(model_name: str, term: str) -> float:
        sel = marginals.query("model == @model_name and term == @term")
        return float(sel["estimate"].iloc[0])

    ame_att_rows = []
    for term, steps in models.TRAJECTORIES.items():
        for a, b in steps:
            delta, ratio = models.attenuation(_ame(a, term), _ame(b, term))
            for stat, est in (("delta", delta), ("ratio", ratio)):
                lo, hi = _ci_cols(f"ame_att_{term}_{a}_{b}_{stat}")
                ame_att_rows.append({"model": f"attenuation_{a}_{b}",
                                     "term": f"{stat}_ame_{term}",
                                     "estimate": est,
                                     "ci_lo": lo, "ci_hi": hi})
    marginals = pd.concat([marginals, pd.DataFrame(ame_att_rows)],
                          ignore_index=True)
    marginals.to_csv(out / "marginal_effects.csv", index=False)

    # ---- curves ------------------------------------------------------------ #
    # Every step of every sequence carries its own interval, and each step's
    # PAIRED difference from the primary is stored too. That difference is a
    # contrast on the same targets, so its interval is far tighter than either
    # step's marginal one -- quoting the marginals against an across-step
    # movement would understate the evidence, not overstate it.
    for col, frame in curves.items():
        bins_np = frame["bin"].to_numpy()
        primary = models.CURVE_PRIMARY[col]
        for step, _ in models.CURVE_SEQ[col]:
            lo, hi = models.ci(boot[f"curve_{col}_{step}"])
            frame[f"ci_lo_{step}"] = lo[bins_np]
            frame[f"ci_hi_{step}"] = hi[bins_np]
            if step != primary:
                d = boot[f"curve_{col}_{step}"] - boot[f"curve_{col}_{primary}"]
                lo, hi = models.ci(d)
                frame[f"vs_primary_{step}"] = (frame[f"adj_{step}"]
                                               - frame[f"adj_{primary}"])
                frame[f"vs_primary_ci_lo_{step}"] = lo[bins_np]
                frame[f"vs_primary_ci_hi_{step}"] = hi[bins_np]
        frame.to_csv(out / f"curve_{col}.csv", index=False)

    # ---- operators (M1) + depth profile ------------------------------------ #
    lo, hi = models.ci(boot["op_joint"])
    operators["log_odds_ci_lo"] = lo
    operators["log_odds_ci_hi"] = hi
    lo, hi = models.ci(boot["op_gap"])
    operators["gap_ci_lo"] = lo
    operators["gap_ci_hi"] = hi
    operators.to_csv(out / "operators.csv", index=False)

    pos = {d: k for k, d in enumerate(depth_levels)}
    kidx = depth_curve["depth"].map(pos).to_numpy()
    lo, hi = models.ci(boot["depth_raw"])
    depth_curve["raw_ci_lo"] = lo[kidx]
    depth_curve["raw_ci_hi"] = hi[kidx]
    lo, hi = models.ci(boot["depth_adj"])
    depth_curve["adj_ci_lo"] = lo[kidx]
    depth_curve["adj_ci_hi"] = hi[kidx]
    depth_curve.to_csv(out / "depth_curve.csv", index=False)

    # ---- checks + manifest ------------------------------------------------- #
    checks_df = pd.DataFrame(checks) if checks else pd.DataFrame(
        columns=["source", "check", "violations"])
    checks_df.to_csv(out / "checks.csv", index=False)
    n_violations = int(checks_df["violations"].sum()) if len(checks_df) else 0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(args.features_dir),
        "run_dir": str(args.run_dir),
        "shuffle_run_dir": (str(args.shuffle_run_dir)
                            if args.shuffle_run_dir else None),
        "dataset_dir": str(args.dataset_dir),
        "n_targets": len(frame_raw),
        "depth_levels": depth_levels,
        "frozen": {"n_bins": args.n_bins, "curve_bins": args.curve_bins,
                   "faith_clip": FAITH_CLIP,
                   "bootstrap_samples": args.bootstrap_samples,
                   "seed": args.seed},
        "estimands": ESTIMANDS,
        "inference_note": INFERENCE_NOTE,
        "design_revision": DESIGN_REVISION,
        "tiers": TIERS,
        "assumptions_doc": "scripts/analysis_exp2/ASSUMPTIONS.md",
        "operators": list(OPERATORS),
        "bootstrap_failures": boot["failures"],
        "check_violations": n_violations,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[exp2] tables written -> {out}", file=sys.stderr)
    if n_violations:
        print(f"[exp2] FAIL: {n_violations} consistency violations "
              f"(see checks.csv)", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
