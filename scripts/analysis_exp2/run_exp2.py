"""Experiment 2 analysis driver: cached features + validation JSONL -> tables.

Consumes the feature extraction of compute_features.py, the validation
formulas, and the greedy per-generation records of the conditioned run
(plus, optionally, the shuffle-ablation run for the guessability null).
Emits tidy CSV tables -- everything the local data-viz layer and the thesis
need, and nothing else. No plotting here.

Outputs (under --output-dir):
    occupancy.csv         variance x norm tercile counts        (motivation)
    norm_variance.csv     binned E[log_norm | variance] curve   (Stage A)
    faithfulness.csv      distribution + leverage stats         (Stage A)
    faith_grid.csv        mean faithfulness per var x u cell    (Stage A)
    diagnostic.csv        R2 of u on operator features          (methods text)
    op_signature.csv      mean u by operator presence           (Stage A bridge)
    shuffle_null.csv      chance-level equivalence rates        (check; optional)
    m_ladder.csv          the rung lattice M0/M1/M2/F1/F2/M3/   (Stage B/C)
                          M3q + attenuation trajectories
    marginal_effects.csv  +1 SD probability-scale effects per   (Stage B/C;
                          rung + AME-scale trajectories          cross-rung
                                                                 comparisons)
    curve.csv             u-decile depth-adjusted correctness   (descriptive;
                                                                 Q3 companion)
    var_curve.csv         variance-decile depth-adjusted        (descriptive;
                          correctness                            Q1 companion)
    faith_curve.csv       faithfulness-decile correctness,      (descriptive;
                          depth-adjusted (adj_rate) and          Q2 companion)
                          depth+variance-adjusted (adj_rate_vd)
    operators.csv         S rung: per-operator joint (adjusted) (Stage C;
                          and single (confounded companion)      Q4)
    depth_curve.csv       raw + operator-standardised           (descriptive)
                          correctness per depth
    checks.csv            gate outcomes
    manifest.json         inputs, frozen constants, tier map, estimand family

Confirmatory family (reported in full, 95% percentile-bootstrap CIs, NO
multiplicity adjustment -- declared exploratory): beta_z_variance @ M2,
beta_u @ M3, beta_z_faith @ M3, plus the comparative joint operator
contrasts @ the S rung. The estimand each rung identifies, and the
assumptions identification rests on, are documented in ASSUMPTIONS.md next
to this script; the manifest carries a pointer and the design-revision note.

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

ESTIMANDS = {
    "variance_total": (
        "beta_z_variance in M2: total effect of satisfaction variance on the "
        "greedy semantic-equivalence rate among validation targets, at matched "
        "depth and operator presence; faithfulness deliberately excluded "
        "(mediator of variance). Causal reading under A2 + A3 (ASSUMPTIONS.md)."),
    "residual_norm": (
        "beta_u in M3: effect of the studentised variance-binned log-norm "
        "residual, at matched depth, variance, operator presence, and "
        "faithfulness. Causal reading under A1 + A3."),
    "faithfulness": (
        "beta_z_faith in M3: effect of relational faithfulness (Fisher-z), at "
        "matched depth, variance, operator presence, and u. Causal reading "
        "under A1 + A3."),
    "operators": (
        "joint has_op contrasts at the S rung (operators.csv): per-operator "
        "presence contrast at matched depth and matched co-occurring "
        "operators; comparative-associational, geometry covariates excluded "
        "so the embedding-geometry path stays open (total contrasts)."),
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
    "revision time.")

TIERS = {
    "m_ladder.csv:M2:z_variance": "confirmatory (Q1: total effect of variance; causal under A2+A3)",
    "m_ladder.csv:M3:u": "confirmatory (Q3: residual-norm effect; causal under A1+A3)",
    "m_ladder.csv:M3:z_faith": "confirmatory (Q2: faithfulness effect; causal under A1+A3)",
    "operators.csv": "confirmatory-comparative (Q4: S-rung joint operator contrasts, geometry path open; single = confounded companion; not eight hypotheses)",
    "m_ladder.csv:M1": "secondary (u-branch start: minimal-adjustment association)",
    "m_ladder.csv:F1": "secondary (F-branch start: minimal-adjustment association)",
    "m_ladder.csv:F2": "secondary (F-branch syntax-absorption step)",
    "m_ladder.csv:attenuation": "secondary (attribution trajectories along the lattice; AME scale in marginal_effects.csv)",
    "m_ladder.csv:M3q": "exploratory (curvature at the Q3 rung; motivated by the u-decile curve)",
    "marginal_effects.csv": "secondary (probability-scale effects; the scale for cross-rung comparisons)",
    "curve.csv": "descriptive (Q3 companion: u-decile shape exhibit)",
    "var_curve.csv": "descriptive (Q1 companion: variance-decile shape exhibit)",
    "faith_curve.csv": "descriptive (Q2 companion: faithfulness-decile curves, depth- and depth+variance-adjusted)",
    "depth_curve.csv": "descriptive (raw + operator-standardised depth profile)",
    "occupancy.csv": "motivation exhibit",
    "norm_variance.csv": "Stage A descriptive",
    "faithfulness.csv": "Stage A descriptive",
    "faith_grid.csv": "Stage A descriptive",
    "diagnostic.csv": "design diagnostic (methods text)",
    "op_signature.csv": "Stage A descriptive (bridge)",
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
    p.add_argument("--var-adjust-bins", type=int,
                   default=models.DEFAULT_VAR_ADJUST_BINS,
                   help="Variance-quantile bins adjusting the vd faith curve.")
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
    desc.occupancy(dfc).to_csv(out / "occupancy.csv", index=False)
    desc.norm_variance_curve(dfc).to_csv(out / "norm_variance.csv", index=False)
    desc.faithfulness_stats(dfc).to_csv(out / "faithfulness.csv", index=False)
    desc.faith_grid(dfc).to_csv(out / "faith_grid.csv", index=False)
    desc.design_diagnostic(dfc).to_csv(out / "diagnostic.csv", index=False)
    desc.operator_signature(dfc).to_csv(out / "op_signature.csv", index=False)

    if args.shuffle_run_dir is not None:
        shuffle_greedy, shuffle_checks = load_greedy(args.shuffle_run_dir,
                                                     expected_n=expected_n)
        checks += shuffle_checks
        desc.shuffle_null(dfc, shuffle_greedy).to_csv(out / "shuffle_null.csv",
                                                      index=False)

    # ---- Stage B/C: point fits -------------------------------------------- #
    ladder = models.point_ladder(dfc)
    marginals = models.point_marginals(dfc)
    operators = models.point_operators(dfc)
    depth_curve = models.point_depth_curve(dfc)

    def _curve_frame(col: str, label: str, **kw) -> pd.DataFrame:
        frame = models.curve_descriptives(dfc, args.curve_bins, col=col,
                                          label=label)
        frame["adj_rate"] = models.curve_rates(
            dfc, args.curve_bins, col=col, **kw)[frame["bin"].to_numpy()]
        return frame

    curve = _curve_frame("u", "u")
    var_curve = _curve_frame("variance", "variance")
    faith_curve = _curve_frame("relational_faithfulness", "faith")

    # ---- bootstrap (whole pipeline inside every resample) ------------------ #
    idx = index_matrix(len(frame_raw), b=args.bootstrap_samples, seed=args.seed)
    boot = models.bootstrap(frame_raw, idx, n_bins=args.n_bins,
                            curve_bins=args.curve_bins,
                            depth_levels=depth_levels,
                            var_adjust_bins=args.var_adjust_bins)

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
    ladder = pd.concat([ladder, pd.DataFrame(att_rows)], ignore_index=True)
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
    for frame, key, path in ((curve, "curve", "curve.csv"),
                             (var_curve, "var_curve", "var_curve.csv"),
                             (faith_curve, "faith_curve", "faith_curve.csv")):
        lo, hi = models.ci(boot[key])
        bins_np = frame["bin"].to_numpy()
        frame["ci_lo"] = lo[bins_np]
        frame["ci_hi"] = hi[bins_np]
        if key != "faith_curve":
            frame.to_csv(out / path, index=False)

    # Faithfulness vd variant: depth+variance-adjusted -- the exhibit matched
    # to the u curve's implicit adjustment (u is variance-residualised by
    # construction). Neither geometry curve is adjusted for the other,
    # deliberately (shared-latent selection distortion).
    bins_np = faith_curve["bin"].to_numpy()
    faith_curve["adj_rate_vd"] = models.curve_rates(
        dfc, args.curve_bins, col="relational_faithfulness",
        var_bins=args.var_adjust_bins)[bins_np]
    lo, hi = models.ci(boot["faith_curve_vd"])
    faith_curve["vd_ci_lo"] = lo[bins_np]
    faith_curve["vd_ci_hi"] = hi[bins_np]
    faith_curve.to_csv(out / "faith_curve.csv", index=False)

    # ---- operators (S rung) + depth profile -------------------------------- #
    lo, hi = models.ci(boot["op_joint"])
    operators["joint_ci_lo"] = lo
    operators["joint_ci_hi"] = hi
    lo, hi = models.ci(boot["op_gap"])
    operators["gap_ci_lo"] = lo
    operators["gap_ci_hi"] = hi
    lo, hi = models.ci(boot["op_single"])
    operators["single_ci_lo"] = lo
    operators["single_ci_hi"] = hi
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
                   "var_adjust_bins": args.var_adjust_bins,
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
