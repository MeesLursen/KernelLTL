"""Experiment 2 analysis driver: cached features + validation JSONL -> tables.

Consumes the feature extraction of compute_features.py, the validation
formulas, and the greedy per-generation records of the conditioned run
(plus, optionally, the shuffle-ablation run for the guessability null).
Emits tidy CSV tables -- everything the local data-viz layer and the thesis
need, and nothing else. No plotting here.

Outputs (under --output-dir):
    occupancy.csv       variance x norm tercile counts        (motivation)
    norm_variance.csv   binned E[log_norm | variance] curve   (Stage A)
    faithfulness.csv    distribution + leverage stats         (Stage A)
    faith_grid.csv      mean faithfulness per var x u cell    (Stage A)
    diagnostic.csv      R2 of u on operator features          (methods text)
    op_signature.csv    mean u by operator presence           (Stage A bridge)
    shuffle_null.csv    chance-level equivalence rates        (check; optional)
    m_ladder.csv        M0/M1/contrast/M2 + attenuation rows  (Stage B/C)
    curve.csv           u-decile depth-adjusted correctness   (descriptive)
    faith_contrast.csv  least-faithful-tail gap               (descriptive)
    operators.csv       per-operator H2c log-odds             (Stage C)
    checks.csv          gate outcomes
    manifest.json       inputs, frozen constants, tier map, estimand

Tier convention (recorded in the manifest): the single primary test is
beta_u in M1; everything else is secondary, descriptive, or diagnostic.

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
from frame import (DEFAULT_FAITH_TAIL, DEFAULT_N_BINS, OPERATORS,        # noqa: E402
                   build_frame, derive_covariates, load_formulas)

ESTIMAND = ("Among validation targets of matched variance and depth, those whose "
            "embedding norm sits low relative to variance-matched peers "
            "(studentised log-residual u) show a lower greedy semantic-equivalence "
            "rate; beta_u in M1 is the single primary test and the association is "
            "observational.")

TIERS = {
    "m_ladder.csv:M1:u": "primary",
    "m_ladder.csv:M1:z_variance": "secondary",
    "m_ladder.csv:M2": "secondary (attribution; attenuation of beta_u)",
    "m_ladder.csv:attenuation": "secondary",
    "faith_contrast.csv": "descriptive (outcome-protocol sentences pre-written)",
    "curve.csv": "descriptive (shape exhibit)",
    "operators.csv": "descriptive (H2c decomposition, not eight hypotheses)",
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
                   help="u-quantile bins for the descriptive curve.")
    p.add_argument("--faith-tail", type=float, default=DEFAULT_FAITH_TAIL,
                   help="Low-faithfulness tail fraction for the contrast.")
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

    dfc = derive_covariates(frame_raw, n_bins=args.n_bins,
                            faith_tail=args.faith_tail)

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
    h2c = models.point_h2c(dfc)
    contrast = models.point_contrast(dfc)
    curve = models.curve_descriptives(dfc, args.curve_bins)
    curve["adj_rate"] = models.curve_rates(dfc, args.curve_bins)[
        curve["bin"].to_numpy()]

    # ---- bootstrap (whole pipeline inside every resample) ------------------ #
    idx = index_matrix(len(frame_raw), b=args.bootstrap_samples, seed=args.seed)
    boot = models.bootstrap(frame_raw, idx, n_bins=args.n_bins,
                            faith_tail=args.faith_tail,
                            curve_bins=args.curve_bins)

    def _ci_cols(key: str) -> tuple[float, float]:
        lo, hi = models.ci(boot[key])
        return float(lo), float(hi)

    ladder["ci_lo"] = np.nan
    ladder["ci_hi"] = np.nan
    for model_name, term in [("M1", "z_variance"), ("M1", "u"),
                             ("contrast", "low_faith"), ("contrast", "u"),
                             ("M2", "z_variance"), ("M2", "u")]:
        key = f"{model_name}_{term}"
        lo, hi = _ci_cols(key)
        sel = (ladder["model"] == model_name) & (ladder["term"] == term)
        ladder.loc[sel, ["ci_lo", "ci_hi"]] = (lo, hi)

    atten_rows = []
    m1_u = ladder.query("model == 'M1' and term == 'u'")["estimate"].iloc[0]
    m2_u = ladder.query("model == 'M2' and term == 'u'")["estimate"].iloc[0]
    for term, est, key in [
            ("delta_beta_u", m1_u - m2_u, "atten_delta"),
            ("ratio", 1.0 - m2_u / m1_u if abs(m1_u) > 1e-8 else np.nan,
             "atten_ratio")]:
        lo, hi = _ci_cols(key)
        atten_rows.append({"model": "attenuation", "term": term,
                           "estimate": est, "hc1_se": np.nan,
                           "ci_lo": lo, "ci_hi": hi})
    ladder = pd.concat([ladder, pd.DataFrame(atten_rows)], ignore_index=True)
    ladder.to_csv(out / "m_ladder.csv", index=False)

    lo, hi = models.ci(boot["curve"])
    curve["ci_lo"] = lo[curve["bin"].to_numpy()]
    curve["ci_hi"] = hi[curve["bin"].to_numpy()]
    curve.to_csv(out / "curve.csv", index=False)

    gap_lo, gap_hi = _ci_cols("contrast_gap")
    contrast.update({"adj_gap_ci_lo": gap_lo, "adj_gap_ci_hi": gap_hi})
    pd.DataFrame([contrast]).to_csv(out / "faith_contrast.csv", index=False)

    lo, hi = models.ci(boot["h2c"])
    h2c["ci_lo"] = lo
    h2c["ci_hi"] = hi
    h2c.to_csv(out / "operators.csv", index=False)

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
        "frozen": {"n_bins": args.n_bins, "curve_bins": args.curve_bins,
                   "faith_tail": args.faith_tail,
                   "faith_cut_value": contrast["faith_cut"],
                   "bootstrap_samples": args.bootstrap_samples,
                   "seed": args.seed},
        "estimand": ESTIMAND,
        "tiers": TIERS,
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
