"""Rebuild the curriculum-adjacent datasets against a regenerated kernel, with
semantics-level (satvec) disjointness and reproducible embeddings.

Pipeline (each phase is one resumable *object*; a phase is skipped if its output is
already complete -- a saved dataset's `metadata.json`, or an exclusion source's `.npy`):

  1. recompute_curriculum : re-project each curriculum-train stage's stored satvecs
                            through the new feature matrix F (formulas untouched).
  2. build_finetune       : sample N unique, non-trivial, >=2-equivalent-rewrite base
                            formulas from stage4 train (without replacement); emit 2
                            distinct equivalent rewrites + 1 random negation insertion
                            per base; evaluate satvecs and embeddings.
  3. build_exclusion      : hash the satvecs of (curriculum-train stages U finetune) into
                            a device-independent semantic exclusion set (one file/source).
  4. fill_pool            : seed depth bins {2..5} from stage4 eval (non-trivial, satvec
                            not excluded, satvec-unique), then reverse-order top-up (fill
                            depth 5, ratchet max_depth down to the highest unfilled bin,
                            repeat) until each bin hits its target or the attempt budget.
  5. split_and_derive     : depth-balanced split of the pool into `validation` and the new
                            `stage4/eval` (depth-stacked), then derive stage1..3 eval by
                            dropping the highest depths.

Reproducibility: all embeddings are computed via the exact covariance form
(`LTLKernel.compute_embeddings_from_satisfactions`), so they are bit-identical on any
IEEE-754 hardware. The traces of the regenerated kernel must be the same traces the stored
satvecs were computed on (guaranteed if the kernel came from `regenerate_kernel.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_class import LTLDataset
from formula_class import Formula
from formula_utils import (
    add_random_negation,
    list_semantically_equivalent_transformations,
    sample_formulas,
)
from kernel_class import LTLKernel

DEPTHS = (2, 3, 4, 5)


# --------------------------- small helpers ---------------------------
def _hash_sat(sat_bool_cpu: torch.Tensor) -> bytes:
    """Device-independent 16-byte semantic key of a boolean satvec (bit-packed)."""
    packed = np.packbits(sat_bool_cpu.to(torch.uint8).numpy())
    return hashlib.blake2b(packed.tobytes(), digest_size=16).digest()


def _is_trivial(sat_bool: torch.Tensor) -> bool:
    """True for tautology/contradiction (constant satvec, std == 0)."""
    return bool(torch.all(sat_bool)) or not bool(torch.any(sat_bool))


def _done(dirpath: str) -> bool:
    return os.path.exists(os.path.join(dirpath, "metadata.json"))


def _save_dataset(
    dirpath: str,
    formulas: list[Formula],
    embeddings: torch.Tensor,
    satvecs: torch.Tensor | None,
    extra_meta: dict,
    sat_batch: int,
) -> None:
    ds = LTLDataset(
        store_formula_str=True,
        store_satisfaction=satvecs is not None,
        satisfaction_batch_size=sat_batch,
        satisfaction_time_index=0,
    )
    ds._reset_storage()
    ds.formulas = list(formulas)
    ds.formula_strs = [str(f) for f in formulas]
    ds.embeddings = embeddings.to(dtype=torch.float32, device="cpu")
    ds.satisfactions = None if satvecs is None else satvecs.to(dtype=torch.bool, device="cpu")
    ds.metadata = extra_meta
    ds.save(dirpath)


def _project_satvecs(kernel: LTLKernel, satvecs: torch.Tensor, embed_batch: int, device: str) -> torch.Tensor:
    """Batched exact projection of (n, N) satvecs -> (n, m) float32 embeddings on CPU."""
    n = satvecs.size(0)
    out = torch.empty((n, kernel.m), dtype=torch.float32)
    for i in range(0, n, embed_batch):
        chunk = satvecs[i : i + embed_batch].to(device)
        out[i : i + embed_batch] = kernel.compute_embeddings_from_satisfactions(chunk, move_to_cpu=False).cpu()
    return out


def _stage_out_name(stage_dir: str) -> str:
    parts = stage_dir.rstrip("/").split(os.sep)
    return os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]


# --------------------------- phase 1: curriculum ---------------------------
def phase_recompute_curriculum(kernel, stage_dirs, out_root, embed_batch, sat_batch, device):
    for stage_dir in stage_dirs:
        out_dir = os.path.join(out_root, "curriculum", _stage_out_name(stage_dir))
        if _done(out_dir):
            print(f"[curriculum] skip (done): {out_dir}")
            continue
        print(f"[curriculum] recomputing embeddings for {stage_dir} -> {out_dir}")
        ds = LTLDataset.load(stage_dir, load_satisfactions=True, satisfactions_mmap=True)
        if ds.satisfactions is not None:
            emb = _project_satvecs(kernel, ds.satisfactions, embed_batch, device)
        else:
            print("           (no stored satvecs; evaluating formulas on traces -- slower)")
            emb = torch.empty((len(ds.formulas), kernel.m), dtype=torch.float32)
            for i, phi in enumerate(ds.formulas):
                sats = kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch)
                emb[i] = kernel.compute_embedding_from_satisfaction(sats, move_to_cpu=True)
        meta = {"source": "curriculum_recompute", "from": stage_dir, "count": len(ds.formulas)}
        _save_dataset(out_dir, ds.formulas, emb, satvecs=None, extra_meta=meta, sat_batch=sat_batch)
        print(f"[curriculum] saved {len(ds.formulas)} rows -> {out_dir}")


# --------------------------- phase 2: finetune ---------------------------
def _sample_finetune_bases(stage4_train_dir, sample_count, rng):
    """Draw `sample_count` unique, non-trivial base formulas with >=2 equivalent rewrites,
    without replacement, from stage4 train. Cheap AST test first, satvec (trivial) test only
    for AST-passing candidates -- so we never read a satvec we don't need."""
    ds = LTLDataset.load(stage4_train_dir, load_satisfactions=True, satisfactions_mmap=True)
    if ds.satisfactions is None:
        raise ValueError(f"{stage4_train_dir} has no stored satvecs; needed for trivial filtering.")
    n = len(ds.formulas)
    perm = torch.randperm(n, generator=rng).tolist()
    bases: list[Formula] = []
    seen: set[str] = set()
    scanned = 0
    for idx in perm:
        scanned += 1
        phi = ds.formulas[idx]
        s = str(phi)
        if s in seen:
            continue
        if len(list_semantically_equivalent_transformations(phi)) < 2:  # cheap AST test first
            continue
        if _is_trivial(ds.satisfactions[idx]):  # satvec read only for AST-passing candidates
            continue
        seen.add(s)
        bases.append(phi)
        if len(bases) >= sample_count:
            break
    if len(bases) < sample_count:
        raise ValueError(
            f"Only {len(bases)} eligible bases found in {n} formulas (scanned {scanned}); "
            f"requested {sample_count}."
        )
    print(f"[finetune] selected {len(bases)} bases (scanned {scanned} of {n}).")
    return bases


def _mutate_base(phi, rng):
    """Two distinct equivalent rewrites + one random negation insertion."""
    out: list[Formula] = []
    equivalents = list_semantically_equivalent_transformations(phi)
    pick = torch.randperm(len(equivalents), generator=rng)[:2].tolist()
    out.extend(equivalents[i] for i in pick)
    out.append(add_random_negation(phi, rng))
    return out


def phase_build_finetune(kernel, stage4_train_dir, out_dir, sample_count, sat_batch, embed_batch, rng, device):
    if _done(out_dir):
        print(f"[finetune] skip (done): {out_dir}")
        return
    bases = _sample_finetune_bases(stage4_train_dir, sample_count, rng)

    mutated: list[Formula] = []
    seen: set[str] = set()
    for phi in bases:
        for mut in _mutate_base(phi, rng):
            s = str(mut)
            if s not in seen:
                seen.add(s)
                mutated.append(mut)
    print(f"[finetune] {len(mutated)} unique mutated formulas from {len(bases)} bases; evaluating...")

    N = kernel.traces.size(0)
    satvecs = torch.empty((len(mutated), N), dtype=torch.bool)
    for i, phi in enumerate(mutated):
        satvecs[i] = kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu()
        if (i + 1) % 5000 == 0:
            print(f"           evaluated {i + 1}/{len(mutated)} satvecs")
    emb = _project_satvecs(kernel, satvecs, embed_batch, device)

    meta = {
        "source": "finetune_mutation_v2",
        "stage4_train_dir": stage4_train_dir,
        "base_count": len(bases),
        "mutated_count": len(mutated),
        "equivalent_rewrites_per_base": 2,
        "negation_insertions_per_base": 1,
    }
    _save_dataset(out_dir, mutated, emb, satvecs=satvecs, extra_meta=meta, sat_batch=sat_batch)
    print(f"[finetune] saved {len(mutated)} rows -> {out_dir}")


# --------------------------- phase 3: exclusion set ---------------------------
def _hash_dataset_satvecs(dataset_dir, out_npy):
    if os.path.exists(out_npy):
        print(f"[exclusion] skip (done): {out_npy}")
        return
    ds = LTLDataset.load(dataset_dir, load_satisfactions=True, satisfactions_mmap=True)
    if ds.satisfactions is None:
        raise ValueError(f"{dataset_dir} has no stored satvecs to hash.")
    n = ds.satisfactions.size(0)
    keys = np.empty((n, 16), dtype=np.uint8)
    for i in range(n):
        keys[i] = np.frombuffer(_hash_sat(ds.satisfactions[i]), dtype=np.uint8)
        if (i + 1) % 100000 == 0:
            print(f"           hashed {i + 1}/{n} satvecs of {dataset_dir}")
    tmp = out_npy + ".tmp.npy"
    np.save(tmp, keys)
    os.replace(tmp, out_npy)
    print(f"[exclusion] hashed {n} satvecs -> {out_npy}")


def phase_build_exclusion(stage_dirs, finetune_dir, hashes_dir):
    os.makedirs(hashes_dir, exist_ok=True)
    sources = [(d, os.path.join(hashes_dir, _stage_out_name(d).replace(os.sep, "_") + ".npy")) for d in stage_dirs]
    sources.append((finetune_dir, os.path.join(hashes_dir, "finetune.npy")))
    for src, npy in sources:
        _hash_dataset_satvecs(src, npy)
    exclusion: set[bytes] = set()
    for _, npy in sources:
        keys = np.load(npy)
        exclusion.update(keys[i].tobytes() for i in range(keys.shape[0]))
    print(f"[exclusion] union semantic exclusion set: {len(exclusion)} unique satvec keys.")
    return exclusion


# --------------------------- phase 4: fill pool ---------------------------
def phase_fill_pool(kernel, stage4_eval_dir, exclusion, pool_dir, target, p_leaf_range,
                    sat_batch, sample_batch, attempt_budget, seed, device):
    if _done(pool_dir):
        print(f"[pool] skip (done): {pool_dir}")
        return
    kernel.rng.manual_seed(int(seed))  # reproducible candidate stream, independent of kernel history
    N = kernel.traces.size(0)
    bins: dict[int, list[tuple[Formula, torch.Tensor, torch.Tensor]]] = {d: [] for d in DEPTHS}
    pool_hashes: set[bytes] = set()

    def _consider(phi, max_depth_cap) -> bool:
        d = phi.depth()
        if d not in bins or len(bins[d]) >= target or d > max_depth_cap:
            return False
        sats = kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu()
        if _is_trivial(sats):
            return False
        h = _hash_sat(sats)
        if h in exclusion or h in pool_hashes:
            return False
        emb = kernel.compute_embedding_from_satisfaction(sats.to(device), move_to_cpu=True)
        bins[d].append((phi, sats, emb))
        pool_hashes.add(h)
        return True

    # --- seed from stage4 eval (respect all rejection rules) ---
    seed_ds = LTLDataset.load(stage4_eval_dir, load_satisfactions=False)
    seeded = 0
    for phi in seed_ds.formulas:
        if _consider(phi, max_depth_cap=max(DEPTHS)):
            seeded += 1
    print(f"[pool] seeded {seeded} formulas from stage4 eval; per-bin {{d: len}} = "
          f"{ {d: len(bins[d]) for d in DEPTHS} }")

    # --- reverse-order top-up: fill highest bin, ratchet max_depth down ---
    for target_depth in sorted(DEPTHS, reverse=True):
        attempts = 0
        while len(bins[target_depth]) < target and attempts < attempt_budget:
            candidates = sample_formulas(
                n_formula=sample_batch, p_leaf_range=p_leaf_range, max_depth=target_depth,
                n_ap=kernel.AP, force_tree=False, rng=kernel.rng, device=kernel.device,
            )
            for phi in candidates:
                _consider(phi, max_depth_cap=target_depth)  # opportunistically fills bins <= target_depth
            attempts += sample_batch
        print(f"[pool] after target depth {target_depth} (max_depth={target_depth}, attempts={attempts}): "
              f"{ {d: len(bins[d]) for d in DEPTHS} }")

    # --- persist the pool as one object ---
    formulas, sat_rows, emb_rows = [], [], []
    for d in DEPTHS:
        for phi, sats, emb in bins[d]:
            formulas.append(phi); sat_rows.append(sats); emb_rows.append(emb)
    satvecs = torch.stack(sat_rows) if sat_rows else torch.empty((0, N), dtype=torch.bool)
    emb = torch.stack(emb_rows) if emb_rows else torch.empty((0, kernel.m), dtype=torch.float32)
    meta = {"source": "pool", "target_per_bin": target,
            "achieved_per_bin": {str(d): len(bins[d]) for d in DEPTHS}}
    _save_dataset(pool_dir, formulas, emb, satvecs=satvecs, extra_meta=meta, sat_batch=sat_batch)
    print(f"[pool] saved pool ({len(formulas)} formulas) -> {pool_dir}")


# --------------------------- phase 5: split + derive ---------------------------
def phase_split_and_derive(pool_dir, out_root, sat_batch, rng):
    val_dir = os.path.join(out_root, "validation")
    eval_dir = os.path.join(out_root, "stage4", "eval")
    if _done(val_dir) and _done(eval_dir):
        print("[split] skip (done): validation + stage4/eval")
    else:
        ds = LTLDataset.load(pool_dir, load_satisfactions=True, satisfactions_mmap=True)
        by_depth: dict[int, list[int]] = {d: [] for d in DEPTHS}
        for i, phi in enumerate(ds.formulas):
            d = phi.depth()
            if d in by_depth:
                by_depth[d].append(i)
        half = min(len(by_depth[d]) // 2 for d in DEPTHS)
        print(f"[split] balanced half per depth = {half} "
              f"(available {{d: n}} = { {d: len(by_depth[d]) for d in DEPTHS} })")

        val_idx, eval_idx = [], []
        for d in DEPTHS:  # depth-stacked ordering: all depth-2, then depth-3, ...
            order = [by_depth[d][i] for i in torch.randperm(len(by_depth[d]), generator=rng).tolist()]
            val_idx.extend(order[:half])
            eval_idx.extend(order[half : 2 * half])

        _write_subset(ds, val_idx, val_dir, "validation", sat_batch, half)
        _write_subset(ds, eval_idx, eval_dir, "stage4_eval", sat_batch, half)

    # derive stage1..3 eval by dropping the highest depths (eval is depth-stacked, `half` per depth)
    eval_ds = LTLDataset.load(eval_dir, load_satisfactions=True, satisfactions_mmap=True)
    half = len(eval_ds.formulas) // len(DEPTHS)
    for keep_max, stage in [(4, "stage3"), (3, "stage2"), (2, "stage1")]:
        out_dir = os.path.join(out_root, stage, "eval")
        if _done(out_dir):
            print(f"[split] skip (done): {out_dir}")
            continue
        keep = [d for d in DEPTHS if d <= keep_max]
        idx = list(range(len(keep) * half))  # first blocks are the lowest depths
        _write_subset(eval_ds, idx, out_dir, f"{stage}_eval_from_stage4", sat_batch, half)


def _write_subset(ds, idx, out_dir, source, sat_batch, half):
    formulas = [ds.formulas[i] for i in idx]
    satvecs = ds.satisfactions[torch.tensor(idx, dtype=torch.long)] if ds.satisfactions is not None else None
    emb = ds.embeddings[torch.tensor(idx, dtype=torch.long)]
    meta = {"source": source, "count": len(idx), "per_depth": half}
    _save_dataset(out_dir, formulas, emb, satvecs, meta, sat_batch)
    print(f"[split] wrote {len(idx)} rows -> {out_dir}")


# --------------------------- main ---------------------------
def _positive_int(v):
    iv = int(v)
    if iv <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return iv


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild curriculum/finetune/eval/validation datasets against a regenerated kernel.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kernel-dir", required=True, help="Regenerated kernel (traces must match stored satvecs)")
    p.add_argument("--stage-train-dirs", nargs="+", required=True, help="Curriculum train dirs (with stored satvecs)")
    p.add_argument("--stage4-train-dir", required=True, help="Stage4 train dir for finetune base sampling")
    p.add_argument("--stage4-eval-dir", required=True, help="Existing stage4 eval dir to seed depth bins")
    p.add_argument("--output-root", required=True, help="Root for all rebuilt datasets")
    p.add_argument("--device", default=None)
    p.add_argument("--finetune-sample-count", type=_positive_int, default=30000)
    p.add_argument("--bin-target", type=_positive_int, default=10000, help="Target formulas per depth bin")
    p.add_argument("--p-leaf-range", nargs=2, type=float, default=[0.1, 0.5])
    p.add_argument("--sample-batch", type=_positive_int, default=51200, help="Candidate draws per sampling round")
    p.add_argument("--attempt-budget", type=_positive_int, default=50_000_000, help="Max candidates drawn per target depth")
    p.add_argument("--satisfaction-batch-size", type=_positive_int, default=10240)
    p.add_argument("--embed-batch", type=_positive_int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    kernel = LTLKernel.load(args.kernel_dir, device=args.device)
    if kernel.traces is None or kernel.F is None:
        raise ValueError("Kernel must have both traces and F.")
    device = kernel.device
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))
    os.makedirs(args.output_root, exist_ok=True)

    phase_recompute_curriculum(kernel, args.stage_train_dirs, args.output_root,
                               args.embed_batch, args.satisfaction_batch_size, device)

    finetune_dir = os.path.join(args.output_root, "finetune", "train")
    phase_build_finetune(kernel, args.stage4_train_dir, finetune_dir, args.finetune_sample_count,
                         args.satisfaction_batch_size, args.embed_batch, rng, device)

    exclusion = phase_build_exclusion(args.stage_train_dirs, finetune_dir,
                                      os.path.join(args.output_root, "_hashes"))

    pool_dir = os.path.join(args.output_root, "_pool")
    phase_fill_pool(kernel, args.stage4_eval_dir, exclusion, pool_dir, args.bin_target,
                    tuple(args.p_leaf_range), args.satisfaction_batch_size, args.sample_batch,
                    args.attempt_budget, args.seed, device)

    phase_split_and_derive(pool_dir, args.output_root, args.satisfaction_batch_size, rng)
    print("[done] all phases complete.")


if __name__ == "__main__":
    main()
