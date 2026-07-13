"""Depth-graded rebuild of curriculum / eval / validation / finetune datasets.

Replaces the depth-*bin* approach of rebuild_datasets.py. The depth-2 semantic space
is finite (all forms enumerable), so it is partitioned deliberately instead of
rejection-sampled; depths 3-5 are still sampled. Everything keys on the satvec hash
(same blake2b scheme as rebuild_datasets._hash_sat), so "same behaviour" == "same key".

Curriculum is DEPTH-GRADED: stage_i train = depth <= (i+1), i in 1..4
(sizes ~99k/209k/429k/890k), eval = depth 2..(i+1) with exact-doubling per-depth counts.

Pipeline
--------
P0  index      : depth per curriculum row (cached) + dict {satvec -> row_ids}.
P1  partition  : enumerate all depth-2 classes (census, cached); split into free / covered;
                 hold out `d2` classes (free first, then SEEDED-RANDOM covered); removals =
                 held-out & covered (drop their rows), additions = free & !held-out (one
                 representative each into the depth-2 band, i.e. every stage).
P2  fill 3-5   : sample formulas (reverse depth order), accept satvec not in
                 (train U additions U held-out-so-far); seed from old eval/val first.
P3  split+lock : depth-stacked split into validation + stage4 eval; derive stage1-3 eval as
                 depth prefixes; build the four depth-graded train stages (removals applied,
                 additions appended); save.
P4  finetune   : from the LOCKED stage4 train, exactly 2*N equivalent rewrites + N near-miss
                 negation insertions (near-miss rejected if its satvec is in eval U validation).
P5  certify    : in-memory assertions on disjointness / uniqueness + on-disk row counts.

Reproducibility: embeddings via kernel.compute_embeddings_from_satisfactions (exact
covariance form); the kernel's traces must match the stored satvecs (guaranteed for kernel_v2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_class import LTLDataset
from formula_class import Formula
from formula_utils import (
    list_negation_insertions,
    list_semantically_equivalent_transformations,
    sample_formulas,
    str_to_formula,
)
from kernel_class import LTLKernel

DEPTHS = (2, 3, 4, 5)


# =========================== small helpers ===========================
def _hash_sat(sat_bool_cpu: torch.Tensor) -> bytes:
    return hashlib.blake2b(np.packbits(sat_bool_cpu.to(torch.uint8).numpy()).tobytes(), digest_size=16).digest()


def _is_trivial(sat_bool: torch.Tensor) -> bool:
    return bool(torch.all(sat_bool)) or not bool(torch.any(sat_bool))


def _done(dirpath: str) -> bool:
    return os.path.exists(os.path.join(dirpath, "metadata.json"))


def _save_dataset(dirpath, formulas, embeddings, satvecs, meta, sat_batch):
    ds = LTLDataset(store_formula_str=True, store_satisfaction=satvecs is not None,
                    satisfaction_batch_size=sat_batch, satisfaction_time_index=0)
    ds._reset_storage()
    ds.formulas = list(formulas)
    ds.formula_strs = [str(f) for f in formulas]
    ds.embeddings = embeddings.to(dtype=torch.float32, device="cpu")
    ds.satisfactions = None if satvecs is None else satvecs.to(dtype=torch.bool, device="cpu")
    ds.metadata = meta
    ds.save(dirpath)


def _project(kernel, satvecs, embed_batch, device):
    n = satvecs.size(0)
    out = torch.empty((n, kernel.m), dtype=torch.float32)
    for i in range(0, n, embed_batch):
        out[i:i + embed_batch] = kernel.compute_embeddings_from_satisfactions(
            satvecs[i:i + embed_batch].to(device), move_to_cpu=False).cpu()
    return out


def _eval_formulas(kernel, formulas, sat_batch, embed_batch, device):
    """Formulas -> (satvecs (n,N) bool cpu, embeddings (n,m) f32 cpu)."""
    N = kernel.traces.size(0)
    if len(formulas) == 0:
        return torch.empty((0, N), dtype=torch.bool), torch.empty((0, kernel.m), dtype=torch.float32)
    sat = torch.empty((len(formulas), N), dtype=torch.bool)
    for i, phi in enumerate(formulas):
        sat[i] = kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu()
    emb = _project(kernel, sat, embed_batch, device)
    return sat, emb


# =========================== census (depth-exactly-2) ===========================
def enumerate_depth2_classes(traces: torch.Tensor):
    """Return {satvec_hash -> representative formula str} over all non-trivial depth-2 forms."""
    N, AP, T = traces.shape

    def sweep(c, op):
        out = c.clone()
        for t in range(T - 2, -1, -1):
            out[:, t] = (out[:, t + 1] | c[:, t]) if op == "F" else (out[:, t + 1] & c[:, t])
        return out

    def until_table(L, R):
        out = torch.empty_like(R); out[:, T - 1] = R[:, T - 1]
        for t in range(T - 2, -1, -1):
            out[:, t] = R[:, t] | (L[:, t] & out[:, t + 1])
        return out

    def xnext(c):
        out = torch.zeros_like(c); out[:, :-1] = c[:, 1:]; return out

    kids = [(traces[:, i, :].contiguous(), 0, f"p_{i}") for i in range(AP)]
    atoms = list(kids)
    for fn, fmt in [(lambda c: ~c, "(~ {})"), (xnext, "(X {})"),
                    (lambda c: sweep(c, "F"), "(F {})"), (lambda c: sweep(c, "G"), "(G {})")]:
        for tbl, _, s in atoms:
            kids.append((fn(tbl), 1, fmt.format(s)))
    for fn, fmt in [(torch.logical_and, "({} AND {})"), (torch.logical_or, "({} OR {})"),
                    (lambda a, b: (~a) | b, "({} -> {})"), (until_table, "({} U {})")]:
        for ta, _, sa in atoms:
            for tb, _, sb in atoms:
                kids.append((fn(ta, tb), 1, fmt.format(sa, sb)))
    n_kids = len(kids)
    depths = np.array([d for _, d, _ in kids])
    strs = [s for _, _, s in kids]
    CHt = torch.stack([tbl for tbl, _, _ in kids]).permute(0, 2, 1).contiguous()  # (125, T, N)
    T0 = CHt[:, 0, :].contiguous()
    T1 = CHt[:, 1, :].contiguous()

    classes: dict[bytes, str] = {}

    def add(sats_2d, labels):
        arr = sats_2d.numpy()
        rs = arr.sum(axis=1)
        packed = np.packbits(arr, axis=1)
        for r in range(arr.shape[0]):
            if rs[r] == 0 or rs[r] == N:
                continue
            classes.setdefault(hashlib.blake2b(packed[r].tobytes(), digest_size=16).digest(), labels[r])

    d1 = np.where(depths == 1)[0]; d1_t = torch.from_numpy(d1)
    add(~T0[d1_t], [f"(~ {strs[i]})" for i in d1])
    add(T1[d1_t], [f"(X {strs[i]})" for i in d1])
    add(CHt[d1_t].any(dim=1), [f"(F {strs[i]})" for i in d1])
    add(CHt[d1_t].all(dim=1), [f"(G {strs[i]})" for i in d1])
    for i in range(n_kids):
        js = np.where(np.maximum(depths[i], depths) == 1)[0]
        if len(js) == 0:
            continue
        jt = torch.from_numpy(js); Rt0 = T0[jt]; a0 = T0[i].unsqueeze(0)
        add(a0 & Rt0, [f"({strs[i]} AND {strs[j]})" for j in js])
        add(a0 | Rt0, [f"({strs[i]} OR {strs[j]})" for j in js])
        add((~a0) | Rt0, [f"({strs[i]} -> {strs[j]})" for j in js])
        Lc = CHt[i]; Rc = CHt[jt]
        out = Rc[:, T - 1, :].clone()
        for t in range(T - 2, -1, -1):
            out = Rc[:, t, :] | (Lc[t].unsqueeze(0) & out)
        add(out, [f"({strs[i]} U {strs[j]})" for j in js])
    return classes


def load_or_build_census(kernel, cache_dir):
    hpath = os.path.join(cache_dir, "depth2_hashes.npy")
    rpath = os.path.join(cache_dir, "depth2_reps.jsonl")
    if os.path.exists(hpath) and os.path.exists(rpath):
        reps = {}
        with open(rpath) as fp:
            for line in fp:
                d = json.loads(line); reps[bytes.fromhex(d["hash"])] = d["formula"]
        print(f"[census] loaded {len(reps)} depth-2 classes from cache")
        return reps
    os.makedirs(cache_dir, exist_ok=True)
    print("[census] enumerating depth-2 classes ...", flush=True)
    t0 = time.time()
    reps = enumerate_depth2_classes(kernel.traces.cpu())
    keys = sorted(reps)
    np.save(hpath, np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 16))
    with open(rpath, "w") as fp:
        for k in keys:
            fp.write(json.dumps({"hash": k.hex(), "formula": reps[k]}) + "\n")
    print(f"[census] {len(reps)} non-trivial depth-2 classes ({time.time()-t0:.0f}s)")
    return reps


# =========================== P0: index ===========================
def build_index(stage4_dir, hashes_path, cache_dir):
    ds = LTLDataset.load(stage4_dir, load_satisfactions=False)
    formulas = ds.formulas
    n = len(formulas)
    hashes = np.load(hashes_path)  # (n, 16) uint8
    assert hashes.shape[0] == n, (hashes.shape, n)
    row_key = [hashes[i].tobytes() for i in range(n)]

    dpath = os.path.join(cache_dir, "depths.npy")
    if os.path.exists(dpath):
        depths = np.load(dpath)
        assert depths.shape[0] == n
    else:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[index] computing depth for {n} curriculum rows ...", flush=True)
        depths = np.fromiter((f.depth() for f in formulas), dtype=np.int16, count=n)
        np.save(dpath, depths)
    A: dict[bytes, list[int]] = {}
    for i, k in enumerate(row_key):
        A.setdefault(k, []).append(i)
    print(f"[index] {n} rows, {len(A)} distinct train satvecs, "
          f"depth hist { {int(d): int((depths==d).sum()) for d in np.unique(depths)} }")
    return formulas, depths, A


# =========================== P1: depth-2 partition ===========================
def partition_depth2(census, train_key_set, A, d2_budget, rng):
    census_keys = list(census)
    free = [k for k in census_keys if k not in train_key_set]
    covered = [k for k in census_keys if k in train_key_set]
    print(f"[P1] depth-2 classes: {len(census_keys)}  free={len(free)}  covered={len(covered)}  budget={d2_budget}")
    if d2_budget > len(census_keys):
        raise RuntimeError(f"d2 budget {d2_budget} exceeds total depth-2 classes {len(census_keys)}")

    free_perm = [free[i] for i in torch.randperm(len(free), generator=rng).tolist()]
    if len(free) >= d2_budget:
        held = free_perm[:d2_budget]
        residual = free_perm[d2_budget:]                  # -> add to train
        removals_keys: list[bytes] = []
    else:
        cov_perm = [covered[i] for i in torch.randperm(len(covered), generator=rng).tolist()]
        removals_keys = cov_perm[:d2_budget - len(free)]  # SEEDED-RANDOM covered
        held = list(free) + removals_keys
        residual = []
    removal_ids = {i for k in removals_keys for i in A[k]}
    print(f"[P1] held-out={len(held)}  additions(residual free)={len(residual)}  "
          f"removed covered classes={len(removals_keys)} -> {len(removal_ids)} train rows")
    return held, residual, removal_ids, removals_keys


# =========================== P2: sample depths 3-5 ===========================
def sample_depths_345(kernel, forbidden, need_per_depth, seed_forms, p_leaf_range,
                      sat_batch, sample_batch, attempt_budget):
    got = {d: [] for d in need_per_depth}
    seen = set(forbidden)

    def consider(phi, cap):
        d = phi.depth()
        if d not in got or len(got[d]) >= need_per_depth[d] or d > cap:
            return
        sat = kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu()
        if _is_trivial(sat):
            return
        k = _hash_sat(sat)
        if k in seen:
            return
        seen.add(k); got[d].append(phi)

    for phi in seed_forms:
        consider(phi, cap=max(need_per_depth))
    print(f"[P2] after seeding: { {d: len(got[d]) for d in need_per_depth} }")
    for target_d in sorted(need_per_depth, reverse=True):
        attempts = 0
        while len(got[target_d]) < need_per_depth[target_d] and attempts < attempt_budget:
            for phi in sample_formulas(n_formula=sample_batch, p_leaf_range=p_leaf_range,
                                       max_depth=target_d, n_ap=kernel.AP, force_tree=False,
                                       rng=kernel.rng, device=kernel.device):
                consider(phi, cap=target_d)
            attempts += sample_batch
        print(f"[P2] depth {target_d} (max_depth={target_d}, attempts={attempts}): "
              f"{ {d: len(got[d]) for d in need_per_depth} }")
    return got


# =========================== P4: finetune ===========================
def build_finetune(kernel, stage4_train_dir, out_dir, sample_count, forbidden_keys,
                   rng, sat_batch, embed_batch, device):
    ds = LTLDataset.load(stage4_train_dir, load_satisfactions=False)
    forms = ds.formulas
    perm = torch.randperm(len(forms), generator=rng).tolist()
    target_eq, target_nm = 2 * sample_count, 1 * sample_count
    eq_forms, nm_forms = [], []
    eq_seen, nm_seen = set(), set()
    used = 0
    for idx in perm:
        if len(eq_forms) >= target_eq and len(nm_forms) >= target_nm:
            break
        phi = forms[idx]
        equivs = list_semantically_equivalent_transformations(phi)
        if len(equivs) < 2:
            continue
        if _is_trivial(kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu()):
            continue
        used += 1
        if len(eq_forms) < target_eq:                      # up to 2 new-unique equivalents
            added = 0
            for j in torch.randperm(len(equivs), generator=rng).tolist():
                s = str(equivs[j])
                if s not in eq_seen:
                    eq_seen.add(s); eq_forms.append(equivs[j]); added += 1
                    if added >= 2 or len(eq_forms) >= target_eq:
                        break
        if len(nm_forms) < target_nm:                      # 1 near-miss, satvec not in eval U val
            cands = list_negation_insertions(phi)
            for j in torch.randperm(len(cands), generator=rng).tolist():
                mut = cands[j]; s = str(mut)
                if s in nm_seen:
                    continue
                if _hash_sat(kernel._evaluate_formula_on_traces(mut, batch_size=sat_batch).cpu()) in forbidden_keys:
                    continue
                nm_seen.add(s); nm_forms.append(mut); break
        if used % 5000 == 0:
            print(f"[P4] bases={used} eq={len(eq_forms)}/{target_eq} nm={len(nm_forms)}/{target_nm}", flush=True)
    if len(eq_forms) < target_eq or len(nm_forms) < target_nm:
        raise RuntimeError(f"finetune shortfall: eq={len(eq_forms)}/{target_eq} nm={len(nm_forms)}/{target_nm}")
    mutated = eq_forms + nm_forms
    print(f"[P4] {len(eq_forms)} equivalents + {len(nm_forms)} near-miss = {len(mutated)}; evaluating ...", flush=True)
    sat, emb = _eval_formulas(kernel, mutated, sat_batch, embed_batch, device)
    _save_dataset(out_dir, mutated, emb, sat,
                  {"source": "finetune_depthgraded", "bases_used": used,
                   "equivalents": len(eq_forms), "near_miss": len(nm_forms)}, sat_batch)
    print(f"[P4] saved finetune ({len(mutated)} rows) -> {out_dir}")


# =========================== main ===========================
def parse_args():
    p = argparse.ArgumentParser(description="Depth-graded curriculum/eval/validation/finetune rebuild.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kernel-dir", required=True)
    p.add_argument("--stage4-recompute-dir", required=True, help="datasets_v2/curriculum/stage4/train (formulas+embeddings)")
    p.add_argument("--stage4-hashes", required=True, help="row-aligned satvec hashes .npy for stage4 train")
    p.add_argument("--old-eval-dir", default=None)
    p.add_argument("--old-val-dir", default=None)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--eval-per-depth", nargs=4, type=int, default=[250, 250, 500, 1000], help="d2 d3 d4 d5")
    p.add_argument("--val-per-depth", nargs=4, type=int, default=[1000, 1000, 1000, 1000], help="d2 d3 d4 d5")
    p.add_argument("--finetune-sample-count", type=int, default=30000)
    p.add_argument("--p-leaf-range", nargs=2, type=float, default=[0.1, 0.5])
    p.add_argument("--sample-batch", type=int, default=51200)
    p.add_argument("--attempt-budget", type=int, default=5_000_000)
    p.add_argument("--satisfaction-batch-size", type=int, default=10240)
    p.add_argument("--embed-batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    kernel = LTLKernel.load(args.kernel_dir, device=args.device)
    if kernel.traces is None or kernel.F is None:
        raise ValueError("kernel needs traces and F")
    device = kernel.device
    rng = torch.Generator(device="cpu"); rng.manual_seed(int(args.seed))
    kernel.rng.manual_seed(int(args.seed))
    out = args.output_root
    cache = os.path.join(out, "_cache")
    os.makedirs(out, exist_ok=True)
    sat_batch, embed_batch = args.satisfaction_batch_size, args.embed_batch

    eval_pd = {d: c for d, c in zip(DEPTHS, args.eval_per_depth)}
    val_pd = {d: c for d, c in zip(DEPTHS, args.val_per_depth)}
    need = {d: eval_pd[d] + val_pd[d] for d in DEPTHS}
    print(f"[cfg] eval_per_depth={eval_pd} val_per_depth={val_pd} holdout_per_depth={need}")

    census = load_or_build_census(kernel, cache)
    formulas, depths, A = build_index(args.stage4_recompute_dir, args.stage4_hashes, cache)
    train_key_set = set(A.keys())

    seed_forms = []
    for d in (args.old_eval_dir, args.old_val_dir):
        if d and os.path.isdir(d):
            seed_forms.extend(LTLDataset.load(d, load_satisfactions=False).formulas)
    print(f"[seed] loaded {len(seed_forms)} old eval/val formulas")

    # ---- P1 ----
    held_d2_keys, residual_keys, removal_ids, removed_cov_keys = partition_depth2(
        census, train_key_set, A, need[2], rng)

    # depth-2 held-out reps (prefer an old-seed formula for the class, else training member, else census rep)
    seed_by_key: dict[bytes, Formula] = {}
    for phi in seed_forms:
        if phi.depth() == 2:
            k = _hash_sat(kernel._evaluate_formula_on_traces(phi, batch_size=sat_batch).cpu())
            seed_by_key.setdefault(k, phi)
    d2_reps = [seed_by_key[k] if k in seed_by_key
               else (formulas[A[k][0]] if k in train_key_set else str_to_formula(census[k]))
               for k in held_d2_keys]
    residual_reps = [str_to_formula(census[k]) for k in residual_keys]

    # ---- P2 ----
    addition_keys = set(residual_keys)
    forbidden = set(train_key_set) | set(held_d2_keys) | addition_keys
    pool_345 = sample_depths_345(kernel, forbidden, {d: need[d] for d in (3, 4, 5)},
                                 [f for f in seed_forms if f.depth() in (3, 4, 5)],
                                 tuple(args.p_leaf_range), sat_batch, args.sample_batch, args.attempt_budget)

    pool = {2: d2_reps, 3: pool_345[3], 4: pool_345[4], 5: pool_345[5]}
    for d in DEPTHS:
        if len(pool[d]) < need[d]:
            raise RuntimeError(f"depth {d}: only {len(pool[d])} held-out, need {need[d]}")
    print("[eval] evaluating held-out pool + additions satvecs/embeddings ...", flush=True)
    pool_sat, pool_emb = {}, {}
    for d in DEPTHS:
        pool_sat[d], pool_emb[d] = _eval_formulas(kernel, pool[d][:need[d]], sat_batch, embed_batch, device)
        pool[d] = pool[d][:need[d]]
    # sanity: depth-2 reps realise exactly their held-out classes
    for j, k in enumerate(held_d2_keys):
        assert _hash_sat(pool_sat[2][j]) == k, "depth-2 representative satvec != class key"
    add_sat, add_emb = _eval_formulas(kernel, residual_reps, sat_batch, embed_batch, device)

    # ---- P3 split eval/val (depth-stacked) ----
    val_f, val_e, val_s = [], [], []
    ev_f, ev_e, ev_s = {}, {}, {}          # per-depth eval slices
    eval_keys, val_keys = set(), set()
    for d in DEPTHS:
        perm = torch.randperm(len(pool[d]), generator=rng).tolist()
        vi, ei = perm[:val_pd[d]], perm[val_pd[d]:val_pd[d] + eval_pd[d]]
        val_f += [pool[d][i] for i in vi]; val_e.append(pool_emb[d][vi]); val_s.append(pool_sat[d][vi])
        ev_f[d] = [pool[d][i] for i in ei]; ev_e[d] = pool_emb[d][ei]; ev_s[d] = pool_sat[d][ei]
        val_keys |= {_hash_sat(pool_sat[d][i]) for i in vi}
        eval_keys |= {_hash_sat(pool_sat[d][i]) for i in ei}

    _save_dataset(os.path.join(out, "validation"), val_f, torch.cat(val_e), torch.cat(val_s),
                  {"source": "validation_depthgraded", "per_depth": val_pd}, sat_batch)
    for i, keep_max in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        ds = [d for d in DEPTHS if d <= keep_max]
        _save_dataset(os.path.join(out, f"stage{i}", "eval"),
                      sum((ev_f[d] for d in ds), []), torch.cat([ev_e[d] for d in ds]),
                      torch.cat([ev_s[d] for d in ds]),
                      {"source": f"stage{i}_eval", "depths": ds, "per_depth": {d: eval_pd[d] for d in ds}}, sat_batch)

    # ---- P3 build depth-graded train stages ----
    base_emb = LTLDataset.load(args.stage4_recompute_dir, load_satisfactions=False).embeddings
    kept = np.array([i for i in range(len(formulas)) if i not in removal_ids], dtype=np.int64)
    kept_depth = depths[kept]
    for i, keep_max in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        sel = kept[kept_depth <= keep_max]
        f = [formulas[j] for j in sel] + list(residual_reps)   # additions are depth 2 -> in every stage
        e = torch.cat([base_emb[torch.from_numpy(sel)], add_emb], dim=0)
        _save_dataset(os.path.join(out, f"stage{i}", "train"), f, e, None,
                      {"source": f"stage{i}_train_depthgraded", "max_depth": keep_max,
                       "kept_rows": int(sel.shape[0]), "additions": len(residual_reps)}, sat_batch)
        print(f"[P3] stage{i} train: {len(f)} rows (kept {sel.shape[0]} + add {len(residual_reps)})")

    # ---- P5 in-memory certification of the split ----
    final_train_keys = (train_key_set - set(removed_cov_keys)) | addition_keys
    heldout_keys = eval_keys | val_keys
    assert not (eval_keys & val_keys), "eval and validation share a satvec"
    assert not (heldout_keys & final_train_keys), "held-out satvec leaks into training"
    assert len(eval_keys) == sum(eval_pd.values()) and len(val_keys) == sum(val_pd.values()), "non-unique holdout"
    print(f"[certify] eval({len(eval_keys)}) / val({len(val_keys)}) unique & disjoint; "
          f"holdout disjoint from final train ({len(final_train_keys)} keys). OK")

    # ---- P4 finetune from locked stage4 train ----
    finetune_dir = os.path.join(out, "finetune", "train")
    if not _done(finetune_dir):
        build_finetune(kernel, os.path.join(out, "stage4", "train"), finetune_dir,
                       args.finetune_sample_count, heldout_keys, rng, sat_batch, embed_batch, device)
    print("[done] depth-graded rebuild complete.")


if __name__ == "__main__":
    main()
