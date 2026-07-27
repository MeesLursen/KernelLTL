"""Self-contained validation passes for KernelLTL models (Experiment 1).

Two passes per model, both writing raw per-generation JSONL records for
post-hoc analysis. The run computes only what needs the model (generation) or
the kernel traces (satvec scoring -> semantic distance / equivalence). All
downstream metrics -- pass@k, self-BLEU, distinct-correct, by-depth breakdowns,
bootstrap CIs -- are derived post-hoc from these records and live in the
analysis scripts, so they do not consume validation walltime.

- ``run_greedy_pass`` -- deterministic single sample per target  -> ``greedy.jsonl``
- ``run_topk_pass``   -- K sampled (T=1) sequences per target     -> ``topk_flat.jsonl``

Policy diagnostics (per-token entropy, KL to a reference policy, action
log-probs) are NOT computed here. They are finetuning-experiment quantities and
will be added by that experiment's own validation entry point.

Per-generation record schema (both passes share the semantic fields):

  greedy.jsonl     : formula_id, target_formula_str, target_depth,
                     generated_formula_str, generated_depth,
                     is_invalid, is_exact_match, is_semantic_equivalent,
                     semantic_distance, token_ids
  topk_flat.jsonl  : formula_id, target_depth, k_idx, generated_formula_str,
                     generated_depth, is_invalid, is_exact_match,
                     is_semantic_equivalent, semantic_distance, token_ids

``token_ids`` is the exact generated sequence as produced by ``generate``
(BOS prefix and EOS included, trailing padding stripped). It exists so the
finetuning experiment's policy-diagnostics pass can teacher-force the stored
trajectories exactly, without relying on string re-tokenization round-trips.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset_class import LTLDataset
from formula_utils import ParseError, str_to_formula
from kernel_class import LTLKernel
from model_class import LTLModel
from tokenizer_pretrained_class import LTLTokenizer


def _is_main(accelerator: Accelerator) -> bool:
    return bool(accelerator.is_main_process)


def _pad_to(t: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    """Pad/truncate the last dimension of ``t`` to a fixed ``length`` (for gather)."""
    cur = t.size(-1)
    if cur == length:
        return t
    if cur > length:
        return t[..., :length].contiguous()
    return F.pad(t, (0, length - cur), value=pad_value)


# --------------------------------------------------------------------------- #
# Embedding ablation (Experiment 1 floor, G1b)
# --------------------------------------------------------------------------- #

EMBEDDING_ABLATIONS = ("none", "zero", "mean", "shuffle")


def _apply_embedding_ablation(
    embs: torch.Tensor,
    mode: str,
    *,
    mean_embedding: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Destroy or corrupt the conditioning signal, for the embedding-ablation floor.

    ``none``    -- pass through (the real conditioned model).
    ``zero``    -- the zero embedding: the unconditional prior. (The kernel also maps every
                   tautology/contradiction to the zero embedding, so the decoder has seen it.)
    ``mean``    -- the dataset-mean embedding for every target: a constant, target-agnostic
                   signal of realistic magnitude.
    ``shuffle`` -- another target's embedding (permuted within the batch): a real,
                   in-distribution signal that does not match the target.

    The targets / satisfaction vectors the caller scores against are NOT permuted, so
    ``shuffle`` measures generating from a mismatched embedding.
    """
    if mode == "none":
        return embs
    if mode == "zero":
        return torch.zeros_like(embs)
    if mode == "mean":
        if mean_embedding is None:
            raise ValueError("embedding_ablation='mean' requires mean_embedding")
        me = mean_embedding.to(device=embs.device, dtype=embs.dtype).reshape(-1)
        return me.expand_as(embs).contiguous()
    if mode == "shuffle":
        B = embs.size(0)
        if B < 2:
            return embs
        perm = torch.randperm(B, generator=generator)
        return embs[perm.to(embs.device)].contiguous()
    raise ValueError(f"unknown embedding_ablation mode: {mode!r} (expected {EMBEDDING_ABLATIONS})")


def _make_ablation_generator(mode: str, seed: int, process_index: int) -> torch.Generator | None:
    """Seeded CPU generator for the ``shuffle`` ablation (per-process for DDP determinism)."""
    if mode != "shuffle":
        return None
    g = torch.Generator()
    g.manual_seed(int(seed) + int(process_index))
    return g


# --------------------------------------------------------------------------- #
# Semantic scoring (kept on the fly: needs the kernel traces)
# --------------------------------------------------------------------------- #

def _score_one_generated(
    *,
    generated_str: str,
    target_str: str,
    target_sat: torch.Tensor,
    kernel: LTLKernel,
    semantic_eval_batch_size: int,
) -> dict[str, float | bool | int]:
    """Return the per-generation semantic outcome for one (generated, target) pair.

    Parses the generation, evaluates its satisfaction vector over the kernel's trace
    sample, and compares it against the target's. An unparseable generation is flagged
    invalid and assigned the maximal semantic distance of 1.
    """
    out: dict[str, float | bool | int] = {
        "is_invalid": False,
        "is_exact_match": False,
        "is_semantic_equivalent": False,
        "semantic_distance": 1.0,
        "generated_depth": 0,
    }
    try:
        gen_formula = str_to_formula(generated_str)
        out["generated_depth"] = int(gen_formula.depth())
        if generated_str == target_str:
            out["is_exact_match"] = True
        gen_sat = kernel._evaluate_formula_on_traces(
            formula=gen_formula, batch_size=semantic_eval_batch_size
        )
        xor = torch.logical_xor(target_sat, gen_sat)
        distance = float(xor.to(dtype=torch.float32).mean().item())
        out["semantic_distance"] = distance
        if distance == 0.0:
            out["is_semantic_equivalent"] = True
    except ParseError:
        out["is_invalid"] = True
    except Exception as exc:  # noqa: BLE001 -- re-raise with context
        raise RuntimeError(
            "Unexpected validation failure while scoring a generated formula."
        ) from exc
    return out


def _score_batch(
    *,
    generated_strs: list[str],
    target_strs: list[str],
    target_sats: torch.Tensor,
    idx_to_target: list[int],
    kernel: LTLKernel,
    semantic_eval_batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Score a flat list of generations, returning gather-ready (N,) tensors.

    ``idx_to_target[j]`` maps generation ``j`` to its target row in ``target_sats`` /
    ``target_strs`` (identity for greedy; ``j // K`` for the K-sample pass).
    """
    n = len(generated_strs)
    scored = {
        "is_invalid": torch.zeros(n, dtype=torch.bool, device=device),
        "is_exact_match": torch.zeros(n, dtype=torch.bool, device=device),
        "is_semantic_equivalent": torch.zeros(n, dtype=torch.bool, device=device),
        "semantic_distance": torch.ones(n, dtype=torch.float32, device=device),
        "generated_depth": torch.zeros(n, dtype=torch.long, device=device),
    }
    for j in range(n):
        t = idx_to_target[j]
        oc = _score_one_generated(
            generated_str=generated_strs[j],
            target_str=target_strs[t],
            target_sat=target_sats[t],
            kernel=kernel,
            semantic_eval_batch_size=semantic_eval_batch_size,
        )
        scored["is_invalid"][j] = oc["is_invalid"]
        scored["is_exact_match"][j] = oc["is_exact_match"]
        scored["is_semantic_equivalent"][j] = oc["is_semantic_equivalent"]
        scored["semantic_distance"][j] = oc["semantic_distance"]
        scored["generated_depth"][j] = oc["generated_depth"]
    return scored


def _strip_trailing_pad(ids: list[int], pad_id: int) -> list[int]:
    """Drop trailing padding from a gathered fixed-width token row (keeps BOS/EOS)."""
    end = len(ids)
    while end > 0 and ids[end - 1] == pad_id:
        end -= 1
    return ids[:end]


def _headline(total: int, n_invalid: int, n_exact: int, n_equiv: int, sum_distance: float) -> dict[str, Any]:
    """A small sanity summary printed/persisted per pass (not the analysis output)."""
    summary: dict[str, Any] = {"n_samples": total}
    if total > 0:
        summary["semantic_equivalent_rate"] = n_equiv / total
        summary["semantic_distance"] = sum_distance / total
        summary["invalid_rate"] = n_invalid / total
        summary["syntactic_equal_rate"] = n_exact / total
    return summary


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #

def run_greedy_pass(
    *,
    model: LTLModel,
    eval_dataloader: DataLoader,
    kernel: LTLKernel,
    tokenizer: LTLTokenizer,
    dataset: LTLDataset,
    accelerator: Accelerator,
    output_jsonl_path: str,
    semantic_eval_batch_size: int = 10240,
    embedding_ablation: str = "none",
    mean_embedding: torch.Tensor | None = None,
    ablation_seed: int = 0,
) -> dict[str, Any]:
    """Greedy (do_sample=False) validation pass -> one record per target."""
    device = accelerator.device
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    T_max = int(model.config.n_positions)

    gen_model = model.module if hasattr(model, "module") else model
    original_training = bool(gen_model.training)
    gen_model.eval()
    ablation_gen = _make_ablation_generator(embedding_ablation, ablation_seed, accelerator.process_index)

    writer = None
    if _is_main(accelerator):
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        writer = open(output_jsonl_path, "w")

    total = n_invalid = n_exact = n_equiv = 0
    sum_distance = 0.0

    try:
        with torch.no_grad(), accelerator.autocast():
            for batch in eval_dataloader:
                embs = batch["encoder_hidden_states"].to(device, non_blocking=True)
                embs = _apply_embedding_ablation(
                    embs, embedding_ablation, mean_embedding=mean_embedding, generator=ablation_gen)
                target_sats = batch["target_satisfaction"].to(device)
                target_strs = batch["target_formula_strs"]
                formula_ids = batch["formula_ids"].to(device)
                B = embs.size(0)

                out = gen_model.generate(
                    encoder_hidden_states=embs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=T_max,
                    return_dict_in_generate=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                sequences = out.sequences  # (B, L); special tokens stripped on decode
                generated_strs = tokenizer.batch_decode(
                    sequences.detach().cpu(), skip_special_tokens=True)

                scored = _score_batch(
                    generated_strs=generated_strs,
                    target_strs=target_strs,
                    target_sats=target_sats,
                    idx_to_target=list(range(B)),
                    kernel=kernel,
                    semantic_eval_batch_size=semantic_eval_batch_size,
                    device=device,
                )
                # Pad to T_max + 1: sequences carry the BOS prefix plus up to
                # T_max new tokens, so padding to T_max would clip the final
                # token of a maximum-length generation.
                seq_pad = _pad_to(sequences.to(torch.long), T_max + 1, pad_id)  # (B, T_max+1)

                (g_fid, g_seq, g_inv, g_exact, g_equiv, g_dist, g_depth) = (
                    accelerator.gather_for_metrics((
                        formula_ids, seq_pad,
                        scored["is_invalid"], scored["is_exact_match"],
                        scored["is_semantic_equivalent"], scored["semantic_distance"],
                        scored["generated_depth"],
                    ))
                )
                if not _is_main(accelerator):
                    continue

                for i in range(g_fid.size(0)):
                    fid = int(g_fid[i].item())
                    target_formula = dataset.formulas[fid]
                    is_invalid = bool(g_inv[i].item())
                    is_equiv = bool(g_equiv[i].item())
                    is_exact = bool(g_exact[i].item())
                    distance = float(g_dist[i].item())
                    token_ids = _strip_trailing_pad(g_seq[i].cpu().tolist(), pad_id)
                    generated_str = tokenizer.decode(token_ids, skip_special_tokens=True)

                    writer.write(json.dumps({
                        "formula_id": fid,
                        "target_formula_str": str(target_formula),
                        "target_depth": int(target_formula.depth()),
                        "generated_formula_str": generated_str,
                        "generated_depth": (None if is_invalid else int(g_depth[i].item())),
                        "is_invalid": is_invalid,
                        "is_exact_match": is_exact,
                        "is_semantic_equivalent": is_equiv,
                        "semantic_distance": distance,
                        "token_ids": token_ids,
                    }) + "\n")

                    total += 1
                    sum_distance += distance
                    n_invalid += int(is_invalid)
                    n_exact += int(is_exact)
                    n_equiv += int(is_equiv)
    finally:
        if writer is not None:
            writer.close()
        if original_training:
            gen_model.train()

    if not _is_main(accelerator):
        return {}
    return _headline(total, n_invalid, n_exact, n_equiv, sum_distance)


def run_topk_pass(
    *,
    model: LTLModel,
    eval_dataloader: DataLoader,
    kernel: LTLKernel,
    tokenizer: LTLTokenizer,
    dataset: LTLDataset,
    accelerator: Accelerator,
    top_k: int,
    output_flat_path: str,
    semantic_eval_batch_size: int = 10240,
    embedding_ablation: str = "none",
    mean_embedding: torch.Tensor | None = None,
    ablation_seed: int = 0,
) -> dict[str, Any]:
    """Top-K (T=1, do_sample=True) validation pass -> one record per (target, k)."""
    device = accelerator.device
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    T_max = int(model.config.n_positions)
    K = int(top_k)

    gen_model = model.module if hasattr(model, "module") else model
    original_training = bool(gen_model.training)
    gen_model.eval()
    ablation_gen = _make_ablation_generator(embedding_ablation, ablation_seed, accelerator.process_index)

    writer = None
    if _is_main(accelerator):
        os.makedirs(os.path.dirname(output_flat_path), exist_ok=True)
        writer = open(output_flat_path, "w")

    total = n_invalid = n_exact = n_equiv = 0
    sum_distance = 0.0

    try:
        with torch.no_grad(), accelerator.autocast():
            for batch in eval_dataloader:
                embs = batch["encoder_hidden_states"].to(device, non_blocking=True)
                embs = _apply_embedding_ablation(
                    embs, embedding_ablation, mean_embedding=mean_embedding, generator=ablation_gen)
                target_sats = batch["target_satisfaction"].to(device)
                target_strs = batch["target_formula_strs"]
                formula_ids = batch["formula_ids"].to(device)
                B = embs.size(0)

                out = gen_model.generate(
                    encoder_hidden_states=embs,
                    do_sample=True,
                    num_beams=1,
                    num_return_sequences=K,
                    temperature=1.0,
                    max_new_tokens=T_max,
                    return_dict_in_generate=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                sequences = out.sequences  # (B*K, L), ordered as [b0k0, b0k1, ..., b1k0, ...]
                generated_strs = tokenizer.batch_decode(
                    sequences.detach().cpu(), skip_special_tokens=True)

                scored = _score_batch(
                    generated_strs=generated_strs,
                    target_strs=target_strs,
                    target_sats=target_sats,
                    idx_to_target=[idx // K for idx in range(B * K)],
                    kernel=kernel,
                    semantic_eval_batch_size=semantic_eval_batch_size,
                    device=device,
                )
                # T_max + 1: BOS prefix + up to T_max new tokens (see greedy pass).
                seq_pad = _pad_to(sequences.to(torch.long), T_max + 1, pad_id).view(B, K, T_max + 1)
                scored = {k: v.view(B, K) if v.dim() == 1 else v for k, v in scored.items()}

                (g_fid, g_seq, g_inv, g_exact, g_equiv, g_dist, g_depth) = (
                    accelerator.gather_for_metrics((
                        formula_ids, seq_pad,
                        scored["is_invalid"], scored["is_exact_match"],
                        scored["is_semantic_equivalent"], scored["semantic_distance"],
                        scored["generated_depth"],
                    ))
                )
                if not _is_main(accelerator):
                    continue

                for b in range(g_fid.size(0)):
                    fid = int(g_fid[b].item())
                    target_depth = int(dataset.formulas[fid].depth())
                    for k in range(K):
                        is_invalid = bool(g_inv[b, k].item())
                        is_equiv = bool(g_equiv[b, k].item())
                        is_exact = bool(g_exact[b, k].item())
                        distance = float(g_dist[b, k].item())
                        token_ids = _strip_trailing_pad(g_seq[b, k].cpu().tolist(), pad_id)
                        generated_str = tokenizer.decode(token_ids, skip_special_tokens=True)

                        writer.write(json.dumps({
                            "formula_id": fid,
                            "target_depth": target_depth,
                            "k_idx": k,
                            "generated_formula_str": generated_str,
                            "generated_depth": (None if is_invalid else int(g_depth[b, k].item())),
                            "is_invalid": is_invalid,
                            "is_exact_match": is_exact,
                            "is_semantic_equivalent": is_equiv,
                            "semantic_distance": distance,
                            "token_ids": token_ids,
                        }) + "\n")

                        total += 1
                        sum_distance += distance
                        n_invalid += int(is_invalid)
                        n_exact += int(is_exact)
                        n_equiv += int(is_equiv)
    finally:
        if writer is not None:
            writer.close()
        if original_training:
            gen_model.train()

    if not _is_main(accelerator):
        return {}
    summary = _headline(total, n_invalid, n_exact, n_equiv, sum_distance)
    summary["top_k"] = K
    summary["n_targets"] = total // K if K else 0
    return summary
