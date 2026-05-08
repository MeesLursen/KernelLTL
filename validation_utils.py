"""Self-contained validation passes for KernelLTL models.

Two passes per model:

- ``run_greedy_pass``  -- deterministic single-sample (do_sample=False).
  Collects per-sample semantic outcomes plus per-token policy entropy,
  log-prob of the chosen token, and KL(policy || base) along the greedy
  trajectory.

- ``run_topk_pass``    -- K=5 sampled (T=1) per target. Collects per-(target, k)
  reward, per-(target, k) per-token entropy/log-prob/KL, and per-target
  self-BLEU + reward variance.

Each pass writes per-sample (and per-token) JSONL records on the main
process and returns aggregate metrics.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset_class import LTLDataset
from formula_utils import ParseError, str_to_formula
from kernel_class import LTLKernel
from model_class import LTLModel
from tokenizer_pretrained_class import LTLTokenizer


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _is_main(accelerator: Accelerator) -> bool:
    return bool(accelerator.is_main_process)


def _pad_T(t: torch.Tensor, T_target: int, pad_value: float = 0.0) -> torch.Tensor:
    """Pad/truncate the last dimension of `t` to `T_target`."""
    T_actual = t.size(-1)
    if T_actual == T_target:
        return t
    if T_actual > T_target:
        return t[..., :T_target].contiguous()
    return F.pad(t, (0, T_target - T_actual), value=pad_value)


def _sentence_bleu(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """Sentence-level BLEU. Behaviour matches training_utils.SemanticEvaluationCallback._sentence_bleu."""
    if not candidate or not references:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(candidate) < n:
            precisions.append(1e-8)
            continue

        cand_counts: dict[tuple[str, ...], int] = defaultdict(int)
        for i in range(len(candidate) - n + 1):
            cand_counts[tuple(candidate[i : i + n])] += 1

        max_ref_counts: dict[tuple[str, ...], int] = defaultdict(int)
        for ref in references:
            ref_counts: dict[tuple[str, ...], int] = defaultdict(int)
            if len(ref) >= n:
                for i in range(len(ref) - n + 1):
                    ref_counts[tuple(ref[i : i + n])] += 1
            for ng, cnt in ref_counts.items():
                if cnt > max_ref_counts[ng]:
                    max_ref_counts[ng] = cnt

        clipped = 0
        total = 0
        for ng, cnt in cand_counts.items():
            clipped += min(cnt, max_ref_counts.get(ng, 0))
            total += cnt
        precisions.append((clipped + 1e-8) / (total + 1e-8))

    ref_lens = [len(ref) for ref in references]
    cand_len = len(candidate)
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - cand_len), x))
    if cand_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(closest_ref_len) / max(float(cand_len), 1.0))

    log_precision = sum(math.log(p) for p in precisions) / float(max_n)
    return float(bp * math.exp(log_precision))


def _bleu_tokens_from_sequence(seq: torch.Tensor, bos_id: int, eos_id: int, pad_id: int) -> list[str]:
    ids = seq.tolist()
    if bos_id in ids:
        start = ids.index(bos_id) + 1
    else:
        start = 0
    try:
        end = ids.index(eos_id, start)
    except ValueError:
        end = len(ids)
    content = ids[start:end]
    return [str(x) for x in content if x != pad_id]


# ---------------------------------------------------------------------------
# semantic evaluation of a single generated formula
# ---------------------------------------------------------------------------

def _score_one_generated(
    *,
    generated_str: str,
    target_str: str,
    target_sat: torch.Tensor,
    kernel: LTLKernel,
    semantic_eval_batch_size: int,
) -> dict[str, float | bool | int]:
    """Return per-sample outcome dict for one (generated, target) pair.

    Always returns the same keys; depth/length default to 0 when invalid.
    """
    out = {
        "is_invalid": False,
        "is_exact_match": False,
        "is_semantic_equivalent": False,
        "semantic_distance": 1.0,  # max distance; overwritten if valid
        "generated_depth": 0,
        "reward": 0.0,
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
        out["reward"] = 1.0 - distance
        if distance == 0.0:
            out["is_semantic_equivalent"] = True
    except ParseError:
        out["is_invalid"] = True
    except Exception as exc:
        raise RuntimeError(
            "Unexpected validation failure while scoring a generated formula."
        ) from exc
    return out


# ---------------------------------------------------------------------------
# greedy pass
# ---------------------------------------------------------------------------

def run_greedy_pass(
    *,
    model: LTLModel,
    ref_model_path: str | None,
    eval_dataloader: DataLoader,
    kernel: LTLKernel,
    tokenizer: LTLTokenizer,
    dataset: LTLDataset,
    accelerator: Accelerator,
    output_jsonl_path: str,
    semantic_eval_batch_size: int = 10240,
) -> dict[str, Any]:
    """Greedy validation pass.

    Stage A: gen model autoregressively on every batch; collect per-token
             entropy / log-prob and per-sample semantic outcomes.
             Buffer (CPU) what's needed for KL.
    Stage B: swap gen->ref on the GPU, run reference forward per batch,
             compute per-token KL.
    Stage C: gather across processes, write per-sample JSONL on rank 0,
             return aggregate dict.
    """
    device = accelerator.device
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    T_max = int(model.config.n_positions)

    gen_model = model.module if hasattr(model, "module") else model
    original_training = bool(gen_model.training)
    gen_model.eval()

    # Each entry in buffered_batches contains everything needed to write JSONL
    # (after gather) plus the tensors needed for the reference-model pass.
    buffered_batches: list[dict[str, torch.Tensor]] = []

    with torch.no_grad(), accelerator.autocast():
        for batch in eval_dataloader:
            embs = batch["encoder_hidden_states"].to(device, non_blocking=True)
            target_sats = batch["target_satisfaction"].to(device)
            target_strs = batch["target_formula_strs"]
            formula_ids = batch["formula_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            B = embs.size(0)

            out = gen_model.generate(
                encoder_hidden_states=embs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=T_max,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

            sequences = out.sequences
            scores = list(out.scores) if isinstance(out.scores, tuple) else out.scores
            if not scores:
                # Degenerate batch: nothing generated. Skip.
                continue

            # (B, T_actual, V)
            score_tensor = torch.stack(scores, dim=1)
            re_log_probs = torch.log_softmax(score_tensor, dim=-1)
            re_probs = torch.exp(re_log_probs)

            T_actual = score_tensor.size(1)
            seq_len = sequences.size(-1)
            prefix_len = max(0, seq_len - T_actual)
            gen_tokens = sequences[:, prefix_len : prefix_len + T_actual].long()  # (B, T_actual)

            # (B, T_actual)
            token_mask = (gen_tokens != pad_id)
            token_mask_f = token_mask.to(dtype=score_tensor.dtype)
            token_entropy = -(re_probs * re_log_probs).sum(dim=-1)
            token_log_prob = re_log_probs.gather(
                dim=-1, index=gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # per-sample semantic outcomes
            generated_strs = tokenizer.batch_decode(
                gen_tokens.detach().cpu(), skip_special_tokens=True
            )
            generated_length_tokens = token_mask.sum(dim=-1).to(dtype=torch.long)  # (B,)
            target_length_tokens = attention_mask.sum(dim=-1).to(dtype=torch.long)  # (B,)

            per_sample = {
                "is_invalid": torch.zeros(B, dtype=torch.bool, device=device),
                "is_exact_match": torch.zeros(B, dtype=torch.bool, device=device),
                "is_semantic_equivalent": torch.zeros(B, dtype=torch.bool, device=device),
                "semantic_distance": torch.ones(B, dtype=torch.float32, device=device),
                "generated_depth": torch.zeros(B, dtype=torch.long, device=device),
            }
            for i in range(B):
                outcome = _score_one_generated(
                    generated_str=generated_strs[i],
                    target_str=target_strs[i],
                    target_sat=target_sats[i],
                    kernel=kernel,
                    semantic_eval_batch_size=semantic_eval_batch_size,
                )
                per_sample["is_invalid"][i] = outcome["is_invalid"]
                per_sample["is_exact_match"][i] = outcome["is_exact_match"]
                per_sample["is_semantic_equivalent"][i] = outcome["is_semantic_equivalent"]
                per_sample["semantic_distance"][i] = outcome["semantic_distance"]
                per_sample["generated_depth"][i] = outcome["generated_depth"]

            # Pad per-token tensors to T_max so all batches have the same shape
            # (gather_for_metrics would otherwise need pad_across_processes).
            gen_tokens_pad = _pad_T(gen_tokens, T_max, pad_value=pad_id).to(dtype=torch.long)
            mask_pad = _pad_T(token_mask.to(dtype=torch.uint8), T_max, pad_value=0)
            entropy_pad = _pad_T(token_entropy.to(dtype=torch.float32), T_max, pad_value=0.0)
            log_prob_pad = _pad_T(token_log_prob.to(dtype=torch.float32), T_max, pad_value=0.0)

            # Move re_log_probs to CPU now to keep GPU memory low while we
            # iterate the rest of the dataloader. Keep the per-token summary
            # tensors on GPU so we can gather them at the end.
            re_log_probs_cpu = re_log_probs.detach().cpu().to(dtype=torch.float32)
            sequences_cpu = sequences.detach().cpu()
            embs_cpu = embs.detach().cpu()

            buffered_batches.append({
                # Per-token summary tensors on GPU (will be padded again w/ KL):
                "gen_tokens_pad": gen_tokens_pad,
                "mask_pad": mask_pad,
                "entropy_pad": entropy_pad,
                "log_prob_pad": log_prob_pad,
                # Per-sample tensors on GPU:
                "formula_ids": formula_ids.to(dtype=torch.long),
                "generated_length_tokens": generated_length_tokens,
                "target_length_tokens": target_length_tokens,
                **per_sample,
                # CPU tensors for ref-model pass:
                "_sequences_cpu": sequences_cpu,
                "_embs_cpu": embs_cpu,
                "_re_log_probs_cpu": re_log_probs_cpu,
                "_T_actual": T_actual,
            })

    # ------------------------------------------------------------------
    # Stage B: reference-model pass (per-token KL)
    # ------------------------------------------------------------------
    do_kl = ref_model_path is not None and len(buffered_batches) > 0
    if do_kl:
        # Mirror the swap pattern from training_utils.py:1125-1157
        gen_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ref_model: LTLModel | None = None
        try:
            ref_model = LTLModel.from_pretrained(ref_model_path)
            ref_model.to(device)
            ref_model.eval()
            with torch.no_grad(), accelerator.autocast():
                for buf in buffered_batches:
                    sequences_cpu = buf.pop("_sequences_cpu")
                    embs_cpu = buf.pop("_embs_cpu")
                    re_log_probs_cpu = buf.pop("_re_log_probs_cpu")
                    T_actual = buf.pop("_T_actual")

                    shifted = sequences_cpu[:, :-1].to(device, non_blocking=True)
                    shifted_attn = (shifted != pad_id).to(dtype=torch.long)
                    embs_g = embs_cpu.to(device, non_blocking=True)
                    re_lp_g = re_log_probs_cpu.to(device, non_blocking=True)

                    ce_logits = ref_model(
                        input_ids=shifted,
                        attention_mask=shifted_attn,
                        encoder_hidden_states=embs_g,
                    ).logits[:, -T_actual:, :]
                    ce_log_probs = torch.log_softmax(ce_logits, dim=-1)
                    re_p = torch.exp(re_lp_g)
                    token_kl = (re_p * (re_lp_g - ce_log_probs)).sum(dim=-1)  # (B, T_actual)

                    buf["kl_pad"] = _pad_T(token_kl.to(dtype=torch.float32), T_max, pad_value=0.0)
        finally:
            if ref_model is not None:
                del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gen_model.to(device)
    else:
        # Drop buffered CPU tensors we no longer need
        for buf in buffered_batches:
            buf.pop("_sequences_cpu", None)
            buf.pop("_embs_cpu", None)
            buf.pop("_re_log_probs_cpu", None)
            buf.pop("_T_actual", None)
            buf["kl_pad"] = torch.zeros(
                buf["entropy_pad"].size(), dtype=torch.float32, device=device
            )

    if original_training:
        gen_model.train()

    # ------------------------------------------------------------------
    # Stage C: gather, write JSONL, aggregate
    # ------------------------------------------------------------------
    total = 0
    sum_distance = 0.0
    n_invalid = 0
    n_exact = 0
    n_equiv = 0
    sum_gen_depth = 0
    sum_gen_len = 0
    n_valid_for_depth_len = 0  # i.e. valid (parseable) generations
    sum_token_entropy = 0.0
    sum_token_kl = 0.0
    sum_token_log_prob = 0.0
    sum_tokens = 0  # for token-mean aggregations
    sum_seq_entropy_means = 0.0
    sum_seq_kl_means = 0.0
    sum_seq_log_prob_means = 0.0
    n_with_tokens = 0

    writer = None
    if _is_main(accelerator):
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        writer = open(output_jsonl_path, "w")

    try:
        for buf in buffered_batches:
            gathered = accelerator.gather_for_metrics((
                buf["formula_ids"],
                buf["generated_length_tokens"],
                buf["target_length_tokens"],
                buf["is_invalid"],
                buf["is_exact_match"],
                buf["is_semantic_equivalent"],
                buf["semantic_distance"],
                buf["generated_depth"],
                buf["gen_tokens_pad"],
                buf["mask_pad"],
                buf["entropy_pad"],
                buf["log_prob_pad"],
                buf["kl_pad"],
            ))
            (
                g_formula_ids,
                g_gen_len,
                g_tgt_len,
                g_invalid,
                g_exact,
                g_equiv,
                g_distance,
                g_gen_depth,
                g_tok_ids,
                g_mask,
                g_entropy,
                g_log_prob,
                g_kl,
            ) = gathered

            if not _is_main(accelerator):
                continue

            n = g_formula_ids.size(0)
            for i in range(n):
                fid = int(g_formula_ids[i].item())
                target_formula = dataset.formulas[fid]
                target_str = str(target_formula)
                target_depth = int(target_formula.depth())

                mask_row = g_mask[i].to(dtype=torch.bool).cpu()
                tok_count = int(mask_row.sum().item())
                tok_ids = g_tok_ids[i].cpu().tolist()[:tok_count]
                ent_row = g_entropy[i].cpu().tolist()[:tok_count]
                lp_row = g_log_prob[i].cpu().tolist()[:tok_count]
                kl_row = g_kl[i].cpu().tolist()[:tok_count]

                generated_str = tokenizer.decode(tok_ids, skip_special_tokens=True)

                is_invalid = bool(g_invalid[i].item())
                is_exact = bool(g_exact[i].item())
                is_equiv = bool(g_equiv[i].item())
                distance = float(g_distance[i].item())
                gen_depth = int(g_gen_depth[i].item())
                gen_len_tokens = int(g_gen_len[i].item())
                tgt_len_tokens = int(g_tgt_len[i].item())

                row = {
                    "formula_id": fid,
                    "target_formula_str": target_str,
                    "target_depth": target_depth,
                    "target_length_tokens": tgt_len_tokens,
                    "generated_formula_str": generated_str,
                    "generated_depth": gen_depth,
                    "generated_length_tokens": gen_len_tokens,
                    "is_invalid": is_invalid,
                    "is_exact_match": is_exact,
                    "is_semantic_equivalent": is_equiv,
                    "semantic_distance": distance,
                    "token_ids": tok_ids,
                    "token_entropies": ent_row,
                    "token_log_probs": lp_row,
                    "token_kls": kl_row,
                }
                writer.write(json.dumps(row) + "\n")

                # Aggregates
                total += 1
                sum_distance += distance
                if is_invalid:
                    n_invalid += 1
                else:
                    sum_gen_depth += gen_depth
                    sum_gen_len += gen_len_tokens
                    n_valid_for_depth_len += 1
                if is_exact:
                    n_exact += 1
                if is_equiv:
                    n_equiv += 1

                if tok_count > 0:
                    seq_entropy_mean = sum(ent_row) / tok_count
                    seq_kl_mean = sum(kl_row) / tok_count
                    seq_lp_mean = sum(lp_row) / tok_count
                    sum_seq_entropy_means += seq_entropy_mean
                    sum_seq_kl_means += seq_kl_mean
                    sum_seq_log_prob_means += seq_lp_mean
                    sum_token_entropy += sum(ent_row)
                    sum_token_kl += sum(kl_row)
                    sum_token_log_prob += sum(lp_row)
                    sum_tokens += tok_count
                    n_with_tokens += 1
    finally:
        if writer is not None:
            writer.close()

    if not _is_main(accelerator):
        return {}

    summary: dict[str, Any] = {
        "n_samples": total,
    }
    if total > 0:
        summary["semantic_distance"] = sum_distance / total
        summary["invalid_rate"] = n_invalid / total
        summary["syntactic_equal_rate"] = n_exact / total
        summary["semantic_equivalent_rate"] = n_equiv / total
        summary["syntax_semantics_gap"] = (n_equiv - n_exact) / total
    if n_valid_for_depth_len > 0:
        summary["generated_depth_mean"] = sum_gen_depth / n_valid_for_depth_len
        summary["generated_length_tokens_mean"] = sum_gen_len / n_valid_for_depth_len
    if n_with_tokens > 0:
        # (a) sequence-mean: each formula contributes equally
        summary["policy_entropy_seq_mean"] = sum_seq_entropy_means / n_with_tokens
        summary["kl_from_base_seq_mean"] = sum_seq_kl_means / n_with_tokens
        summary["action_log_prob_seq_mean"] = sum_seq_log_prob_means / n_with_tokens
    if sum_tokens > 0:
        # (b) token-mean: each token contributes equally
        summary["policy_entropy_token_mean"] = sum_token_entropy / sum_tokens
        summary["kl_from_base_token_mean"] = sum_token_kl / sum_tokens
        summary["action_log_prob_token_mean"] = sum_token_log_prob / sum_tokens
        summary["total_generated_tokens"] = sum_tokens
    return summary


# ---------------------------------------------------------------------------
# top-K pass
# ---------------------------------------------------------------------------

def run_topk_pass(
    *,
    model: LTLModel,
    ref_model_path: str | None,
    eval_dataloader: DataLoader,
    kernel: LTLKernel,
    tokenizer: LTLTokenizer,
    dataset: LTLDataset,
    accelerator: Accelerator,
    top_k: int,
    output_flat_path: str,
    output_grouped_path: str,
    semantic_eval_batch_size: int = 10240,
) -> dict[str, Any]:
    """Top-K (T=1, do_sample=True) validation pass.

    For each target we sample K=top_k sequences and collect:

      - per (target, k): reward, per-token entropy/log_prob/KL, generated str
      - per target: self-BLEU over the K sequences, reward variance

    Writes two JSONL files on rank 0:
      - flat:    one row per (formula_id, k_idx)
      - grouped: one row per formula_id (per-target aggregates)

    Returns aggregate dict.
    """
    device = accelerator.device
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    T_max = int(model.config.n_positions)
    K = int(top_k)

    gen_model = model.module if hasattr(model, "module") else model
    original_training = bool(gen_model.training)
    gen_model.eval()

    buffered_batches: list[dict[str, torch.Tensor]] = []

    with torch.no_grad(), accelerator.autocast():
        for batch in eval_dataloader:
            embs = batch["encoder_hidden_states"].to(device, non_blocking=True)
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
                output_scores=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

            sequences = out.sequences  # (B*K, L)
            scores = list(out.scores) if isinstance(out.scores, tuple) else out.scores
            if not scores:
                continue

            score_tensor = torch.stack(scores, dim=1)  # (B*K, T_actual, V)
            re_log_probs = torch.log_softmax(score_tensor, dim=-1)
            re_probs = torch.exp(re_log_probs)

            T_actual = score_tensor.size(1)
            seq_len = sequences.size(-1)
            prefix_len = max(0, seq_len - T_actual)
            gen_tokens = sequences[:, prefix_len : prefix_len + T_actual].long()  # (B*K, T_actual)

            token_mask = (gen_tokens != pad_id)
            token_mask_f = token_mask.to(dtype=score_tensor.dtype)
            token_entropy = -(re_probs * re_log_probs).sum(dim=-1)
            token_log_prob = re_log_probs.gather(
                dim=-1, index=gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

            sequences_cpu = sequences.detach().cpu()
            generated_strs = tokenizer.batch_decode(
                gen_tokens.detach().cpu(), skip_special_tokens=True
            )

            # Per-(target, k) outcomes -- stored as (B, K) so the leading dim
            # matches the dataset stride for accelerator.gather_for_metrics.
            per_sk_invalid = torch.zeros(B, K, dtype=torch.bool, device=device)
            per_sk_reward = torch.zeros(B, K, dtype=torch.float32, device=device)

            # Self-BLEU per target needs all K bleu-token sequences for that target
            grouped_token_sequences: list[list[list[str]]] = [[] for _ in range(B)]
            grouped_rewards: list[list[float]] = [[] for _ in range(B)]
            grouped_invalid_idx: list[list[int]] = [[] for _ in range(B)]

            for idx in range(B * K):
                b_idx = idx // K
                k_idx = idx % K
                outcome = _score_one_generated(
                    generated_str=generated_strs[idx],
                    target_str=target_strs[b_idx],
                    target_sat=target_sats[b_idx],
                    kernel=kernel,
                    semantic_eval_batch_size=semantic_eval_batch_size,
                )
                per_sk_invalid[b_idx, k_idx] = outcome["is_invalid"]
                per_sk_reward[b_idx, k_idx] = outcome["reward"]
                if outcome["is_invalid"]:
                    grouped_invalid_idx[b_idx].append(k_idx)
                grouped_rewards[b_idx].append(outcome["reward"])
                grouped_token_sequences[b_idx].append(
                    _bleu_tokens_from_sequence(sequences_cpu[idx], bos_id, eos_id, pad_id)
                )

            per_target_self_bleu = torch.zeros(B, dtype=torch.float32, device=device)
            per_target_has_bleu = torch.zeros(B, dtype=torch.bool, device=device)
            per_target_reward_mean = torch.zeros(B, dtype=torch.float32, device=device)
            per_target_reward_var = torch.zeros(B, dtype=torch.float32, device=device)
            per_target_n_invalid = torch.zeros(B, dtype=torch.long, device=device)

            for b_idx in range(B):
                rewards = grouped_rewards[b_idx]
                if rewards:
                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                    per_target_reward_mean[b_idx] = rewards_t.mean()
                    per_target_reward_var[b_idx] = torch.var(rewards_t, unbiased=False)
                per_target_n_invalid[b_idx] = len(grouped_invalid_idx[b_idx])

                token_seqs = grouped_token_sequences[b_idx]
                if len(token_seqs) >= 2:
                    bleu_vals: list[float] = []
                    for i, cand in enumerate(token_seqs):
                        refs = [r for j, r in enumerate(token_seqs) if j != i]
                        bleu_vals.append(_sentence_bleu(cand, refs))
                    if bleu_vals:
                        per_target_self_bleu[b_idx] = float(sum(bleu_vals) / len(bleu_vals))
                        per_target_has_bleu[b_idx] = True

            # Pad token-level tensors to T_max, then reshape to (B, K, T_max)
            # so the leading dimension matches the dataset stride.
            gen_tokens_pad = _pad_T(gen_tokens, T_max, pad_value=pad_id).to(dtype=torch.long).view(B, K, T_max)
            mask_pad = _pad_T(token_mask.to(dtype=torch.uint8), T_max, pad_value=0).view(B, K, T_max)
            entropy_pad = _pad_T(token_entropy.to(dtype=torch.float32), T_max, pad_value=0.0).view(B, K, T_max)
            log_prob_pad = _pad_T(token_log_prob.to(dtype=torch.float32), T_max, pad_value=0.0).view(B, K, T_max)

            re_log_probs_cpu = re_log_probs.detach().cpu().to(dtype=torch.float32)
            embs_cpu = embs.detach().cpu()

            buffered_batches.append({
                "B": B,
                "K": K,
                # per (target, k) reshaped to (B, K, ...) for gather_for_metrics
                "gen_tokens_pad": gen_tokens_pad,                    # (B, K, T_max)
                "mask_pad": mask_pad,                                # (B, K, T_max)
                "entropy_pad": entropy_pad,                          # (B, K, T_max)
                "log_prob_pad": log_prob_pad,                        # (B, K, T_max)
                "per_sk_invalid": per_sk_invalid,                    # (B, K)
                "per_sk_reward": per_sk_reward,                      # (B, K)
                # per target
                "formula_ids": formula_ids.to(dtype=torch.long),     # (B,)
                "per_target_self_bleu": per_target_self_bleu,        # (B,)
                "per_target_has_bleu": per_target_has_bleu,          # (B,)
                "per_target_reward_mean": per_target_reward_mean,    # (B,)
                "per_target_reward_var": per_target_reward_var,      # (B,)
                "per_target_n_invalid": per_target_n_invalid,        # (B,)
                # ref-pass cpu buffers
                "_sequences_cpu": sequences_cpu,
                "_embs_cpu": embs_cpu,
                "_re_log_probs_cpu": re_log_probs_cpu,
                "_T_actual": T_actual,
            })

    # ----- Stage B: ref-model swap for KL -----
    do_kl = ref_model_path is not None and len(buffered_batches) > 0
    if do_kl:
        gen_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ref_model: LTLModel | None = None
        try:
            ref_model = LTLModel.from_pretrained(ref_model_path)
            ref_model.to(device)
            ref_model.eval()
            with torch.no_grad(), accelerator.autocast():
                for buf in buffered_batches:
                    sequences_cpu = buf.pop("_sequences_cpu")
                    embs_cpu = buf.pop("_embs_cpu")
                    re_log_probs_cpu = buf.pop("_re_log_probs_cpu")
                    T_actual = buf.pop("_T_actual")

                    shifted = sequences_cpu[:, :-1].to(device, non_blocking=True)
                    shifted_attn = (shifted != pad_id).to(dtype=torch.long)
                    # Repeat embeddings K times to match (B*K) dimension
                    embs_g = embs_cpu.to(device, non_blocking=True)
                    embs_rep = embs_g.repeat_interleave(K, dim=0)
                    re_lp_g = re_log_probs_cpu.to(device, non_blocking=True)

                    ce_logits = ref_model(
                        input_ids=shifted,
                        attention_mask=shifted_attn,
                        encoder_hidden_states=embs_rep,
                    ).logits[:, -T_actual:, :]
                    ce_log_probs = torch.log_softmax(ce_logits, dim=-1)
                    re_p = torch.exp(re_lp_g)
                    token_kl = (re_p * (re_lp_g - ce_log_probs)).sum(dim=-1)  # (B*K, T_actual)

                    B_local = int(buf["formula_ids"].size(0))
                    buf["kl_pad"] = (
                        _pad_T(token_kl.to(dtype=torch.float32), T_max, pad_value=0.0)
                        .view(B_local, K, T_max)
                    )
        finally:
            if ref_model is not None:
                del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gen_model.to(device)
    else:
        for buf in buffered_batches:
            buf.pop("_sequences_cpu", None)
            buf.pop("_embs_cpu", None)
            buf.pop("_re_log_probs_cpu", None)
            buf.pop("_T_actual", None)
            buf["kl_pad"] = torch.zeros(
                buf["entropy_pad"].size(), dtype=torch.float32, device=device
            )

    if original_training:
        gen_model.train()

    # ----- Stage C: gather, write JSONLs, aggregate -----
    n_targets = 0
    sum_self_bleu = 0.0
    sum_self_bleu_sq = 0.0
    n_self_bleu = 0
    sum_reward_mean = 0.0
    sum_reward_var = 0.0
    sum_token_entropy = 0.0
    sum_token_kl = 0.0
    sum_token_log_prob = 0.0
    sum_tokens = 0
    sum_seq_entropy_means = 0.0
    sum_seq_kl_means = 0.0
    sum_seq_log_prob_means = 0.0
    n_seq_with_tokens = 0
    sum_target_entropy_means = 0.0
    sum_target_kl_means = 0.0
    n_targets_with_tokens = 0

    flat_writer = None
    grouped_writer = None
    if _is_main(accelerator):
        os.makedirs(os.path.dirname(output_flat_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_grouped_path), exist_ok=True)
        flat_writer = open(output_flat_path, "w")
        grouped_writer = open(output_grouped_path, "w")

    try:
        for buf in buffered_batches:
            # Gather per-(target, k) tensors  (shape (B_local*K,) or (B_local*K, T_max))
            sk_gathered = accelerator.gather_for_metrics((
                buf["gen_tokens_pad"],
                buf["mask_pad"],
                buf["entropy_pad"],
                buf["log_prob_pad"],
                buf["kl_pad"],
                buf["per_sk_invalid"],
                buf["per_sk_reward"],
            ))
            (
                g_tok_ids,
                g_mask,
                g_entropy,
                g_log_prob,
                g_kl,
                g_invalid,
                g_reward,
            ) = sk_gathered

            # Gather per-target tensors  (shape (B_local,))
            t_gathered = accelerator.gather_for_metrics((
                buf["formula_ids"],
                buf["per_target_self_bleu"],
                buf["per_target_has_bleu"],
                buf["per_target_reward_mean"],
                buf["per_target_reward_var"],
                buf["per_target_n_invalid"],
            ))
            (
                g_formula_ids,
                g_self_bleu,
                g_has_bleu,
                g_reward_mean,
                g_reward_var,
                g_n_invalid,
            ) = t_gathered

            if not _is_main(accelerator):
                continue

            B_total = g_formula_ids.size(0)
            assert g_tok_ids.shape[0] == B_total and g_tok_ids.shape[1] == K, (
                f"Top-K gather shape mismatch: {tuple(g_tok_ids.shape)} vs ({B_total},{K},...)"
            )

            # Per-(target, k) flat rows
            for b_idx in range(B_total):
                fid = int(g_formula_ids[b_idx].item())
                target_formula = dataset.formulas[fid]
                target_depth = int(target_formula.depth())

                k_entropy_means: list[float] = []
                k_kl_means: list[float] = []
                target_token_entropy_sum = 0.0
                target_token_kl_sum = 0.0
                target_token_count = 0

                for k_idx in range(K):
                    mask_row = g_mask[b_idx, k_idx].to(dtype=torch.bool).cpu()
                    tok_count = int(mask_row.sum().item())
                    tok_ids = g_tok_ids[b_idx, k_idx].cpu().tolist()[:tok_count]
                    ent_row = g_entropy[b_idx, k_idx].cpu().tolist()[:tok_count]
                    lp_row = g_log_prob[b_idx, k_idx].cpu().tolist()[:tok_count]
                    kl_row = g_kl[b_idx, k_idx].cpu().tolist()[:tok_count]

                    generated_str = tokenizer.decode(tok_ids, skip_special_tokens=True)
                    is_invalid = bool(g_invalid[b_idx, k_idx].item())
                    reward = float(g_reward[b_idx, k_idx].item())

                    flat_row = {
                        "formula_id": fid,
                        "target_depth": target_depth,
                        "k_idx": k_idx,
                        "generated_formula_str": generated_str,
                        "is_invalid": is_invalid,
                        "reward": reward,
                        "token_ids": tok_ids,
                        "token_entropies": ent_row,
                        "token_log_probs": lp_row,
                        "token_kls": kl_row,
                    }
                    flat_writer.write(json.dumps(flat_row) + "\n")

                    if tok_count > 0:
                        seq_ent_mean = sum(ent_row) / tok_count
                        seq_kl_mean = sum(kl_row) / tok_count
                        seq_lp_mean = sum(lp_row) / tok_count
                        sum_seq_entropy_means += seq_ent_mean
                        sum_seq_kl_means += seq_kl_mean
                        sum_seq_log_prob_means += seq_lp_mean
                        n_seq_with_tokens += 1
                        sum_token_entropy += sum(ent_row)
                        sum_token_kl += sum(kl_row)
                        sum_token_log_prob += sum(lp_row)
                        sum_tokens += tok_count
                        k_entropy_means.append(seq_ent_mean)
                        k_kl_means.append(seq_kl_mean)
                        target_token_entropy_sum += sum(ent_row)
                        target_token_kl_sum += sum(kl_row)
                        target_token_count += tok_count

                # Per-target grouped row
                self_bleu = float(g_self_bleu[b_idx].item())
                has_bleu = bool(g_has_bleu[b_idx].item())
                reward_mean = float(g_reward_mean[b_idx].item())
                reward_var = float(g_reward_var[b_idx].item())
                n_inv = int(g_n_invalid[b_idx].item())

                target_entropy_token_mean = (
                    target_token_entropy_sum / target_token_count
                    if target_token_count > 0 else None
                )
                target_kl_token_mean = (
                    target_token_kl_sum / target_token_count
                    if target_token_count > 0 else None
                )
                target_entropy_seq_mean = (
                    sum(k_entropy_means) / len(k_entropy_means)
                    if k_entropy_means else None
                )
                target_kl_seq_mean = (
                    sum(k_kl_means) / len(k_kl_means) if k_kl_means else None
                )

                grouped_row = {
                    "formula_id": fid,
                    "target_depth": target_depth,
                    "k": K,
                    "n_invalid": n_inv,
                    "reward_mean": reward_mean,
                    "reward_variance": reward_var,
                    "self_bleu": self_bleu if has_bleu else None,
                    "policy_entropy_target_seq_mean": target_entropy_seq_mean,
                    "policy_entropy_target_token_mean": target_entropy_token_mean,
                    "kl_from_base_target_seq_mean": target_kl_seq_mean,
                    "kl_from_base_target_token_mean": target_kl_token_mean,
                }
                grouped_writer.write(json.dumps(grouped_row) + "\n")

                # Dataset-level accumulators
                n_targets += 1
                sum_reward_mean += reward_mean
                sum_reward_var += reward_var
                if has_bleu:
                    sum_self_bleu += self_bleu
                    sum_self_bleu_sq += self_bleu * self_bleu
                    n_self_bleu += 1
                if target_entropy_seq_mean is not None:
                    sum_target_entropy_means += target_entropy_seq_mean
                    sum_target_kl_means += target_kl_seq_mean
                    n_targets_with_tokens += 1
    finally:
        if flat_writer is not None:
            flat_writer.close()
        if grouped_writer is not None:
            grouped_writer.close()

    if not _is_main(accelerator):
        return {}

    summary: dict[str, Any] = {
        "n_targets": n_targets,
        "k": K,
    }
    if n_targets > 0:
        summary["reward_mean"] = sum_reward_mean / n_targets
        summary["reward_variance_within_target_mean"] = sum_reward_var / n_targets
    if n_self_bleu > 0:
        bleu_mean = sum_self_bleu / n_self_bleu
        summary["self_bleu_mean"] = bleu_mean
        summary["self_bleu_variance"] = max(
            0.0, (sum_self_bleu_sq / n_self_bleu) - bleu_mean * bleu_mean
        )
    if n_seq_with_tokens > 0:
        summary["policy_entropy_seq_mean"] = sum_seq_entropy_means / n_seq_with_tokens
        summary["kl_from_base_seq_mean"] = sum_seq_kl_means / n_seq_with_tokens
        summary["action_log_prob_seq_mean"] = sum_seq_log_prob_means / n_seq_with_tokens
    if sum_tokens > 0:
        summary["policy_entropy_token_mean"] = sum_token_entropy / sum_tokens
        summary["kl_from_base_token_mean"] = sum_token_kl / sum_tokens
        summary["action_log_prob_token_mean"] = sum_token_log_prob / sum_tokens
        summary["total_generated_tokens"] = sum_tokens
    if n_targets_with_tokens > 0:
        summary["policy_entropy_target_mean"] = sum_target_entropy_means / n_targets_with_tokens
        summary["kl_from_base_target_mean"] = sum_target_kl_means / n_targets_with_tokens
    return summary


# ---------------------------------------------------------------------------
# per-depth aggregation utility (post-processing)
# ---------------------------------------------------------------------------

def aggregate_greedy_by_depth(jsonl_path: str) -> dict[int, dict[str, float]]:
    """Read greedy per-sample JSONL and bucket aggregates by target_depth.

    Discovers depths dynamically. Returns {depth: {metric: value}}.
    """
    by_depth: dict[int, dict[str, list]] = defaultdict(lambda: {
        "distance": [],
        "exact": [],
        "equiv": [],
        "invalid": [],
        "gen_depth_valid": [],
        "gen_len_valid": [],
    })
    with open(jsonl_path, "r") as f:
        for line in f:
            r = json.loads(line)
            d = int(r["target_depth"])
            bucket = by_depth[d]
            bucket["distance"].append(float(r["semantic_distance"]))
            bucket["exact"].append(1.0 if r["is_exact_match"] else 0.0)
            bucket["equiv"].append(1.0 if r["is_semantic_equivalent"] else 0.0)
            bucket["invalid"].append(1.0 if r["is_invalid"] else 0.0)
            if not r["is_invalid"]:
                bucket["gen_depth_valid"].append(float(r["generated_depth"]))
                bucket["gen_len_valid"].append(float(r["generated_length_tokens"]))

    out: dict[int, dict[str, float]] = {}
    for d, b in by_depth.items():
        n = len(b["distance"])
        if n == 0:
            continue
        equiv_rate = sum(b["equiv"]) / n
        exact_rate = sum(b["exact"]) / n
        depth_metrics = {
            "n_samples": n,
            "semantic_distance": sum(b["distance"]) / n,
            "syntactic_equal_rate": exact_rate,
            "semantic_equivalent_rate": equiv_rate,
            "syntax_semantics_gap": equiv_rate - exact_rate,
            "invalid_rate": sum(b["invalid"]) / n,
        }
        if b["gen_depth_valid"]:
            depth_metrics["generated_depth_mean"] = (
                sum(b["gen_depth_valid"]) / len(b["gen_depth_valid"])
            )
            depth_metrics["generated_length_tokens_mean"] = (
                sum(b["gen_len_valid"]) / len(b["gen_len_valid"])
            )
        out[d] = depth_metrics
    return out


def aggregate_topk_by_depth(grouped_jsonl_path: str) -> dict[int, dict[str, float]]:
    """Read top-K grouped per-target JSONL and bucket aggregates by target_depth."""
    by_depth: dict[int, dict[str, list]] = defaultdict(lambda: {
        "reward_mean": [],
        "reward_var": [],
        "self_bleu": [],
        "entropy": [],
        "kl": [],
    })
    with open(grouped_jsonl_path, "r") as f:
        for line in f:
            r = json.loads(line)
            d = int(r["target_depth"])
            bucket = by_depth[d]
            bucket["reward_mean"].append(float(r["reward_mean"]))
            bucket["reward_var"].append(float(r["reward_variance"]))
            if r.get("self_bleu") is not None:
                bucket["self_bleu"].append(float(r["self_bleu"]))
            if r.get("policy_entropy_target_seq_mean") is not None:
                bucket["entropy"].append(float(r["policy_entropy_target_seq_mean"]))
            if r.get("kl_from_base_target_seq_mean") is not None:
                bucket["kl"].append(float(r["kl_from_base_target_seq_mean"]))

    out: dict[int, dict[str, float]] = {}
    for d, b in by_depth.items():
        n = len(b["reward_mean"])
        if n == 0:
            continue
        depth_metrics = {
            "n_targets": n,
            "reward_mean": sum(b["reward_mean"]) / n,
            "reward_variance_within_target_mean": sum(b["reward_var"]) / n,
        }
        if b["self_bleu"]:
            depth_metrics["self_bleu_mean"] = sum(b["self_bleu"]) / len(b["self_bleu"])
        if b["entropy"]:
            depth_metrics["policy_entropy_target_mean"] = sum(b["entropy"]) / len(b["entropy"])
        if b["kl"]:
            depth_metrics["kl_from_base_target_mean"] = sum(b["kl"]) / len(b["kl"])
        out[d] = depth_metrics
    return out
