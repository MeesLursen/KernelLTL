"""Norm-rescaling intervention on the conditioning signal (Experiment 3, Test 2).

Holds each validation target's embedding *direction* fixed and sweeps its
*norm*: for every scale ``c`` in ``--scales`` the decoder is run greedily on
``c * emb(phi)`` over the whole validation set. If the decoder reads the norm
as a variance signal (thesis Sec. 5.2.4), the variance of what it generates
should move with ``c`` while the relational profile stays put; if it reads
direction only, nothing should move.

The single-key cross-attention makes this a clean intervention: with one
encoder token the softmax is identically 1, so the conditioning is an affine
injection of the embedding into the residual stream and rescaling the
embedding rescales exactly that injection (model_class.py; see the thesis
discussion of the Kobayashi bound).

Per scale, writes ``<output-dir>/scale_<c>/per_sample/greedy.jsonl`` in the
Experiment 1 greedy schema (so ``analysis_exp1.load.load_greedy`` reads it
unchanged) plus the intervention fields:

  scale                 the multiplier applied to the stored embedding
  target_emb_norm       ||emb(phi)|| before scaling
  input_emb_norm        ||c * emb(phi)||, what the decoder actually received
  target_base_rate      mean of the target's satisfaction vector
  target_variance       p(1-p) of the target
  generated_base_rate   mean of the generated formula's satvec (null if invalid)
  generated_variance    p(1-p) of the generated formula (null if invalid)
  generated_emb_norm    ||emb(generated)|| under the kernel (null if invalid)
  direction_cosine      cos(emb(generated), emb(phi)), both unscaled (null if invalid)
  mean_token_entropy    mean per-step softmax entropy (nats) over the emitted tokens

``scale = 1.0`` reproduces the conditioned run and ``scale = 0.0`` the zero
ablation, so both references can be produced in the same sweep. All headline
numbers here are sanity prints; the analysis lives in ``scripts/analysis_exp3``.

Usage (single GPU)::

    python scripts/validation_variance_rescaling.py \
        --kernel-dir <kernel> --tokenizer-dir <tokenizer> \
        --eval-dataset-dir <validation_dataset> \
        --model-load-dir <trained_model> \
        --output-dir <out> --scales 0 0.25 0.5 0.75 1 1.5 2 4

Usage (multi-GPU)::

    torchrun --nproc_per_node=N scripts/validation_variance_rescaling.py ...
"""

from __future__ import annotations

import argparse
import json
import math
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
from validation_utils import _headline, _pad_to, _strip_trailing_pad


def _positive_int(value: str) -> int:
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _nonnegative_float(value: str) -> float:
    fval = float(value)
    if fval < 0.0 or not math.isfinite(fval):
        raise argparse.ArgumentTypeError("Scales must be finite and non-negative")
    return fval


def scale_dirname(scale: float) -> str:
    """Directory name for one scale; ``scale_1`` / ``scale_0.25`` / ``scale_1.5``."""
    return f"scale_{scale:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the embedding norm at fixed direction and record what the decoder generates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--eval-dataset-dir", required=True)
    parser.add_argument("--model-load-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--scales", type=_nonnegative_float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0],
                        help="Multipliers applied to every stored embedding. 1 is the "
                             "conditioned run, 0 the zero ablation.")

    parser.add_argument("--per-device-eval-batch-size", type=_positive_int, default=96)
    parser.add_argument("--semantic-eval-batch-size", type=_positive_int, default=10240)
    parser.add_argument("--embedding-batch-size", type=_positive_int, default=256,
                        help="Rows per kernel matmul when embedding the generated satvecs.")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading (mirrors validate_model.py)
# --------------------------------------------------------------------------- #

def _load_kernel(kernel_dir: str) -> LTLKernel:
    kernel = LTLKernel.load(kernel_dir)
    if kernel.F is None or kernel.traces is None or kernel.m is None:
        raise RuntimeError(
            "Kernel must include traces, anchor formulas, and the feature matrix F. "
            "Recreate it via prepare_kernel.py."
        )
    return kernel


def _load_dataset(path: str) -> LTLDataset:
    dataset = LTLDataset.load(path, load_satisfactions=True)
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {path} is empty")
    if dataset.satisfactions is None:
        raise ValueError(f"Dataset at {path} is missing per-sample satisfaction tensors.")
    if dataset.embeddings is None:
        raise ValueError(f"Dataset at {path} is missing embeddings; the sweep rescales them.")
    return dataset


def _build_dataloader(*, dataset: LTLDataset, tokenizer: LTLTokenizer,
                      model_n_positions: int, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: tokenizer.collate_batch(b, max_len=model_n_positions, include_metadata=True),
    )


def _build_model(model_dir: str, kernel: LTLKernel) -> LTLModel:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model = LTLModel.from_pretrained(model_dir)
    if model.config.n_embd != kernel.m and model.semantic_emb_dim != kernel.m:
        raise ValueError(
            f"Loaded model embedding dim ({model.config.n_embd}) does not match "
            f"kernel anchor count ({kernel.m})."
        )
    return model


# --------------------------------------------------------------------------- #
# Scoring: the Experiment 1 outcome plus the generated satvec it discards
# --------------------------------------------------------------------------- #

def _score_and_keep_satvec(
    *, generated_str: str, target_sat: torch.Tensor, kernel: LTLKernel, semantic_eval_batch_size: int,
) -> tuple[dict[str, float | bool | int], torch.Tensor | None]:
    """As ``validation_utils._score_one_generated`` but also returns the generated satvec."""
    out: dict[str, float | bool | int] = {
        "is_invalid": False, "is_semantic_equivalent": False,
        "semantic_distance": 1.0, "generated_depth": 0,
    }
    try:
        gen_formula = str_to_formula(generated_str)
        out["generated_depth"] = int(gen_formula.depth())
        gen_sat = kernel._evaluate_formula_on_traces(formula=gen_formula, batch_size=semantic_eval_batch_size)
        distance = float(torch.logical_xor(target_sat, gen_sat).to(torch.float32).mean().item())
        out["semantic_distance"] = distance
        out["is_semantic_equivalent"] = distance == 0.0
        return out, gen_sat
    except ParseError:
        out["is_invalid"] = True
        return out, None


def _mean_token_entropy(scores: tuple[torch.Tensor, ...], sequences: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mean per-step softmax entropy (nats) over each row's emitted tokens.

    ``scores[t]`` is the (B, vocab) logit block behind generated token ``t``;
    ``sequences[:, 1:]`` are those tokens, padded after EOS. Steps at pad are masked
    out, so the EOS step itself counts and nothing after it does.
    """
    logits = torch.stack(scores, dim=1).to(torch.float32)          # (B, L, V)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)                           # (B, L)
    emitted = sequences[:, 1:1 + ent.size(1)] != pad_id             # (B, L)
    n = emitted.sum(dim=1).clamp_min(1).to(torch.float32)
    return (ent * emitted).sum(dim=1) / n


def _embed_generated(
    *, gen_sats: list[torch.Tensor | None], kernel: LTLKernel, batch_size: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel embeddings of the valid generated satvecs; NaN rows where invalid.

    Returns ``(emb, base_rate)`` with ``emb`` of shape (n, m) and ``base_rate`` (n,).
    """
    n = len(gen_sats)
    m = int(kernel.m)
    emb = torch.full((n, m), float("nan"), dtype=torch.float32, device=device)
    base = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    valid = [j for j, s in enumerate(gen_sats) if s is not None]
    for start in range(0, len(valid), batch_size):
        rows = valid[start:start + batch_size]
        sats = torch.stack([gen_sats[j] for j in rows], dim=0)      # (b, N) bool
        emb[rows] = kernel.compute_embeddings_from_satisfactions(sats, move_to_cpu=False).to(device)
        base[rows] = sats.to(torch.float32).mean(dim=1).to(device)
    return emb, base


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)


def _finite_or_none(x: float) -> float | None:
    return None if not math.isfinite(x) else x


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #

def run_rescaled_greedy_pass(
    *, model: LTLModel, eval_dataloader: DataLoader, kernel: LTLKernel, tokenizer: LTLTokenizer,
    dataset: LTLDataset, accelerator: Accelerator, scale: float, output_jsonl_path: str,
    semantic_eval_batch_size: int, embedding_batch_size: int,
) -> dict[str, Any]:
    """Greedy pass on ``scale * emb(phi)`` -> one record per target."""
    device = accelerator.device
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    T_max = int(model.config.n_positions)

    gen_model = model.module if hasattr(model, "module") else model
    was_training = bool(gen_model.training)
    gen_model.eval()

    writer = None
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        writer = open(output_jsonl_path, "w")

    total = n_invalid = n_equiv = 0
    sum_distance = 0.0
    sum_gen_var = 0.0
    n_gen_var = 0

    try:
        with torch.no_grad(), accelerator.autocast():
            for batch in eval_dataloader:
                embs = batch["encoder_hidden_states"].to(device, non_blocking=True).to(torch.float32)
                target_sats = batch["target_satisfaction"].to(device)
                formula_ids = batch["formula_ids"].to(device)
                B = embs.size(0)

                target_norm = embs.norm(dim=1)                                   # (B,)
                target_p = target_sats.to(torch.float32).mean(dim=1)             # (B,)
                scaled = embs * float(scale)

                out = gen_model.generate(
                    encoder_hidden_states=scaled,
                    do_sample=False, num_beams=1, max_new_tokens=T_max,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=pad_id, eos_token_id=eos_id,
                )
                sequences = out.sequences
                entropy = _mean_token_entropy(out.scores, sequences, pad_id)     # (B,)
                generated_strs = tokenizer.batch_decode(sequences.detach().cpu(), skip_special_tokens=True)

                is_invalid = torch.zeros(B, dtype=torch.bool, device=device)
                is_equiv = torch.zeros(B, dtype=torch.bool, device=device)
                distance = torch.ones(B, dtype=torch.float32, device=device)
                gen_depth = torch.zeros(B, dtype=torch.long, device=device)
                gen_sats: list[torch.Tensor | None] = []
                for j in range(B):
                    oc, sat = _score_and_keep_satvec(
                        generated_str=generated_strs[j], target_sat=target_sats[j],
                        kernel=kernel, semantic_eval_batch_size=semantic_eval_batch_size)
                    is_invalid[j] = oc["is_invalid"]
                    is_equiv[j] = oc["is_semantic_equivalent"]
                    distance[j] = oc["semantic_distance"]
                    gen_depth[j] = oc["generated_depth"]
                    gen_sats.append(sat)

                gen_emb, gen_p = _embed_generated(
                    gen_sats=gen_sats, kernel=kernel, batch_size=embedding_batch_size, device=device)
                gen_norm = gen_emb.norm(dim=1)                                   # NaN where invalid
                dir_cos = _cosine(gen_emb, embs)                                 # NaN where invalid

                seq_pad = _pad_to(sequences.to(torch.long), T_max + 1, pad_id)

                gathered = accelerator.gather_for_metrics((
                    formula_ids, seq_pad, is_invalid, is_equiv, distance, gen_depth,
                    target_norm, target_p, gen_p, gen_norm, dir_cos, entropy,
                ))
                if not accelerator.is_main_process:
                    continue
                (g_fid, g_seq, g_inv, g_equiv, g_dist, g_depth,
                 g_tnorm, g_tp, g_gp, g_gnorm, g_cos, g_ent) = gathered

                for i in range(g_fid.size(0)):
                    fid = int(g_fid[i].item())
                    target_formula = dataset.formulas[fid]
                    inv = bool(g_inv[i].item())
                    eq = bool(g_equiv[i].item())
                    dist = float(g_dist[i].item())
                    token_ids = _strip_trailing_pad(g_seq[i].cpu().tolist(), pad_id)
                    generated_str = tokenizer.decode(token_ids, skip_special_tokens=True)
                    tp = float(g_tp[i].item())
                    gp = float(g_gp[i].item())
                    gen_var = gp * (1.0 - gp) if math.isfinite(gp) else float("nan")

                    writer.write(json.dumps({
                        # Experiment 1 greedy schema
                        "formula_id": fid,
                        "target_formula_str": str(target_formula),
                        "target_depth": int(target_formula.depth()),
                        "generated_formula_str": generated_str,
                        "generated_depth": (None if inv else int(g_depth[i].item())),
                        "is_invalid": inv,
                        "is_semantic_equivalent": eq,
                        "semantic_distance": dist,
                        "token_ids": token_ids,
                        # intervention fields
                        "scale": float(scale),
                        "target_emb_norm": float(g_tnorm[i].item()),
                        "input_emb_norm": float(scale) * float(g_tnorm[i].item()),
                        "target_base_rate": tp,
                        "target_variance": tp * (1.0 - tp),
                        "generated_base_rate": _finite_or_none(gp),
                        "generated_variance": _finite_or_none(gen_var),
                        "generated_emb_norm": _finite_or_none(float(g_gnorm[i].item())),
                        "direction_cosine": _finite_or_none(float(g_cos[i].item())),
                        "mean_token_entropy": float(g_ent[i].item()),
                    }) + "\n")

                    total += 1
                    sum_distance += dist
                    n_invalid += int(inv)
                    n_equiv += int(eq)
                    if math.isfinite(gen_var):
                        sum_gen_var += gen_var
                        n_gen_var += 1
    finally:
        if writer is not None:
            writer.close()
        if was_training:
            gen_model.train()

    if not accelerator.is_main_process:
        return {}
    summary = _headline(total, n_invalid, n_equiv, sum_distance)
    summary["scale"] = float(scale)
    summary["mean_generated_variance"] = (sum_gen_var / n_gen_var) if n_gen_var else None
    return summary


def main() -> None:
    args = parse_args()

    mixed_precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed + accelerator.process_index)

    scales = sorted(set(args.scales))
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    kernel = _load_kernel(args.kernel_dir)
    tokenizer = LTLTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = _load_dataset(args.eval_dataset_dir)
    model = _build_model(args.model_load_dir, kernel)

    if hasattr(kernel, "set_device"):
        kernel.set_device(accelerator.device)
    model.to(accelerator.device)

    n_positions = int(model.config.n_positions)
    if n_positions > 0:
        tokenizer.model_max_length = n_positions

    dataloader = _build_dataloader(
        dataset=dataset, tokenizer=tokenizer, model_n_positions=n_positions,
        batch_size=args.per_device_eval_batch_size, num_workers=args.dataloader_num_workers)
    dataloader = accelerator.prepare(dataloader)

    manifest: dict[str, Any] = {
        "model_load_dir": args.model_load_dir,
        "eval_dataset_dir": args.eval_dataset_dir,
        "kernel_dir": args.kernel_dir,
        "n_dataset_samples": len(dataset),
        "scales": scales,
        "decoding": {"do_sample": False, "num_beams": 1},
        "seed": args.seed,
        "per_scale": {},
    }

    for scale in scales:
        run_dir = os.path.join(args.output_dir, scale_dirname(scale))
        greedy_jsonl = os.path.join(run_dir, "per_sample", "greedy.jsonl")
        if accelerator.is_main_process:
            print("=" * 60)
            print(f"Greedy pass at scale {scale:g}  ->  {greedy_jsonl}")
            print("=" * 60)
        summary = run_rescaled_greedy_pass(
            model=model, eval_dataloader=dataloader, kernel=kernel, tokenizer=tokenizer,
            dataset=dataset, accelerator=accelerator, scale=scale, output_jsonl_path=greedy_jsonl,
            semantic_eval_batch_size=args.semantic_eval_batch_size,
            embedding_batch_size=args.embedding_batch_size,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            manifest["per_scale"][scale_dirname(scale)] = summary
            with open(os.path.join(run_dir, "validation_summary.json"), "w") as f:
                json.dump({"scale": scale, "greedy": summary,
                           "model_load_dir": args.model_load_dir,
                           "eval_dataset_dir": args.eval_dataset_dir}, f, indent=2)
            print("[summary]", json.dumps(summary, indent=2))

    if accelerator.is_main_process:
        path = os.path.join(args.output_dir, "rescaling_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("=" * 60)
        print(f"Wrote: {path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
