from __future__ import annotations

import torch
from transformers import Trainer

from formula_utils import str_to_formula
from kernel_class import LTLKernel
from tokenizer_pretrained_class import LTLTokenizer


class HybridTrainer(Trainer):
    """Trainer that mixes CE loss with a REINFORCE signal over kernel NMSE."""

    def __init__(
        self,
        *args,
        kernel: LTLKernel,
        tokenizer: LTLTokenizer,
        reinforce_weight: float = 0.1,
        baseline_momentum: float = 0.9,
        reward_clip: float | None = 1.0,
        rng: torch.Generator | None = None,
        inspect: bool = False,
        inspect_sample_count: int = 5,
        **kwargs,
    ) -> None:
        self.processing_class = kwargs.pop("processing_class", None)
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = tokenizer
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.formula_tokenizer = tokenizer
        self.reinforce_weight = reinforce_weight
        self.baseline_momentum = baseline_momentum
        self.reward_clip = reward_clip
        self.rng = rng
        self._reward_baseline: float | None = None
        self._nmse_eps: float = 1e-8
        self._last_reward_mean: float | None = None
        self._last_reward_std: float | None = None
        self._last_valid_ratio: float | None = None
        self.inspect = inspect
        self.inspect_sample_count = max(1, inspect_sample_count)

    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        outputs = model(**inputs)

        # ----------- CE loss -----------
        ce_term = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]

        # ----------- Weight calculation -----------
        effective_weight = 0.0
        if self.reinforce_weight > 0:
            warmup_steps = getattr(self.args, "warmup_steps", 0)
            step = getattr(self.state, "global_step", 0)

            if warmup_steps <= 0:
                effective_weight = self.reinforce_weight
            else:
                alpha = min(step / float(warmup_steps), 1.0)
                effective_weight = self.reinforce_weight * alpha

        # ----------- REINFORCE loss -----------
        reinforce_term = None
        if effective_weight > 0.0:
            attention_mask = inputs.get("attention_mask")
            semantic_embeddings = inputs.get("semantic_embeddings")
            target_token_ids = inputs.get("labels")
            if semantic_embeddings is not None:
                max_length = None
                if attention_mask is not None:
                    max_length = int(attention_mask.size(-1))
                elif hasattr(model, "config") and getattr(model.config, "n_positions", None):
                    max_length = int(getattr(model.config, "n_positions"))
                if max_length is None:
                    max_length = 32

                reinforce_term = self._compute_reinforce_term(
                    model=model,
                    semantic_embeddings=semantic_embeddings,
                    target_token_ids=target_token_ids,
                    generation_max_length=max_length,
                )

        # ----------- Combine losses -----------
        if reinforce_term is None or effective_weight <= 0.0:
            loss = ce_term
        elif effective_weight >= 1.0 or ce_term is None:
            loss = reinforce_term
        else:
            loss = (1 - effective_weight) * ce_term + effective_weight * reinforce_term

        # ----------- Logging -----------
        logging_steps = getattr(self.args, "logging_steps", None)
        step = getattr(self.state, "global_step", None)
        if (
            logging_steps is not None
            and logging_steps > 0
            and step is not None
            and step % logging_steps == 0
        ):
            log_payload: dict[str, float] = {
                "loss": loss.detach().item(),
                "reinforce_effective_weight": float(effective_weight),
            }
            if ce_term is not None:
                log_payload["loss_ce"] = ce_term.detach().item()
            if reinforce_term is not None:
                log_payload["loss_reinforce"] = reinforce_term.detach().item()
            else:
                log_payload["loss_reinforce"] = 'None'
            if self._last_reward_mean is not None:
                log_payload["reward_mean"] = self._last_reward_mean
            if self._last_reward_std is not None:
                log_payload["reward_std"] = self._last_reward_std
            if self._last_valid_ratio is not None:
                log_payload["valid_ratio"] = self._last_valid_ratio
            self.log(log_payload)

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_reinforce_term(
        self,
        *,
        model,
        semantic_embeddings: torch.Tensor,
        generation_max_length: int,
        target_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if semantic_embeddings is None or semantic_embeddings.ndim < 2:
            print("[HybridTrainer] RL: semantic_embeddings invalid -> returning None")
            return None

        generation_max_length = max(1, int(generation_max_length))
        device = semantic_embeddings.device
        pad_id = getattr(self.formula_tokenizer, "pad_token_id", None)
        eos_id = getattr(self.formula_tokenizer, "eos_token_id", None)
        bos_id = getattr(self.formula_tokenizer, "bos_token_id", None)

        generate_kwargs: dict[str, object] = {
            "semantic_embeddings": semantic_embeddings,
            "do_sample": True,
            "max_new_tokens": generation_max_length,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "output_scores": True,
            "temperature": 1.0,
        }
        if pad_id is not None:
            generate_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            generate_kwargs["eos_token_id"] = eos_id
        if bos_id is not None:
            generate_kwargs["bos_token_id"] = bos_id

        try:
            generation = model.generate(**generate_kwargs)
        except Exception as e:
            print("[HybridTrainer] RL: model.generate failed:", repr(e))
            return None

        sequences = getattr(generation, "sequences", None)
        scores = getattr(generation, "scores", None)
        if sequences is None or scores is None or len(scores) == 0:
            print("[HybridTrainer] RL: empty sequences/scores -> returning None")
            return None

        if isinstance(scores, tuple):
            scores = list(scores)

        total_steps = len(scores)
        seq_len = sequences.size(-1)
        prefix_len = max(0, seq_len - total_steps)
        generated_tokens = sequences[:, prefix_len:].long()
        if generated_tokens.size(-1) > total_steps:
            generated_tokens = generated_tokens[:, :total_steps]

        score_tensor = torch.stack(scores, dim=0).transpose(0, 1)  # (B, T, V)
        log_probs = torch.log_softmax(score_tensor, dim=-1)
        generated_tokens = generated_tokens.to(log_probs.device)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        if pad_id is not None:
            gen_mask = (generated_tokens != pad_id).to(log_probs.dtype)
        else:
            gen_mask = torch.ones_like(generated_tokens, dtype=log_probs.dtype)
        lengths = gen_mask.sum(dim=-1).clamp(min=1.0)
        seq_log_prob = (token_log_probs * gen_mask).sum(dim=-1) / lengths

        generated_tokens_cpu = generated_tokens.detach().cpu()
        try:
            generated_strings = self.formula_tokenizer.batch_decode(
                generated_tokens_cpu, skip_special_tokens=True
            )
        except Exception as e:
            print("[HybridTrainer] RL: batch_decode failed:", repr(e))
            return None

        target_strings: list[str] | None = None
        if target_token_ids is not None:
            target_ids = target_token_ids.detach().cpu()
            if target_ids.ndim == 1:
                target_ids = target_ids.unsqueeze(0)
            pad_id = getattr(self.formula_tokenizer, "pad_token_id", 0) or 0
            target_ids = target_ids.clone()
            target_ids[target_ids < 0] = pad_id
            try:
                target_strings = self.formula_tokenizer.batch_decode(target_ids.tolist(), skip_special_tokens=True)
            except Exception:
                target_strings = None

        reward_values: list[torch.Tensor] = []
        valid_count = 0
        generated_embeds: list[torch.Tensor | None] = []
        target_embeds_cpu: list[torch.Tensor] = []

        with torch.no_grad():
            for generated_str, target_emb in zip(generated_strings, semantic_embeddings):
                target_vec = target_emb.detach()
                try:
                    generated_formula = str_to_formula(generated_str)
                    generated_vec = self.kernel.compute_formula_embedding_no_move(generated_formula)
                    generated_vec = generated_vec.to(device=device, dtype=torch.float32, non_blocking=True)
                    target_vec = target_vec.to(device=device, dtype=torch.float32, non_blocking=True)
                    diff = generated_vec - target_vec
                    nmse = diff.pow(2).sum() / (target_vec.pow(2).sum() + self._nmse_eps)
                    reward = 1.0 - nmse
                    valid_count += 1
                    generated_embeds.append(generated_vec.detach().cpu())
                    target_embeds_cpu.append(target_vec.detach().cpu())
                except Exception:
                    reward = torch.tensor(-1.0, device=device)
                    generated_embeds.append(None)
                    target_embeds_cpu.append(target_vec.detach().cpu())
                if self.reward_clip is not None:
                    reward = torch.clamp(reward, min=-self.reward_clip, max=self.reward_clip)
                reward_values.append(reward)

        reward_tensor = torch.stack(reward_values).to(device=device, dtype=seq_log_prob.dtype)
        reward_mean = reward_tensor.mean().item()
        reward_std = reward_tensor.std(unbiased=False).item() if reward_tensor.numel() > 1 else 0.0
        self._last_reward_mean = reward_mean
        self._last_reward_std = reward_std
        self._last_valid_ratio = float(valid_count / len(reward_values)) if reward_values else 0.0

        logging_steps = getattr(self.args, "logging_steps", None)
        step = getattr(self.state, "global_step", None)
        should_inspect = (
            self.inspect
            and logging_steps
            and step
            and step % logging_steps == 0
        )
        if should_inspect:
            sample_total = min(self.inspect_sample_count, len(generated_strings))
            print(f"[HybridTrainer] Inspection at step {step}: showing {sample_total} samples")
            for idx in range(sample_total):
                gen_str = generated_strings[idx]
                tgt_str = target_strings[idx] if target_strings is not None and idx < len(target_strings) else "<unknown>"
                reward_val = reward_tensor[idx].item() if idx < reward_tensor.numel() else float("nan")
                # gen_vec = generated_embeds[idx]
                # tgt_vec = target_embeds_cpu[idx]
                print(f"  Sample {idx + 1}:")
                print(f"    Target   : {tgt_str}")
                print(f"    Generated: {gen_str}")
                print(f"    Reward   : {reward_val:.4f}")
                # if gen_vec is not None:
                #     print(f"    Embeddings:")
                #     print(f"      target: {tgt_vec.numpy()}")
                #     print(f"      gen   : {gen_vec.numpy()}")
                # else:
                #     print("    Embeddings: generated embedding unavailable (invalid formula)")
            print()

        if self._reward_baseline is None:
            self._reward_baseline = reward_mean
        else:
            self._reward_baseline = (
                self.baseline_momentum * self._reward_baseline
                + (1.0 - self.baseline_momentum) * reward_mean
            )
        baseline = torch.tensor(self._reward_baseline, device=device, dtype=reward_tensor.dtype)
        advantage = (reward_tensor - baseline).detach()

        reinforce_loss = -(advantage * seq_log_prob).mean()
        if torch.isnan(reinforce_loss):
            print("[HybridTrainer] reinforce_loss is NaN")
            print("  reward_tensor:", reward_tensor)
            print("  baseline:", baseline)
            print("  seq_log_prob:", seq_log_prob) # WIP: Run and test this 
            return None
        return reinforce_loss
