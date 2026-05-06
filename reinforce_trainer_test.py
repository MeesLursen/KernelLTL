import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Trainer

from formula_utils import str_to_formula
from kernel_class import LTLKernel
from tokenizer_pretrained_class import LTLTokenizer


class REINFORCETrainerRB(Trainer):

    def __init__(
            self,
            *args,
            kernel: LTLKernel,
            tokenizer: LTLTokenizer,
            baseline_momentum: float = 0.9,
            reward_clip: float | None = 1.0,
            semantic_eval_batch_size: int = 10240,
            satisfactions_mmap: bool = False,
            **kwargs,
        ) -> None:
            self.processing_class = kwargs.pop("processing_class", None)
            if "tokenizer" not in kwargs:
                kwargs["tokenizer"] = tokenizer
            super().__init__(*args, **kwargs)
            self.kernel = kernel
            self.trainer_kind = 'rb'
            self.formula_tokenizer = tokenizer
            self.baseline_momentum = baseline_momentum
            self.reward_clip = reward_clip
            self.semantic_eval_batch_size = semantic_eval_batch_size
            self._reward_baseline: torch.Tensor | None = None
            self._reward_sq_mean: torch.Tensor | None = None
            self._last_train_metrics: dict[str, float | torch.Tensor] = {}
            self._last_rl_metrics: dict[str, float | torch.Tensor] = {}
            self._sync_kernel_device(getattr(self.args, "device", None))



    # ------------------------------- CORE LOSS METHODS -------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        self._last_rl_metrics = {}
        # ----------- REINFORCE loss -----------
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        attention_mask = inputs.get("attention_mask")
        batch_target_satisfaction = inputs.get("target_satisfaction")
        generation_max_length: int | None = None
        if encoder_hidden_states is not None:
            if attention_mask is not None:
                generation_max_length = int(attention_mask.size(-1))
            elif hasattr(model, "config") and getattr(model.config, "n_positions", None):
                generation_max_length = int(getattr(model.config, "n_positions"))
            else:
                generation_max_length = 512

        reinforce_loss = None
        valid_mask = None
        if (
            encoder_hidden_states is not None
            and generation_max_length is not None
        ):
            reinforce_loss, valid_mask = self._compute_reinforce_term(
                model=model,
                encoder_hidden_states=encoder_hidden_states,
                batch_target_satisfaction=batch_target_satisfaction,
                generation_max_length=generation_max_length,
                require_grad=model.training,
            )

        local_needs_dummy_rl = bool(reinforce_loss is None or valid_mask is None)
        global_needs_dummy_rl = local_needs_dummy_rl
        if dist.is_available() and dist.is_initialized():
            dummy_flag = torch.tensor(
                [1 if local_needs_dummy_rl else 0],
                device=torch.device(getattr(self.args, "device", "cpu")),
                dtype=torch.int32,
            )
            dist.all_reduce(dummy_flag, op=dist.ReduceOp.MAX)
            global_needs_dummy_rl = bool(dummy_flag.item())

        if global_needs_dummy_rl:
            ref_tensor = next(
                (value for value in inputs.values() if torch.is_tensor(value) and value.ndim > 0),
                None,
            )
            if ref_tensor is None or ref_tensor.size(0) == 0:
                raise RuntimeError("Could not build dummy RL anchor because batch tensors are missing or empty.")
            dummy_mask = torch.zeros(ref_tensor.size(0), dtype=torch.bool, device=ref_tensor.device)
            dummy_mask[0] = True
            dummy_inputs = self._slice_inputs_by_mask(inputs, dummy_mask)
            dummy_outputs = model(**dummy_inputs)
            dummy_ce_loss = (
                dummy_outputs.loss
                if hasattr(dummy_outputs, "loss") and dummy_outputs.loss is not None
                else dummy_outputs[0]
            )
            if local_needs_dummy_rl:
                reinforce_loss = dummy_ce_loss * 0.0
                valid_mask = torch.zeros(ref_tensor.size(0), dtype=torch.bool, device=reinforce_loss.device)

        # ----------- Combine losses -----------
        valid_mask = valid_mask.to(device=reinforce_loss.device)
        invalid_mask = ~valid_mask
        loss_terms: list[torch.Tensor] = []
        valid_ratio = valid_mask.to(dtype=reinforce_loss.dtype).mean()
        invalid_ratio = reinforce_loss.detach().new_ones(()) - valid_ratio.detach()
        ce_invalid_metric = reinforce_loss.detach().new_zeros(())

        loss_terms.append(valid_ratio * reinforce_loss)

        local_has_invalid = bool(invalid_mask.any())
        if local_has_invalid:
            invalid_inputs = self._slice_inputs_by_mask(inputs, invalid_mask)
            invalid_weight = invalid_ratio
        else:
            dummy_mask = torch.zeros_like(invalid_mask, dtype=torch.bool)
            if dummy_mask.numel() > 0:
                dummy_mask[0] = True
            invalid_inputs = self._slice_inputs_by_mask(inputs, dummy_mask)
            invalid_weight = invalid_ratio.detach().new_zeros(())

        invalid_outputs = model(**invalid_inputs)
        ce_invalid_loss = (
            invalid_outputs.loss
            if hasattr(invalid_outputs, "loss") and invalid_outputs.loss is not None
            else invalid_outputs[0]
        )
        if local_has_invalid:
            ce_invalid_metric = ce_invalid_loss.detach()
        loss_terms.append(invalid_weight * ce_invalid_loss)

        loss = sum(loss_terms)

        self._last_train_metrics = {
            "train_loss": loss.detach(),
            "train_valid_ratio": valid_ratio.detach(),
            "train_invalid_ratio": invalid_ratio,
            "train_rl_loss": reinforce_loss.detach(),
            "train_ce_loss": ce_invalid_metric,
        }
        self._last_train_metrics.update(self._last_rl_metrics)

        if return_outputs:
            outputs = model(**inputs)
            return loss, outputs
        return loss


    def _compute_reinforce_term(
        self,
        *,
        model,
        encoder_hidden_states: torch.Tensor,
        generation_max_length: int,
        batch_target_satisfaction: torch.Tensor | None = None,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if encoder_hidden_states is None or encoder_hidden_states.ndim < 2:
            print("[REINFORCETrainer] RL: encoder_hidden_states invalid -> returning None")
            return None, None

        if batch_target_satisfaction is None:
            print("[REINFORCETrainer] RL: target_satisfaction missing -> returning None")
            return None, None

        device = self.args.device
        if not torch.is_tensor(batch_target_satisfaction):
            print("[REINFORCETrainer] RL: target_satisfaction has unsupported type -> returning None")
            return None, None
        if batch_target_satisfaction.ndim == 1:
            batch_target_satisfaction = batch_target_satisfaction.unsqueeze(0)
        if batch_target_satisfaction.ndim != 2:
            print("[REINFORCETrainer] RL: target_satisfaction must be rank-2 -> returning None")
            return None, None
        target_satisfactions = batch_target_satisfaction.to(device=device, dtype=torch.bool, non_blocking=True)

        generation_max_length = max(1, int(generation_max_length))
        pad_id = getattr(self.formula_tokenizer, "pad_token_id", None)
        eos_id = getattr(self.formula_tokenizer, "eos_token_id", None)
        bos_id = getattr(self.formula_tokenizer, "bos_token_id", None)

        generate_kwargs: dict[str, object] = {
            "encoder_hidden_states": encoder_hidden_states,
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

        gen_model = model.module if hasattr(model, 'module') else model

        try:
            generation = gen_model.generate(**generate_kwargs)
        except Exception as e:
            print("[REINFORCETrainer] RL: model.generate failed:", repr(e))
            return None, None

        sequences = getattr(generation, "sequences", None)
        scores = getattr(generation, "scores", None)
        if sequences is None or scores is None or len(scores) == 0:
            print("[REINFORCETrainer] RL: empty sequences/scores -> returning None")
            return None, None

        if isinstance(scores, tuple):
            scores = list(scores)

        total_steps = len(scores)
        seq_len = sequences.size(-1)
        prefix_len = max(0, seq_len - total_steps)
        generated_tokens = sequences[:, prefix_len:].long()
        if generated_tokens.size(-1) > total_steps:
            generated_tokens = generated_tokens[:, :total_steps]

        score_tensor = torch.stack(scores, dim=0).transpose(0, 1)  # (B, T, V)
        score_log_probs = torch.log_softmax(score_tensor, dim=-1)
        score_probs = torch.exp(score_log_probs)
        token_entropy = -(score_probs * score_log_probs).sum(dim=-1)

        if require_grad:
            token_log_probs = self._recompute_log_probs_with_grad(
                model=model,
                sequences=sequences,
                generated_tokens=generated_tokens,
                encoder_hidden_states=encoder_hidden_states,
                prefix_len=prefix_len,
                pad_id=pad_id,
            )
        else:
            token_log_probs = score_log_probs.gather(
                dim=-1, index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)

        token_mask = (generated_tokens != pad_id)
        token_mask_f = token_mask.to(dtype=token_log_probs.dtype)
        token_count_per_sample = token_mask_f.sum(dim=1).detach()
        token_entropy_sum = ((token_entropy * token_mask_f).sum(dim=1)).detach()
        train_action_log_prob_sum = ((score_log_probs.amax(dim=-1)) * token_mask_f).sum(dim=1).detach()

        generated_tokens_cpu = generated_tokens.detach().cpu()
        try:
            generated_strings = self.formula_tokenizer.batch_decode(
                generated_tokens_cpu, skip_special_tokens=True
            )
        except Exception as e:
            print("[REINFORCETrainer] RL: batch_decode failed:", repr(e))
            return None, None

        reward_tensor = torch.zeros(len(generated_strings), dtype=token_log_probs.dtype, device=device)
        valid_mask = torch.zeros(len(generated_strings), dtype=torch.bool, device=device)

        with torch.no_grad():
            for i, generated_str in enumerate(generated_strings):
                try:
                    generated_formula = str_to_formula(generated_str)
                    generated_sats = self.kernel._evaluate_formula_on_traces(generated_formula, self.semantic_eval_batch_size)
                    target_sats = target_satisfactions[i]
                    if generated_sats.numel() != target_sats.numel():
                        raise ValueError("Satisfaction length mismatch")

                    hamming = torch.logical_xor(generated_sats, target_sats).to(torch.float32).mean()
                    reward = 1.0 - hamming
                    valid_mask[i] = True
                    if self.reward_clip is not None:
                        reward = torch.clamp(reward, min=-self.reward_clip, max=self.reward_clip)
                    reward_tensor[i] = reward
                except Exception:
                    continue

        reward_valid = reward_tensor[valid_mask]
        reward_mean, reward_sq_mean = self._sync_reward_moments(reward_valid)
        if reward_mean is not None and reward_sq_mean is not None:
            if self._reward_baseline is None or self._reward_sq_mean is None:
                self._reward_baseline = reward_mean
                self._reward_sq_mean = reward_sq_mean
            else:
                momentum = float(self.baseline_momentum)
                self._reward_baseline = (
                    momentum * self._reward_baseline
                    + (1.0 - momentum) * reward_mean
                )
                self._reward_sq_mean = (
                    momentum * self._reward_sq_mean
                    + (1.0 - momentum) * reward_sq_mean
                )

        if not bool(valid_mask.any()):
            self._last_rl_metrics = {
                "token_count_per_sample": token_count_per_sample,
                "token_entropy_sum": token_entropy_sum,
                "train_action_log_prob_sum": train_action_log_prob_sum,
                "valid_formula_mask_per_sample": valid_mask.detach(),
                "reward_per_sample": reward_tensor.detach(),
                "advantage_per_sample": torch.zeros_like(reward_tensor),
            }
            return None, valid_mask

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        reward_valid = reward_tensor[valid_idx]
        token_log_probs = token_log_probs[valid_idx]
        token_mask = token_mask[valid_idx]
        token_mask_f = token_mask_f[valid_idx]
        lengths_valid = token_mask.sum(dim=-1).clamp(min=1)
        seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths_valid
        # Normalize advantage (reduces variance significantly)
        baseline = self._reward_baseline.to(device=device, dtype=reward_tensor.dtype)
        variance = (self._reward_sq_mean.to(device=device, dtype=reward_tensor.dtype) - baseline.square()).clamp(min=1e-8)
        std = variance.sqrt().clamp(min=1e-4)
        advantage_valid = ((reward_valid - baseline) / std).detach()
        advantage_per_sample = torch.zeros_like(reward_tensor)
        advantage_per_sample[valid_idx] = advantage_valid

        reinforce_loss = -(advantage_valid * seq_log_prob).mean()

        self._last_rl_metrics = {
            "token_count_per_sample": token_count_per_sample,
            "token_entropy_sum": token_entropy_sum,
            "train_action_log_prob_sum": train_action_log_prob_sum,
            "valid_formula_mask_per_sample": valid_mask.detach(),
            "reward_per_sample": reward_tensor.detach(),
            "advantage_per_sample": advantage_per_sample.detach(),
        }

        return reinforce_loss, valid_mask


    def _sync_reward_moments(
        self,
        reward_valid: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if reward_valid.numel() == 0 and not (dist.is_available() and dist.is_initialized()):
            return None, None

        local_sum = reward_valid.sum()
        local_sq_sum = (reward_valid ** 2).sum()
        local_count = reward_valid.new_tensor(float(reward_valid.numel()))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        if float(local_count.detach().cpu().item()) <= 0.0:
            return None, None

        reward_mean = (local_sum / local_count).detach()
        reward_sq_mean = (local_sq_sum / local_count).detach()
        return reward_mean, reward_sq_mean



    def _slice_inputs_by_mask(self, inputs: dict, mask: torch.Tensor) -> dict:
        sliced: dict = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == mask.size(0):
                sliced[key] = value[mask.to(device=value.device)]
            else:
                sliced[key] = value
        return sliced


    def _recompute_log_probs_with_grad(
        self,
        *,
        model,
        sequences: torch.Tensor,
        generated_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prefix_len: int,
        pad_id: int | None,
    ) -> torch.Tensor:
        target_len = generated_tokens.size(-1)
        teacher_inputs = sequences[:, : prefix_len + target_len].detach()
        shifted_inputs = teacher_inputs[:, :-1]
        if pad_id is None:
            shifted_attention_mask = torch.ones_like(shifted_inputs, dtype=torch.long)
        else:
            shifted_attention_mask = (shifted_inputs != pad_id).to(dtype=torch.long)

        with torch.enable_grad():
            outputs = model(
                input_ids=shifted_inputs,
                attention_mask=shifted_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
            logits = outputs.logits[:, -target_len:, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            gather_tokens = generated_tokens.unsqueeze(-1)
            tf_log_probs = log_probs.gather(dim=-1, index=gather_tokens).squeeze(-1)

        if pad_id is not None:
            mask = (generated_tokens != pad_id).to(tf_log_probs.dtype)
            tf_log_probs = tf_log_probs * mask

        return tf_log_probs



    # ------------------------------- MISC METHODS -------------------------------
    def _sync_kernel_device(self, device: torch.device | str | None) -> None:
        if device is None:
            return
        if self.kernel is None:
            return
        self.kernel.set_device(device)



class REINFORCETrainerGAE(Trainer):

    def __init__(
            self,
            *args,
            kernel: LTLKernel,
            tokenizer: LTLTokenizer,
            reward_clip: float | None = 1.0,
            semantic_eval_batch_size: int = 10240,
            satisfactions_mmap: bool = False,
            gae_gamma: float = 1.0,
            gae_lambda: float = 1.0,
            critic_lr: float | None = None,
            critic_hidden_dim: int = 256,
            critic_weight_decay: float = 0.0,
            critic_pretraining_steps: int = 0,
            critic_pretraining_ratio: float = 0.0,
            **kwargs,
        ) -> None:
            self.processing_class = kwargs.pop("processing_class", None)
            if "tokenizer" not in kwargs:
                kwargs["tokenizer"] = tokenizer
            super().__init__(*args, **kwargs)
            self.kernel = kernel
            self.trainer_kind = 'gae'
            self.formula_tokenizer = tokenizer
            self.reward_clip = reward_clip
            self.semantic_eval_batch_size = semantic_eval_batch_size
            self.gae_gamma = float(gae_gamma)
            self.gae_lambda = float(gae_lambda)
            self.critic_lr = float(critic_lr) if critic_lr is not None else float(self.args.learning_rate)
            self.critic_weight_decay = float(critic_weight_decay)
            self._last_train_metrics: dict[str, float | torch.Tensor] = {}
            self._last_rl_metrics: dict[str, float | torch.Tensor] = {}
            self._actor_frozen: bool = False
            self._critic_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            self._critic_cache_full: bool = False
            self._critic_cache_idx: int = 0
            self._critic_pretraining_steps: int | None = None
            self._critic_pretraining_steps_arg: int = max(0, int(critic_pretraining_steps))
            self._critic_pretraining_ratio_arg: float = max(0.0, float(critic_pretraining_ratio))

            hidden_dim = int(getattr(self.model.config, "n_embd", 0))
            if hidden_dim <= 0:
                raise ValueError("Model config must expose positive n_embd for critic construction.")

            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, critic_hidden_dim),
                nn.Tanh(),
                nn.Linear(critic_hidden_dim, 1),
            )

            self._attach_critic_to_model()
            self._sync_kernel_critic_device(getattr(self.args, "device", None))


    def _resolve_critic_pretraining_steps(self) -> int:
        if self._critic_pretraining_steps is not None:
            return self._critic_pretraining_steps

        steps = getattr(self.args, "critic_pretraining_steps", None)
        if steps is not None:
            steps = max(0, int(steps))
        else:
            steps = self._critic_pretraining_steps_arg

        if steps == 0:
            ratio = getattr(self.args, "critic_pretraining_ratio", None)
            if ratio is None:
                ratio = self._critic_pretraining_ratio_arg
            ratio = max(0.0, float(ratio))
            if ratio > 0.0:
                max_steps = int(getattr(self.state, "max_steps", 0) or 0)
                if max_steps > 0:
                    import math
                    steps = math.ceil(max_steps * ratio)
                else:
                    # max_steps not yet known; don't cache so we retry later.
                    return 0

        self._critic_pretraining_steps = steps
        return self._critic_pretraining_steps


    def _is_critic_pretraining_active(self) -> bool:
        return int(getattr(self.state, "global_step", 0) or 0) < self._resolve_critic_pretraining_steps()


    def _set_actor_requires_grad(self, requires_grad: bool) -> None:
        if self.model is None:
            return
        if requires_grad and not self._actor_frozen:
            return
        if (not requires_grad) and self._actor_frozen:
            return

        target_model = self.model.module if hasattr(self.model, "module") else self.model
        critic = getattr(target_model, "critic", None)
        critic_param_ids = {id(p) for p in critic.parameters()} if critic is not None else set()

        for p in self.model.parameters():
            if id(p) in critic_param_ids:
                p.requires_grad = True
            else:
                p.requires_grad = requires_grad
        self._actor_frozen = not requires_grad




    # ------------------------------- CORE LOSS METHODS -------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        if self.state.global_step == 0:
            actor_lr = self.args.learning_rate
            for i, group in enumerate(self.optimizer.param_groups):
                n = len(group["params"])
                lr = group["lr"]
                print(f"  param_group[{i}]: n_params={n}, lr={lr:.2e}")
            
            critic_ids = {id(p) for p in self.critic.parameters()}
            for i, group in enumerate(self.optimizer.param_groups):
                overlap = sum(1 for p in group["params"] if id(p) in critic_ids)
                print(f"  param_group[{i}]: {overlap} critic params")
                
        critic_warmup = self._is_critic_pretraining_active()

        # Seal the cache after the first full epoch of collection
        if critic_warmup and not self._critic_cache_full and self.state.epoch >= 1.0 and self._critic_cache:
            self._critic_cache_full = True

        # Replay mode: critic pretraining is still active but the cache is ready.
        # Skip generation entirely — sample a cached entry and train the critic from it.
        if critic_warmup and self._critic_cache_full:
            if model.training:
                self._set_actor_requires_grad(False)
            device = self.args.device
            entry = self._critic_cache[self._critic_cache_idx % len(self._critic_cache)]
            self._critic_cache_idx += 1
            token_hidden, rewards, token_mask_f = [t.to(device) for t in entry]
            loss, cache_rl_metrics = self._critic_loss_from_cache(token_hidden, rewards, token_mask_f)
            self._last_rl_metrics = cache_rl_metrics
            self._last_train_metrics = {
                "train_loss": loss.detach(),
                "train_critic_loss": loss.detach(),
                "train_actor_loss": loss.detach().new_zeros(()),
            }
            self._last_train_metrics.update(cache_rl_metrics)
            if return_outputs:
                actor_model = model.module if hasattr(model, "module") else model
                with torch.no_grad():
                    outputs = actor_model(**inputs)
                return loss, outputs
            return loss

        actor_model = model.module if (critic_warmup and hasattr(model, "module")) else model
        if model.training:
            self._set_actor_requires_grad(not critic_warmup)
        self._last_rl_metrics = {}
        # ----------- REINFORCE loss -----------
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        attention_mask = inputs.get("attention_mask")
        batch_target_satisfaction = inputs.get("target_satisfaction")
        generation_max_length: int | None = None
        if encoder_hidden_states is not None:
            if attention_mask is not None:
                generation_max_length = int(attention_mask.size(-1))
            elif hasattr(model, "config") and getattr(model.config, "n_positions", None):
                generation_max_length = int(getattr(model.config, "n_positions"))
            else:
                generation_max_length = 512

        reinforce_loss = None
        valid_mask = None
        if (
            encoder_hidden_states is not None
            and generation_max_length is not None
        ):
            reinforce_loss, valid_mask = self._compute_reinforce_term(
                model=actor_model,
                encoder_hidden_states=encoder_hidden_states,
                batch_target_satisfaction=batch_target_satisfaction,
                generation_max_length=generation_max_length,
                require_grad=model.training and not critic_warmup,
            )

        local_needs_dummy_rl = bool(reinforce_loss is None or valid_mask is None)
        global_needs_dummy_rl = local_needs_dummy_rl
        if dist.is_available() and dist.is_initialized():
            dummy_flag = torch.tensor(
                [1 if local_needs_dummy_rl else 0],
                device=torch.device(getattr(self.args, "device", "cpu")),
                dtype=torch.int32,
            )
            dist.all_reduce(dummy_flag, op=dist.ReduceOp.MAX)
            global_needs_dummy_rl = bool(dummy_flag.item())

        if global_needs_dummy_rl:
            ref_tensor = next(
                (value for value in inputs.values() if torch.is_tensor(value) and value.ndim > 0),
                None,
            )
            if ref_tensor is None or ref_tensor.size(0) == 0:
                raise RuntimeError("Could not build dummy RL anchor because batch tensors are missing or empty.")
            dummy_mask = torch.zeros(ref_tensor.size(0), dtype=torch.bool, device=ref_tensor.device)
            dummy_mask[0] = True
            dummy_inputs = self._slice_inputs_by_mask(inputs, dummy_mask)
            if critic_warmup:
                with torch.no_grad():
                    dummy_outputs = actor_model(**dummy_inputs)
            else:
                dummy_outputs = actor_model(**dummy_inputs)
            dummy_ce_loss = (
                dummy_outputs.loss
                if hasattr(dummy_outputs, "loss") and dummy_outputs.loss is not None
                else dummy_outputs[0]
            )
            if local_needs_dummy_rl:
                reinforce_loss = dummy_ce_loss * 0.0
                valid_mask = torch.zeros(ref_tensor.size(0), dtype=torch.bool, device=reinforce_loss.device)

        # ----------- Combine losses -----------
        valid_mask = valid_mask.to(device=reinforce_loss.device)
        invalid_mask = ~valid_mask
        loss_terms: list[torch.Tensor] = []
        valid_ratio = valid_mask.to(dtype=reinforce_loss.dtype).mean()
        invalid_ratio = reinforce_loss.detach().new_ones(()) - valid_ratio.detach()
        ce_invalid_metric = reinforce_loss.detach().new_zeros(())

        loss_terms.append(valid_ratio * reinforce_loss)

        local_has_invalid = bool(invalid_mask.any())
        if local_has_invalid:
            invalid_inputs = self._slice_inputs_by_mask(inputs, invalid_mask)
            invalid_weight = invalid_ratio
        else:
            dummy_mask = torch.zeros_like(invalid_mask, dtype=torch.bool)
            if dummy_mask.numel() > 0:
                dummy_mask[0] = True
            invalid_inputs = self._slice_inputs_by_mask(inputs, dummy_mask)
            invalid_weight = invalid_ratio.detach().new_zeros(())

        if critic_warmup:
            with torch.no_grad():
                invalid_outputs = actor_model(**invalid_inputs)
        else:
            invalid_outputs = actor_model(**invalid_inputs)
        ce_invalid_loss = (
            invalid_outputs.loss
            if hasattr(invalid_outputs, "loss") and invalid_outputs.loss is not None
            else invalid_outputs[0]
        )
        if local_has_invalid:
            ce_invalid_metric = ce_invalid_loss.detach()
        loss_terms.append(invalid_weight * ce_invalid_loss)

        loss = sum(loss_terms)

        self._last_train_metrics = {
            "train_loss": loss.detach(),
            "train_valid_ratio": valid_ratio.detach(),
            "train_invalid_ratio": invalid_ratio,
            "train_rl_loss": reinforce_loss.detach(),
            "train_ce_loss": ce_invalid_metric,
        }
        self._last_train_metrics.update(self._last_rl_metrics)

        if return_outputs:
            if critic_warmup:
                with torch.no_grad():
                    outputs = actor_model(**inputs)
            else:
                outputs = actor_model(**inputs)
            return loss, outputs
        return loss


    def training_step(self, model, inputs, num_items_in_batch: int | None = None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._is_critic_pretraining_active():
            self._sync_critic_gradients()
        return loss



    def _compute_reinforce_term(
        self,
        *,
        model,
        encoder_hidden_states: torch.Tensor,
        generation_max_length: int,
        batch_target_satisfaction: torch.Tensor | None = None,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if encoder_hidden_states is None or encoder_hidden_states.ndim < 2:
            print("[REINFORCETrainer] RL: encoder_hidden_states invalid -> returning None")
            return None, None

        if batch_target_satisfaction is None:
            print("[REINFORCETrainer] RL: target_satisfaction missing -> returning None")
            return None, None

        device = self.args.device
        if not torch.is_tensor(batch_target_satisfaction):
            print("[REINFORCETrainer] RL: target_satisfaction has unsupported type -> returning None")
            return None, None
        if batch_target_satisfaction.ndim == 1:
            batch_target_satisfaction = batch_target_satisfaction.unsqueeze(0)
        if batch_target_satisfaction.ndim != 2:
            print("[REINFORCETrainer] RL: target_satisfaction must be rank-2 -> returning None")
            return None, None
        target_satisfactions = batch_target_satisfaction.to(device=device, dtype=torch.bool, non_blocking=True)

        generation_max_length = max(1, int(generation_max_length))
        pad_id = getattr(self.formula_tokenizer, "pad_token_id", None)
        eos_id = getattr(self.formula_tokenizer, "eos_token_id", None)
        bos_id = getattr(self.formula_tokenizer, "bos_token_id", None)

        if pad_id is None or eos_id is None or bos_id is None:
            print("[REINFORCETrainer] RL: tokenizer does not expose {pad/eos/bos}_token_id correctly -> returning None")
            return None, None

        generate_kwargs: dict[str, object] = {
            "encoder_hidden_states": encoder_hidden_states,
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

        gen_model = model.module if hasattr(model, 'module') else model

        try:
            generation = gen_model.generate(**generate_kwargs)
        except Exception as e:
            print("[REINFORCETrainer] RL: model.generate failed:", repr(e))
            return None, None

        sequences = getattr(generation, "sequences", None)
        scores = getattr(generation, "scores", None)
        if sequences is None or scores is None or len(scores) == 0:
            print("[REINFORCETrainer] RL: empty sequences/scores -> returning None")
            return None, None

        if isinstance(scores, tuple):
            scores = list(scores)

        total_steps = len(scores)
        seq_len = sequences.size(-1)
        prefix_len = max(0, seq_len - total_steps)
        generated_tokens = sequences[:, prefix_len:].long()
        if generated_tokens.size(-1) > total_steps:
            generated_tokens = generated_tokens[:, :total_steps]

        score_tensor = torch.stack(scores, dim=0).transpose(0, 1)  # (B, T, V)
        score_log_probs = torch.log_softmax(score_tensor, dim=-1)
        score_probs = torch.exp(score_log_probs)
        token_entropy = -(score_probs * score_log_probs).sum(dim=-1)

        token_log_probs, token_hidden = self._recompute_log_probs_and_hidden(
            model=model,
            sequences=sequences,
            generated_tokens=generated_tokens,
            encoder_hidden_states=encoder_hidden_states,
            prefix_len=prefix_len,
            pad_id=pad_id,
            require_grad=require_grad,
        )

        token_mask = (generated_tokens != pad_id)
        token_mask_f = token_mask.to(dtype=token_log_probs.dtype)
        token_count_per_sample = token_mask_f.sum(dim=1).detach()
        token_entropy_sum = ((token_entropy * token_mask_f).sum(dim=1)).detach()
        train_action_log_prob_sum = ((score_log_probs.amax(dim=-1)) * token_mask_f).sum(dim=1).detach()

        generated_tokens_cpu = generated_tokens.detach().cpu()
        try:
            generated_strings = self.formula_tokenizer.batch_decode(
                generated_tokens_cpu, skip_special_tokens=True
            )
        except Exception as e:
            print("[REINFORCETrainer] RL: batch_decode failed:", repr(e))
            return None, None

        reward_tensor = torch.zeros(len(generated_strings), dtype=token_log_probs.dtype, device=device)
        valid_mask = torch.zeros(len(generated_strings), dtype=torch.bool, device=device)

        with torch.no_grad():
            for i, generated_str in enumerate(generated_strings):
                try:
                    generated_formula = str_to_formula(generated_str)
                    generated_sats = self.kernel._evaluate_formula_on_traces(generated_formula, self.semantic_eval_batch_size)
                    target_sats = target_satisfactions[i]
                    if generated_sats.numel() != target_sats.numel():
                        raise ValueError("Satisfaction length mismatch")

                    hamming = torch.logical_xor(generated_sats, target_sats).to(torch.float32).mean()
                    reward = 1.0 - hamming
                    valid_mask[i] = True
                    if self.reward_clip is not None:
                        reward = torch.clamp(reward, min=-self.reward_clip, max=self.reward_clip)
                    reward_tensor[i]=reward
                except Exception:
                    continue

        if not bool(valid_mask.any()):
            zeros_per_sample = torch.zeros_like(reward_tensor)
            zero_loss = reward_tensor.new_zeros(())
            self._last_rl_metrics = {
                "token_count_per_sample": token_count_per_sample,
                "token_entropy_sum": token_entropy_sum,
                "train_action_log_prob_sum": train_action_log_prob_sum,
                "valid_formula_mask_per_sample": valid_mask.detach(),
                "reward_per_sample": reward_tensor.detach(),
                "train_actor_loss": zero_loss,
                "train_critic_loss": zero_loss,
                "advantage_per_sample": zeros_per_sample,
                "value_sum_per_sample": zeros_per_sample,
                "returns_sum": zeros_per_sample,
                "returns_sq_sum": zeros_per_sample,
                "value_err_sq_sum": zeros_per_sample,
                "value_err_sum": zeros_per_sample,
            }
            return None, valid_mask

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        reward_valid = reward_tensor[valid_idx]
        token_log_probs = token_log_probs[valid_idx]
        token_hidden = token_hidden[valid_idx].detach()
        token_mask = token_mask[valid_idx]
        token_mask_f = token_mask_f[valid_idx]
        lengths_valid = token_mask.sum(dim=-1).clamp(min=1)
        rewards = torch.zeros_like(token_log_probs)
        rewards.scatter_(1, (lengths_valid - 1).unsqueeze(-1), reward_valid.unsqueeze(-1))

        # Collect (hidden, rewards, mask) during the first epoch of critic pretraining.
        # token_hidden is already detached; rewards has no grad (derived from no_grad reward_tensor).
        if self._is_critic_pretraining_active() and not self._critic_cache_full:
            self._critic_cache.append((
                token_hidden.cpu(),
                rewards.detach().cpu(),
                token_mask_f.detach().cpu(),
            ))

        Bv, Tv, _ = token_hidden.shape
        values = self.critic(token_hidden).squeeze(-1)

        values_det = values.detach()
        next_values = torch.zeros_like(values_det)
        next_values[:, :-1] = values_det[:, 1:]

        next_mask = torch.zeros_like(token_mask_f)
        next_mask[:, :-1] = token_mask_f[:, 1:]

        deltas = rewards + self.gae_gamma * next_values * next_mask - values_det
        advantages = torch.zeros_like(deltas)
        gae_acc = torch.zeros(Bv, dtype=deltas.dtype, device=deltas.device)
        for t in range(Tv - 1, -1, -1):
            gae_acc = deltas[:, t] + self.gae_gamma * self.gae_lambda * next_mask[:, t] * gae_acc
            advantages[:, t] = gae_acc

        advantages = advantages * token_mask_f
        returns = (advantages + values_det) * token_mask_f

        pos_counts = token_mask_f.sum(dim=0)
        pos_counts_safe = pos_counts.clamp(min=1.0)
        advantage_pos_mean = (advantages.sum(dim=0) / pos_counts_safe).detach()
        value_pos_mean = ((values_det * token_mask_f).sum(dim=0) / pos_counts_safe).detach()
        returns_pos_mean = (returns.sum(dim=0) / pos_counts_safe).detach()

        denom = token_mask_f.sum().clamp(min=1.0)
        actor_loss = -((advantages.detach() * token_log_probs) * token_mask_f).sum() / denom
        critic_loss = torch.nn.functional.mse_loss(values * token_mask_f, returns, reduction="sum") / denom
        if self._is_critic_pretraining_active():
            reinforce_loss = critic_loss
        else:
            reinforce_loss = actor_loss + critic_loss

        advantage_per_sample = torch.zeros_like(reward_tensor)
        advantage_per_sample[valid_idx] = (advantages * token_mask_f).sum(dim=1).detach()

        value_sum_per_sample = torch.zeros_like(reward_tensor)
        value_sum_per_sample[valid_idx] = (values * token_mask_f).sum(dim=1).detach()

        returns_masked = returns * token_mask_f
        returns_sum = torch.zeros_like(reward_tensor)
        returns_sum[valid_idx] = returns_masked.sum(dim=1).detach()

        returns_sq_sum = torch.zeros_like(reward_tensor)
        returns_sq_sum[valid_idx] = (returns_masked * returns_masked).sum(dim=1).detach()

        value_err = (returns - values)
        value_err_masked = value_err * token_mask_f
        value_err_sq_sum = torch.zeros_like(reward_tensor)
        value_err_sq_sum[valid_idx] = (value_err_masked * value_err_masked).sum(dim=1).detach()

        value_err_sum = torch.zeros_like(reward_tensor)
        value_err_sum[valid_idx] = value_err_masked.sum(dim=1).detach()

        self._last_rl_metrics = {
            "token_count_per_sample": token_count_per_sample,
            "token_entropy_sum": token_entropy_sum,
            "train_action_log_prob_sum": train_action_log_prob_sum,
            "valid_formula_mask_per_sample": valid_mask.detach(),
            "reward_per_sample": reward_tensor.detach(),
            "train_actor_loss": actor_loss.detach(),
            "train_critic_loss": critic_loss.detach(),
            "advantage_per_sample": advantage_per_sample,
            "value_sum_per_sample": value_sum_per_sample,
            "returns_sum": returns_sum,
            "returns_sq_sum": returns_sq_sum,
            "value_err_sq_sum": value_err_sq_sum,
            "value_err_sum": value_err_sum,
            "advantage_pos_mean": advantage_pos_mean,
            "value_pos_mean": value_pos_mean,
            "returns_pos_mean": returns_pos_mean,
            "pos_counts": pos_counts.detach(),
            }

        return reinforce_loss, valid_mask



    def _critic_loss_from_cache(
        self,
        token_hidden: torch.Tensor,
        rewards: torch.Tensor,
        token_mask_f: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        Bv, Tv, _ = token_hidden.shape
        values = self.critic(token_hidden).squeeze(-1)
        values_det = values.detach()

        next_values = torch.zeros_like(values_det)
        next_values[:, :-1] = values_det[:, 1:]
        next_mask = torch.zeros_like(token_mask_f)
        next_mask[:, :-1] = token_mask_f[:, 1:]

        deltas = rewards + self.gae_gamma * next_values * next_mask - values_det
        advantages = torch.zeros_like(deltas)
        gae_acc = torch.zeros(Bv, dtype=deltas.dtype, device=deltas.device)
        for t in range(Tv - 1, -1, -1):
            gae_acc = deltas[:, t] + self.gae_gamma * self.gae_lambda * next_mask[:, t] * gae_acc
            advantages[:, t] = gae_acc

        returns = (advantages + values_det) * token_mask_f
        denom = token_mask_f.sum().clamp(min=1.0)
        loss = torch.nn.functional.mse_loss(
            values * token_mask_f, returns, reduction="sum"
        ) / denom

        value_err_masked = (returns - values) * token_mask_f
        rl_metrics = {
            "token_count_per_sample": token_mask_f.sum(dim=1).detach(),
            "valid_formula_mask_per_sample": torch.ones(Bv, dtype=torch.bool, device=token_mask_f.device),
            "reward_per_sample": rewards.sum(dim=1).detach(),
            "advantage_per_sample": (advantages * token_mask_f).sum(dim=1).detach(),
            "value_sum_per_sample": (values_det * token_mask_f).sum(dim=1).detach(),
            "returns_sum": returns.sum(dim=1).detach(),
            "returns_sq_sum": (returns * returns).sum(dim=1).detach(),
            "value_err_sum": value_err_masked.sum(dim=1).detach(),
            "value_err_sq_sum": (value_err_masked * value_err_masked).sum(dim=1).detach(),
        }
        return loss, rl_metrics


    def _slice_inputs_by_mask(self, inputs: dict, mask: torch.Tensor) -> dict:
        sliced: dict = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == mask.size(0):
                sliced[key] = value[mask.to(device=value.device)]
            else:
                sliced[key] = value
        return sliced


    def _recompute_log_probs_and_hidden(
        self,
        *,
        model,
        sequences: torch.Tensor,
        generated_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prefix_len: int,
        pad_id: int | None,
        require_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = generated_tokens.size(-1)
        teacher_inputs = sequences[:, : prefix_len + target_len].detach()
        shifted_inputs = teacher_inputs[:, :-1]
        if pad_id is None:
            shifted_attention_mask = torch.ones_like(shifted_inputs, dtype=torch.long)
        else:
            shifted_attention_mask = (shifted_inputs != pad_id).to(dtype=torch.long)

        grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with grad_ctx:
            outputs = model(
                input_ids=shifted_inputs,
                attention_mask=shifted_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -target_len:, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            gather_tokens = generated_tokens.unsqueeze(-1)
            tf_log_probs = log_probs.gather(dim=-1, index=gather_tokens).squeeze(-1)
            hidden = outputs.hidden_states[-1][:, -target_len:, :]

        if pad_id is not None:
            mask = (generated_tokens != pad_id).to(tf_log_probs.dtype)
            tf_log_probs = tf_log_probs * mask
            hidden = hidden * mask.unsqueeze(-1)

        return tf_log_probs, hidden

    

    # ------------------------------- PERSISTENCE METHODS -------------------------------
    CRITIC_STATE_FILE = "critic_state.pt"
    CRITIC_OPT_STATE_FILE = "critic_optimizer_state.pt"

    def save_trainer_state(self, output_dir: str | None = None, save_optimizer: bool = False) -> str:
        target_dir = output_dir or self.args.output_dir
        os.makedirs(target_dir, exist_ok=True)

        critic_path = os.path.join(target_dir, self.CRITIC_STATE_FILE)
        payload = {
            "critic_state_dict": self.critic.state_dict(),
            "critic_lr": self.critic_lr,
            "critic_weight_decay": self.critic_weight_decay,
            "gae_gamma": self.gae_gamma,
            "gae_lambda": self.gae_lambda,
        }
        torch.save(payload, critic_path)

        if save_optimizer and getattr(self, "optimizer", None) is not None:
            opt_path = os.path.join(target_dir, self.CRITIC_OPT_STATE_FILE)
            torch.save(self.optimizer.state_dict(), opt_path)

        return critic_path



    def load_trainer_state(
        self,
        load_dir: str,
        load_optimizer: bool = False,
        strict_critic: bool = True,
    ) -> bool:
        critic_path = os.path.join(load_dir, self.CRITIC_STATE_FILE)
        if not os.path.exists(critic_path):
            print(f"[REINFORCETrainer] Critic state not found at {critic_path}, skipping load.")
            return False

        ckpt = torch.load(critic_path, map_location="cpu")
        state_dict = ckpt.get("critic_state_dict", ckpt)

        incompatible = self.critic.load_state_dict(state_dict, strict=False)
        if strict_critic and (len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0):
            raise RuntimeError(
                "Critic state incompatible: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        self.critic.to(getattr(self.args, "device", None))

        if load_optimizer and getattr(self, "optimizer", None) is not None:
            opt_path = os.path.join(load_dir, self.CRITIC_OPT_STATE_FILE)
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu")
                self.optimizer.load_state_dict(opt_state)
            else:
                print(f"[REINFORCETrainer] Optimizer state not found at {opt_path}, skipping load.")

        self._attach_critic_to_model()

        return True



    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        self.save_trainer_state(output_dir=output_dir, save_optimizer=False)



    # ------------------------------- MISC METHODS -------------------------------
    def _sync_kernel_critic_device(self, device: torch.device | str | None) -> None:
        if device is None:
            return
        if self.kernel is None:
            return
        self.kernel.set_device(device)
        self.critic.to(device)

    def _sync_critic_gradients(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not accelerator.sync_gradients:
            return

        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        for param in self.critic.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

        

    def _attach_critic_to_model(self) -> None:
        if self.model is None or self.critic is None:
            return

        target_model = self.model.module if hasattr(self.model, "module") else self.model
        existing = getattr(target_model, "critic", None)
        if existing is self.critic:
            return

        target_model.critic = self.critic



    def create_optimizer(self):
        optimizer = super().create_optimizer()

        critic_param_ids = {id(p) for p in self.critic.parameters() if p.requires_grad}
        if not critic_param_ids:
            return optimizer

        # Strip critic params from whichever base groups they landed in.
        # super().create_optimizer() already included them (via model.critic)
        # at the actor learning rate, so we need to move them.
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"]
                            if id(p) not in critic_param_ids]

        # Re-add under a dedicated group at the critic learning rate
        optimizer.add_param_group({
            "params": [p for p in self.critic.parameters() if p.requires_grad],
            "lr": self.critic_lr,
            "weight_decay": self.critic_weight_decay,
        })

        return optimizer


    def create_scheduler(self, num_training_steps: int, optimizer=None):
        from torch.optim.lr_scheduler import LambdaLR

        optimizer = optimizer or self.optimizer
        pretraining_steps = self._resolve_critic_pretraining_steps()
        actor_warmup_steps = self.args.get_warmup_steps(num_training_steps)

        def actor_lambda(step: int) -> float:
            # Warmup begins only after critic pretraining ends.
            effective = max(0, step - pretraining_steps)
            if actor_warmup_steps == 0:
                return 1.0
            return min(1.0, effective / actor_warmup_steps)

        def critic_lambda(step: int) -> float:
            return 1.0  # Critic uses full LR from step 0, no warmup.

        # Critic group is always last (added last in create_optimizer).
        num_groups = len(optimizer.param_groups)
        lambdas = [actor_lambda] * (num_groups - 1) + [critic_lambda]

        self.lr_scheduler = LambdaLR(optimizer, lambdas)
        return self.lr_scheduler

