import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
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
            difficulty_sampling: bool = False,
            difficulty_start_target: float = 0.10,
            difficulty_temperature: float = 0.80,
            difficulty_step_size: float = 2.4,
            difficulty_update_alpha: float = 2.0,
            difficulty_performance_target: float = 0.80,
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
            self.difficulty_sampling = bool(difficulty_sampling)
            self.difficulty_start_target = float(difficulty_start_target)
            self.difficulty_temperature = float(difficulty_temperature)
            self.difficulty_step_size = float(difficulty_step_size)
            self.difficulty_update_alpha = float(difficulty_update_alpha)
            self.difficulty_performance_target = float(difficulty_performance_target)
            self._reward_baseline: torch.Tensor | None = None
            self._reward_sq_mean: torch.Tensor | None = None
            self._last_train_metrics: dict[str, float | torch.Tensor] = {}
            self._last_rl_metrics: dict[str, float | torch.Tensor] = {}
            self._difficulty_sampler: AdaptiveDifficultySampler | None = None
            self._uniform_sampler: UniformRandomSampler | None = None
            self._active_train_sampler: MutableDelegatingSampler | None = None
            self._active_sampler_kind: str = "uniform"
            self._curriculum_warmup_steps: int | None = None
            self._sync_kernel_device(getattr(self.args, "device", None))
            self._maybe_init_difficulty_sampler()
            self._maybe_init_uniform_sampler()


    def _maybe_init_difficulty_sampler(self) -> None:
        if not self.difficulty_sampling:
            return
        formulas = getattr(self.train_dataset, "formulas", None)
        if formulas is None or len(formulas) == 0:
            print("[REINFORCETrainer] Difficulty sampling disabled: formulas unavailable in train dataset.")
            self.difficulty_sampling = False
            return

        depths = torch.tensor([int(phi.depth()) for phi in formulas], dtype=torch.float32)
        min_depth = float(depths.min().item())
        max_depth = float(depths.max().item())
        difficulties = depths
        start_target = min_depth + self.difficulty_start_target * (max_depth - min_depth)
        start_target = float(min(max(start_target, min_depth), max_depth))

        num_samples = int(len(formulas))
        base_seed = int(getattr(self.args, "seed", 0) or 0)

        self._difficulty_sampler = AdaptiveDifficultySampler(
            difficulties=difficulties,
            num_samples=num_samples,
            temperature=self.difficulty_temperature,
            start_target_difficulty=start_target,
            max_difficulty_step=self.difficulty_step_size,
            update_alpha=self.difficulty_update_alpha,
            performance_target=self.difficulty_performance_target,
            seed=base_seed,
        )


    def _maybe_init_uniform_sampler(self) -> None:
        if not self.difficulty_sampling:
            return
        if self.train_dataset is None:
            return

        num_samples = int(len(self.train_dataset))
        base_seed = int(getattr(self.args, "seed", 0) or 0)

        self._uniform_sampler = UniformRandomSampler(
            dataset_size=num_samples,
            num_samples=num_samples,
            seed=base_seed,
        )


    def _resolve_curriculum_warmup_steps(self) -> int:
        if self._curriculum_warmup_steps is not None:
            return self._curriculum_warmup_steps

        configured_steps = getattr(self.args, "critic_warmup_steps", None)
        if configured_steps is not None:
            self._curriculum_warmup_steps = max(0, int(configured_steps))
            return self._curriculum_warmup_steps

        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps > 0:
            self._curriculum_warmup_steps = max(0, int(self.args.get_warmup_steps(max_steps)))
        else:
            self._curriculum_warmup_steps = max(0, int(getattr(self.args, "warmup_steps", 0) or 0))
        return self._curriculum_warmup_steps


    def _is_curriculum_warmup_active(self) -> bool:
        return int(getattr(self.state, "global_step", 0) or 0) < self._resolve_curriculum_warmup_steps()


    def _sync_active_train_sampler(self) -> None:
        if self._active_train_sampler is None or self._uniform_sampler is None:
            return

        use_uniform = self._is_curriculum_warmup_active() or self._difficulty_sampler is None
        target_sampler = self._uniform_sampler if use_uniform else self._difficulty_sampler
        target_kind = "uniform" if use_uniform else "difficulty"

        if target_sampler is not None and self._active_sampler_kind != target_kind:
            self._active_train_sampler.set_inner_sampler(target_sampler)
            self._active_sampler_kind = target_kind

    
    
    # ------------------------------- CORE LOSS METHODS -------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        if model.training:
            self._sync_active_train_sampler()
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


    def _get_train_sampler(self, train_dataset=None):
        if self._uniform_sampler is not None:
            initial_inner = self._uniform_sampler
            self._active_sampler_kind = "uniform"
            self._active_train_sampler = MutableDelegatingSampler(initial_inner)
            return self._active_train_sampler

        if self._difficulty_sampler is not None:
            initial_inner = self._difficulty_sampler
            self._active_sampler_kind = "difficulty"
            self._active_train_sampler = MutableDelegatingSampler(initial_inner)
            return self._active_train_sampler
        return super()._get_train_sampler(train_dataset)


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

        if not bool(valid_mask.any()):
            if require_grad:
                self._update_difficulty_sampler_from_rewards(reward_tensor)
            self._last_rl_metrics = {
                "token_count_per_sample": token_count_per_sample,
                "token_entropy_sum": token_entropy_sum,
                "train_action_log_prob_sum": train_action_log_prob_sum,
                "valid_formula_mask_per_sample": valid_mask.detach(),
                "reward_per_sample": reward_tensor.detach(),
                "advantage_per_sample": torch.zeros_like(reward_tensor),
            }
            if self._difficulty_sampler is not None:
                self._last_rl_metrics["curriculum_target_difficulty"] = float(self._difficulty_sampler.target_difficulty)
            return None, valid_mask

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        reward_valid = reward_tensor[valid_idx]
        token_log_probs = token_log_probs[valid_idx]
        token_mask = token_mask[valid_idx]
        token_mask_f = token_mask_f[valid_idx]
        lengths_valid = token_mask.sum(dim=-1).clamp(min=1)
        seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths_valid
        reward_mean = reward_valid.mean().detach()
        reward_sq_mean = (reward_valid ** 2).mean().detach()

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

        # Normalize advantage (reduces variance significantly)
        baseline = self._reward_baseline.to(device=device, dtype=reward_tensor.dtype)
        variance = (self._reward_sq_mean.to(device=device, dtype=reward_tensor.dtype) - baseline.square()).clamp(min=1e-8)
        std = variance.sqrt().clamp(min=1e-4)
        advantage_valid = ((reward_valid - baseline) / std).detach()
        advantage_per_sample = torch.zeros_like(reward_tensor)
        advantage_per_sample[valid_idx] = advantage_valid

        reinforce_loss = -(advantage_valid * seq_log_prob).mean()
        if require_grad:
            self._update_difficulty_sampler_from_rewards(reward_tensor)

        self._last_rl_metrics = {
            "token_count_per_sample": token_count_per_sample,
            "token_entropy_sum": token_entropy_sum,
            "train_action_log_prob_sum": train_action_log_prob_sum,
            "valid_formula_mask_per_sample": valid_mask.detach(),
            "reward_per_sample": reward_tensor.detach(),
            "advantage_per_sample": advantage_per_sample.detach(),
        }
        if self._difficulty_sampler is not None:
            self._last_rl_metrics["curriculum_target_difficulty"] = float(self._difficulty_sampler.target_difficulty)
    
        
        return reinforce_loss, valid_mask


    def _update_difficulty_sampler_from_rewards(self, reward_tensor: torch.Tensor) -> None:
        if self._difficulty_sampler is None:
            return
        if self._is_curriculum_warmup_active():
            return

        valid_rewards = reward_tensor.detach()
        local_sum = valid_rewards.sum()
        local_count = valid_rewards.new_tensor(float(valid_rewards.numel()))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        global_count = float(local_count.detach().to(dtype=torch.float32).cpu().item())
        if global_count <= 0.0:
            return

        global_sum = float(local_sum.detach().to(dtype=torch.float32).cpu().item())
        batch_performance = global_sum / global_count
        self._difficulty_sampler.update_from_performance(batch_performance)



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
            difficulty_sampling: bool = False,
            difficulty_start_target: float = 0.10,
            difficulty_temperature: float = 0.80,
            difficulty_step_size: float = 2.4,
            difficulty_update_alpha: float = 2.0,
            difficulty_performance_target: float = 0.80,
            gae_gamma: float = 1.0,
            gae_lambda: float = 1.0,
            critic_lr: float | None = None,
            critic_hidden_dim: int = 256,
            critic_weight_decay: float = 0.0,
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
            self.difficulty_sampling = bool(difficulty_sampling)
            self.difficulty_start_target = float(difficulty_start_target)
            self.difficulty_temperature = float(difficulty_temperature)
            self.difficulty_step_size = float(difficulty_step_size)
            self.difficulty_update_alpha = float(difficulty_update_alpha)
            self.difficulty_performance_target = float(difficulty_performance_target)
            self.gae_gamma = float(gae_gamma)
            self.gae_lambda = float(gae_lambda)
            self.critic_lr = float(critic_lr) if critic_lr is not None else float(self.args.learning_rate)
            self.critic_weight_decay = float(critic_weight_decay)
            self._last_train_metrics: dict[str, float | torch.Tensor] = {}
            self._last_rl_metrics: dict[str, float | torch.Tensor] = {}
            self._difficulty_sampler: AdaptiveDifficultySampler | None = None
            self._uniform_sampler: UniformRandomSampler | None = None
            self._active_train_sampler: MutableDelegatingSampler | None = None
            self._active_sampler_kind: str = "uniform"
            self._actor_frozen: bool = False
            self._critic_warmup_steps: int | None = None

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
            self._maybe_init_difficulty_sampler()
            self._maybe_init_uniform_sampler()


    def _maybe_init_difficulty_sampler(self) -> None:
        if not self.difficulty_sampling:
            return
        formulas = getattr(self.train_dataset, "formulas", None)
        if formulas is None or len(formulas) == 0:
            print("[REINFORCETrainer] Difficulty sampling disabled: formulas unavailable in train dataset.")
            self.difficulty_sampling = False
            return

        depths = torch.tensor([int(phi.depth()) for phi in formulas], dtype=torch.float32)
        min_depth = float(depths.min().item())
        max_depth = float(depths.max().item())
        difficulties = depths
        start_target = min_depth + self.difficulty_start_target * (max_depth - min_depth)
        start_target = float(min(max(start_target, min_depth), max_depth))

        num_samples = int(len(formulas))
        base_seed = int(getattr(self.args, "seed", 0) or 0)

        self._difficulty_sampler = AdaptiveDifficultySampler(
            difficulties=difficulties,
            num_samples=num_samples,
            temperature=self.difficulty_temperature,
            start_target_difficulty=start_target,
            max_difficulty_step=self.difficulty_step_size,
            update_alpha=self.difficulty_update_alpha,
            performance_target=self.difficulty_performance_target,
            seed=base_seed,
        )


    def _maybe_init_uniform_sampler(self) -> None:
        if not self.difficulty_sampling:
            return
        if self.train_dataset is None:
            return

        num_samples = int(len(self.train_dataset))
        base_seed = int(getattr(self.args, "seed", 0) or 0)

        self._uniform_sampler = UniformRandomSampler(
            dataset_size=num_samples,
            num_samples=num_samples,
            seed=base_seed,
        )


    def _resolve_critic_warmup_steps(self) -> int:
        if self._critic_warmup_steps is not None:
            return self._critic_warmup_steps

        configured_steps = getattr(self.args, "critic_warmup_steps", None)
        if configured_steps is not None:
            self._critic_warmup_steps = max(0, int(configured_steps))
            return self._critic_warmup_steps

        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps > 0:
            self._critic_warmup_steps = max(0, int(self.args.get_warmup_steps(max_steps)))
        else:
            self._critic_warmup_steps = max(0, int(getattr(self.args, "warmup_steps", 0) or 0))
        return self._critic_warmup_steps


    def _is_critic_warmup_active(self) -> bool:
        return int(getattr(self.state, "global_step", 0) or 0) < self._resolve_critic_warmup_steps()


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


    def _sync_active_train_sampler(self) -> None:
        if self._active_train_sampler is None or self._uniform_sampler is None:
            return

        use_uniform = self._is_critic_warmup_active() or self._difficulty_sampler is None
        target_sampler = self._uniform_sampler if use_uniform else self._difficulty_sampler
        target_kind = "uniform" if use_uniform else "difficulty"

        if target_sampler is not None and self._active_sampler_kind != target_kind:
            self._active_train_sampler.set_inner_sampler(target_sampler)
            self._active_sampler_kind = target_kind



    # ------------------------------- CORE LOSS METHODS -------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        critic_warmup = self._is_critic_warmup_active()
        actor_model = model.module if (critic_warmup and hasattr(model, "module")) else model
        if model.training:
            self._sync_active_train_sampler()
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
        if self._is_critic_warmup_active():
            self._sync_critic_gradients()
        return loss


    def _get_train_sampler(self, train_dataset=None):
        if self._uniform_sampler is not None:
            initial_inner = self._uniform_sampler
            self._active_sampler_kind = "uniform"
            self._active_train_sampler = MutableDelegatingSampler(initial_inner)
            return self._active_train_sampler

        if self._difficulty_sampler is not None:
            initial_inner = self._difficulty_sampler
            self._active_sampler_kind = "difficulty"
            self._active_train_sampler = MutableDelegatingSampler(initial_inner)
            return self._active_train_sampler
        return super()._get_train_sampler(train_dataset)
    


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
            if require_grad:
                self._update_difficulty_sampler_from_rewards(reward_tensor)
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
            if self._difficulty_sampler is not None:
                self._last_rl_metrics["curriculum_target_difficulty"] = float(self._difficulty_sampler.target_difficulty)
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

        denom = token_mask_f.sum().clamp(min=1.0)
        actor_loss = -((advantages.detach() * token_log_probs) * token_mask_f).sum() / denom
        critic_loss = torch.nn.functional.mse_loss(values * token_mask_f, returns, reduction="sum") / denom
        if self._is_critic_warmup_active():
            reinforce_loss = critic_loss
        else:
            reinforce_loss = actor_loss + critic_loss
        if require_grad:
            self._update_difficulty_sampler_from_rewards(reward_tensor)

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
            }
        if self._difficulty_sampler is not None:
            self._last_rl_metrics["curriculum_target_difficulty"] = float(self._difficulty_sampler.target_difficulty)

        return reinforce_loss, valid_mask


    def _update_difficulty_sampler_from_rewards(self, reward_tensor: torch.Tensor) -> None:
        if self._difficulty_sampler is None:
            return
        if self._is_critic_warmup_active():
            return

        valid_rewards = reward_tensor.detach()
        local_sum = valid_rewards.sum()
        local_count = valid_rewards.new_tensor(float(valid_rewards.numel()))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        global_count = float(local_count.detach().to(dtype=torch.float32).cpu().item())
        if global_count <= 0.0:
            return

        global_sum = float(local_sum.detach().to(dtype=torch.float32).cpu().item())
        batch_performance = global_sum / global_count
        self._difficulty_sampler.update_from_performance(batch_performance)



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
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        if not critic_params:
            return optimizer

        seen = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
        new_params = [p for p in critic_params if id(p) not in seen]
        if not new_params:
            return optimizer

        optimizer.add_param_group(
            {
                "params": new_params,
                "lr": self.critic_lr,
                "weight_decay": self.critic_weight_decay,
            }
        )
        return optimizer



class AdaptiveDifficultySampler(Sampler[int]):

    def __init__(
        self,
        *,
        difficulties: torch.Tensor,
        num_samples: int,
        temperature: float = 0.20,
        start_target_difficulty: float = 0.10,
        max_difficulty_step: float = 0.05,
        update_alpha: float = 1.0,
        performance_target: float = 0.80,
        min_weight: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if difficulties.ndim != 1:
            raise ValueError("difficulties must be a 1D tensor")
        if difficulties.numel() == 0:
            raise ValueError("difficulties cannot be empty")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.difficulties = difficulties.detach().to(dtype=torch.float32, device="cpu")
        self.d_min = 2
        self.d_max = 5
        self.num_samples = int(num_samples)
        self.temperature = max(1e-4, float(temperature))
        self.max_difficulty_step = float(max_difficulty_step)
        self.update_alpha = float(update_alpha)
        self.performance_target = float(performance_target)
        self.min_weight = max(0.0, float(min_weight))
        self.seed = int(seed)
        self.target_difficulty = float(min(max(start_target_difficulty, self.d_min), self.d_max))

        self._weights = torch.empty_like(self.difficulties)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._refresh_weights()

    def __len__(self) -> int:
        return self.num_samples

    def update_from_performance(self, batch_performance: float) -> float:
        perf = float(batch_performance)
        delta = self.max_difficulty_step * math.tanh(self.update_alpha * (perf - self.performance_target))
        self.target_difficulty = float(min(max(self.target_difficulty + delta, self.d_min), self.d_max))
        self._refresh_weights()
        return self.target_difficulty

    def _refresh_weights(self) -> None:
        variance = max(self.temperature * self.temperature, 1e-8)
        distance_sq = (self.difficulties - self.target_difficulty).square()
        weights = torch.exp(-0.5 * distance_sq / variance)
        if self.min_weight > 0.0:
            weights = weights + self.min_weight
        total = float(weights.sum().item())
        if not math.isfinite(total) or total <= 0.0:
            weights = torch.ones_like(self.difficulties) / float(self.difficulties.numel())
        else:
            weights = weights / total
        self._weights = weights

    def __iter__(self):
        for _ in range(self.num_samples):
            sampled: torch.Tensor = torch.multinomial(
                self._weights,
                num_samples=1,
                replacement=True,
                generator=self._generator,
            )
            yield int(sampled.item())


class UniformRandomSampler(Sampler[int]):

    def __init__(self, *, dataset_size: int, num_samples: int, seed: int = 0) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset_size = int(dataset_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        sampled = torch.randint(
            low=0,
            high=self.dataset_size,
            size=(self.num_samples,),
            generator=self._generator,
        )
        for idx in sampled.tolist():
            yield int(idx)


class MutableDelegatingSampler(Sampler[int]):

    def __init__(self, inner_sampler: Sampler[int]) -> None:
        self._inner_sampler = inner_sampler
        self._version = 0

    def set_inner_sampler(self, inner_sampler: Sampler[int]) -> None:
        self._inner_sampler = inner_sampler
        self._version += 1

    def __len__(self) -> int:
        return len(self._inner_sampler)

    def __iter__(self):
        return _MutableDelegatingSamplerIterator(self)


class _MutableDelegatingSamplerIterator:

    def __init__(self, wrapper: MutableDelegatingSampler) -> None:
        self._wrapper = wrapper
        self._seen = 0
        self._version = wrapper._version
        self._inner_iter = iter(wrapper._inner_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        if self._seen >= len(self._wrapper):
            raise StopIteration

        if self._version != self._wrapper._version:
            self._version = self._wrapper._version
            self._inner_iter = iter(self._wrapper._inner_sampler)

        try:
            idx = next(self._inner_iter)
        except StopIteration:
            self._inner_iter = iter(self._wrapper._inner_sampler)
            idx = next(self._inner_iter)

        self._seen += 1
        return idx