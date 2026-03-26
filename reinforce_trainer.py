import os
import torch
import torch.nn as nn
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
            satisfactions_path: str | None = None,
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
            self.satisfactions_path = satisfactions_path
            self._reward_baseline: torch.Tensor | None = None
            self._reward_sq_mean: torch.Tensor | None = None
            self._satisfactions_mmap: torch.Tensor | None = None
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
        # ----------- REINFORCE loss -----------
        semantic_embeddings = inputs.get("semantic_embeddings")
        attention_mask = inputs.get("attention_mask")
        batch_formula_ids = inputs.get("formula_ids")
        generation_max_length: int | None = None
        if semantic_embeddings is not None:
            if attention_mask is not None:
                generation_max_length = int(attention_mask.size(-1))
            elif hasattr(model, "config") and getattr(model.config, "n_positions", None):
                generation_max_length = int(getattr(model.config, "n_positions"))
            else:
                generation_max_length = 512

        reinforce_loss = None
        valid_mask = None
        if (
            semantic_embeddings is not None
            and generation_max_length is not None
        ):
            reinforce_loss, valid_mask = self._compute_reinforce_term(
                model=model,
                semantic_embeddings=semantic_embeddings,
                batch_formula_ids=batch_formula_ids,
                generation_max_length=generation_max_length,
                require_grad=model.training,
            )

        # ----------- CE loss (fallback) -----------
        if reinforce_loss is None or valid_mask is None:
            outputs = model(**inputs)
            ce_loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]
            zero = ce_loss.detach().new_zeros(())
            one = ce_loss.detach().new_ones(())
            self._last_train_metrics = {
                "train_loss": ce_loss.detach(),
                "train_valid_ratio": zero,
                "train_invalid_ratio": one,
                "train_rl_loss": zero,
                "train_ce_loss": ce_loss.detach(),
            }
            if return_outputs:
                return ce_loss, outputs
            return ce_loss

        # ----------- Combine losses -----------
        valid_mask = valid_mask.to(device=reinforce_loss.device)
        invalid_mask = ~valid_mask
        loss_terms: list[torch.Tensor] = []
        valid_ratio = valid_mask.to(dtype=reinforce_loss.dtype).mean()
        invalid_ratio = reinforce_loss.detach().new_ones(()) - valid_ratio.detach()
        ce_invalid_metric = reinforce_loss.detach().new_zeros(())

        loss_terms.append(valid_ratio * reinforce_loss)

        if bool(invalid_mask.any()):
            invalid_inputs = self._slice_inputs_by_mask(inputs, invalid_mask)
            invalid_outputs = model(**invalid_inputs)
            ce_invalid_loss = (
                invalid_outputs.loss
                if hasattr(invalid_outputs, "loss") and invalid_outputs.loss is not None
                else invalid_outputs[0]
            )
            ce_invalid_metric = ce_invalid_loss.detach()
            loss_terms.append(invalid_ratio * ce_invalid_loss)

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
        semantic_embeddings: torch.Tensor,
        generation_max_length: int,
        batch_formula_ids: torch.Tensor | None = None,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if semantic_embeddings is None or semantic_embeddings.ndim < 2:
            print("[REINFORCETrainer] RL: semantic_embeddings invalid -> returning None")
            return None, None

        if batch_formula_ids is None:
            print("[REINFORCETrainer] RL: formula_ids missing -> returning None")
            return None, None

        device = self.args.device
        target_satisfactions = self._get_satisfactions_rows(batch_formula_ids)
        if target_satisfactions is None:
            print("[REINFORCETrainer] RL: failed to load target satisfactions -> returning None")
            return None, None
        target_satisfactions = target_satisfactions.to(device=device, non_blocking=True)

        generation_max_length = max(1, int(generation_max_length))
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
                semantic_embeddings=semantic_embeddings,
                prefix_len=prefix_len,
                pad_id=pad_id,
            )
        else:
            token_log_probs = score_log_probs.gather(
                dim=-1, index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)

        token_mask = (generated_tokens != pad_id)
        token_mask_f = token_mask.to(dtype=token_log_probs.dtype)

        generated_tokens_cpu = generated_tokens.detach().cpu()
        try:
            generated_strings = self.formula_tokenizer.batch_decode(
                generated_tokens_cpu, skip_special_tokens=True
            )
        except Exception as e:
            print("[REINFORCETrainer] RL: batch_decode failed:", repr(e))
            return None, None

        reward_values: list[torch.Tensor] = []
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
                    reward_values.append(reward)
                except Exception:
                    continue

        if not bool(valid_mask.any()):
            self._last_rl_metrics = {}
            return None, valid_mask

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        token_log_probs = token_log_probs[valid_idx]
        token_entropy = token_entropy[valid_idx]
        token_mask = token_mask[valid_idx]
        token_mask_f = token_mask_f[valid_idx]
        reward_tensor = torch.stack(reward_values).to(device=device, dtype=token_log_probs.dtype)
        lengths_valid = token_mask.sum(dim=-1).clamp(min=1)
        seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths_valid
        reward_mean = reward_tensor.mean().detach()
        reward_sq_mean = (reward_tensor ** 2).mean().detach()

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
        advantage = ((reward_tensor - baseline) / std).detach()

        reinforce_loss = -(advantage * seq_log_prob).mean() 

        action_logprob_mean = seq_log_prob.mean().detach()
        reward_var = reward_tensor.var(unbiased=False).detach()
        advantage_mean = advantage.mean().detach()
        advantage_var = advantage.var(unbiased=False).detach()
        policy_entropy = ((token_entropy * token_mask_f).sum() / token_mask_f.sum().clamp(min=1.0)).detach()

        self._last_rl_metrics = {
            "train_action_logprob_mean": action_logprob_mean,
            "train_reward_mean": reward_mean,
            "train_reward_variance": reward_var,
            "train_advantage_mean": advantage_mean,
            "train_advantage_variance": advantage_var,
            "train_policy_entropy": policy_entropy,
        }
    
        
        return reinforce_loss, valid_mask



    def _slice_inputs_by_mask(self, inputs: dict, mask: torch.Tensor) -> dict:
        sliced: dict = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == mask.size(0):
                sliced[key] = value[mask.to(device=value.device)]
            else:
                sliced[key] = value
        return sliced
    



    def _get_satisfactions_rows(self, formula_ids: torch.Tensor) -> torch.Tensor | None:
        if self.satisfactions_path is None:
            return None

        if self._satisfactions_mmap is None:
            self._satisfactions_mmap = torch.load(self.satisfactions_path, map_location="cpu", mmap=True)

        if formula_ids.ndim == 0:
            formula_ids = formula_ids.unsqueeze(0)
        if formula_ids.ndim != 1:
            return None

        formula_ids_cpu = formula_ids.detach().to(dtype=torch.long, device="cpu")
        return self._satisfactions_mmap.index_select(0, formula_ids_cpu)
    

    
    def _recompute_log_probs_with_grad(
        self,
        *,
        model,
        sequences: torch.Tensor,
        generated_tokens: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        prefix_len: int,
        pad_id: int | None,
    ) -> torch.Tensor:
        target_len = generated_tokens.size(-1)
        teacher_inputs = sequences[:, : prefix_len + target_len].detach()
        shifted_inputs = teacher_inputs[:, :-1]

        with torch.enable_grad():
            outputs = model(
                input_ids=shifted_inputs,
                semantic_embeddings=semantic_embeddings,
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
            satisfactions_path: str | None = None,
            gae_gamma: float = 0.99,
            gae_lambda: float = 0.95,
            critic_loss_coef: float = 0.5,
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
            self.satisfactions_path = satisfactions_path
            self.gae_gamma = float(gae_gamma)
            self.gae_lambda = float(gae_lambda)
            self.critic_loss_coef = float(critic_loss_coef)
            self.critic_lr = float(critic_lr) if critic_lr is not None else float(self.args.learning_rate)
            self.critic_weight_decay = float(critic_weight_decay)
            self._satisfactions_mmap: torch.Tensor | None = None
            self._last_train_metrics: dict[str, float | torch.Tensor] = {}
            self._last_rl_metrics: dict[str, float | torch.Tensor] = {}

            hidden_dim = int(getattr(self.model.config, "n_embd", 0))
            if hidden_dim <= 0:
                raise ValueError("Model config must expose positive n_embd for critic construction.")

            semantic_dim = int(getattr(kernel, "m", 0))
            if semantic_dim <= 0:
                raise ValueError("Kernel does not have the required m attribute to perform training. Make sure the kernel is initialized properly with a positive semantic embedding dimension.")

            self.critic = nn.Sequential(
                nn.Linear(hidden_dim + semantic_dim, critic_hidden_dim),
                nn.Tanh(),
                nn.Linear(critic_hidden_dim, 1),
            )

            self._sync_kernel_critic_device(getattr(self.args, "device", None))



    # ------------------------------- CORE LOSS METHODS -------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        # ----------- REINFORCE loss -----------
        semantic_embeddings = inputs.get("semantic_embeddings")
        attention_mask = inputs.get("attention_mask")
        batch_formula_ids = inputs.get("formula_ids")
        generation_max_length: int | None = None
        if semantic_embeddings is not None:
            if attention_mask is not None:
                generation_max_length = int(attention_mask.size(-1))
            elif hasattr(model, "config") and getattr(model.config, "n_positions", None):
                generation_max_length = int(getattr(model.config, "n_positions"))
            else:
                generation_max_length = 512

        reinforce_loss = None
        valid_mask = None
        if (
            semantic_embeddings is not None
            and generation_max_length is not None
        ):
            reinforce_loss, valid_mask = self._compute_reinforce_term(
                model=model,
                semantic_embeddings=semantic_embeddings,
                batch_formula_ids=batch_formula_ids,
                generation_max_length=generation_max_length,
                require_grad=model.training,
            )

        # ----------- CE loss (fallback) -----------
        if reinforce_loss is None or valid_mask is None:
            outputs = model(**inputs)
            ce_loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]
            zero = ce_loss.detach().new_zeros(())
            one = ce_loss.detach().new_ones(())
            self._last_train_metrics = {
                "train_loss": ce_loss.detach(),
                "train_valid_ratio": zero,
                "train_invalid_ratio": one,
                "train_rl_loss": zero,
                "train_ce_loss": ce_loss.detach(),
            }
            if return_outputs:
                return ce_loss, outputs
            return ce_loss

        # ----------- Combine losses -----------
        valid_mask = valid_mask.to(device=reinforce_loss.device)
        invalid_mask = ~valid_mask
        loss_terms: list[torch.Tensor] = []
        valid_ratio = valid_mask.to(dtype=reinforce_loss.dtype).mean()
        invalid_ratio = reinforce_loss.detach().new_ones(()) - valid_ratio.detach()
        ce_invalid_metric = reinforce_loss.detach().new_zeros(())

        loss_terms.append(valid_ratio * reinforce_loss)

        if bool(invalid_mask.any()):
            invalid_inputs = self._slice_inputs_by_mask(inputs, invalid_mask)
            invalid_outputs = model(**invalid_inputs)
            ce_invalid_loss = (
                invalid_outputs.loss
                if hasattr(invalid_outputs, "loss") and invalid_outputs.loss is not None
                else invalid_outputs[0]
            )
            ce_invalid_metric = ce_invalid_loss.detach()
            loss_terms.append(invalid_ratio * ce_invalid_loss)

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
        semantic_embeddings: torch.Tensor,
        generation_max_length: int,
        batch_formula_ids: torch.Tensor | None = None,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if semantic_embeddings is None or semantic_embeddings.ndim < 2:
            print("[REINFORCETrainer] RL: semantic_embeddings invalid -> returning None")
            return None, None

        if batch_formula_ids is None:
            print("[REINFORCETrainer] RL: formula_ids missing -> returning None")
            return None, None

        device = self.args.device
        target_satisfactions = self._get_satisfactions_rows(batch_formula_ids)
        if target_satisfactions is None:
            print("[REINFORCETrainer] RL: failed to load target satisfactions -> returning None")
            return None, None
        target_satisfactions = target_satisfactions.to(device=device, non_blocking=True)

        generation_max_length = max(1, int(generation_max_length))
        pad_id = getattr(self.formula_tokenizer, "pad_token_id", None)
        eos_id = getattr(self.formula_tokenizer, "eos_token_id", None)
        bos_id = getattr(self.formula_tokenizer, "bos_token_id", None)

        if pad_id is None or eos_id is None or bos_id is None:
            print("[REINFORCETrainer] RL: tokenizer does not expose {pad/eos/bos}_token_id correctly -> returning None")
            return None, None

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
            semantic_embeddings=semantic_embeddings,
            prefix_len=prefix_len,
            pad_id=pad_id,
            require_grad=require_grad,
        )

        token_mask = (generated_tokens != pad_id)
        token_mask_f = token_mask.to(dtype=token_log_probs.dtype)

        generated_tokens_cpu = generated_tokens.detach().cpu()
        try:
            generated_strings = self.formula_tokenizer.batch_decode(
                generated_tokens_cpu, skip_special_tokens=True
            )
        except Exception as e:
            print("[REINFORCETrainer] RL: batch_decode failed:", repr(e))
            return None, None

        reward_values: list[torch.Tensor] = []
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
                    reward_values.append(reward)
                except Exception:
                    continue

        if not bool(valid_mask.any()):
            self._last_rl_metrics = {}
            return None, valid_mask

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        token_log_probs = token_log_probs[valid_idx]
        token_entropy = token_entropy[valid_idx]
        token_hidden = token_hidden[valid_idx].detach()
        token_mask = token_mask[valid_idx]
        token_mask_f = token_mask_f[valid_idx]
        semantic_valid = semantic_embeddings[valid_idx]
        reward_tensor = torch.stack(reward_values).to(device=device, dtype=token_log_probs.dtype)
        lengths_valid = token_mask.sum(dim=-1).clamp(min=1)
        seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths_valid
        rewards = torch.zeros_like(token_log_probs)
        rewards.scatter_(1, (lengths_valid - 1).unsqueeze(-1), reward_tensor.unsqueeze(-1))

        Bv, Tv, _ = token_hidden.shape
        sem_expanded = semantic_valid.unsqueeze(1).expand(Bv, Tv, semantic_valid.size(-1))
        critic_in = torch.cat([token_hidden, sem_expanded], dim=-1)
        values = self.critic(critic_in).squeeze(-1)

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
        reinforce_loss = actor_loss + self.critic_loss_coef * critic_loss

        values_mean = (values * token_mask_f).sum() / denom
        adv_mean = (advantages * token_mask_f).sum() / denom
        adv_centered = (advantages - adv_mean) * token_mask_f
        adv_var = (adv_centered * adv_centered).sum() / denom

        returns_masked = returns * token_mask_f
        ret_mean = returns_masked.sum() / denom
        ret_centered = (returns_masked - ret_mean) * token_mask_f
        ret_var = (ret_centered * ret_centered).sum() / denom
        value_err = (returns - values) * token_mask_f
        value_err_var = (value_err * value_err).sum() / denom
        explained_var = 1.0 - (value_err_var / ret_var.clamp(min=1e-8))
        action_logprob_mean = seq_log_prob.mean().detach()
        reward_mean = reward_tensor.mean().detach()
        reward_var = reward_tensor.var(unbiased=False).detach()
        policy_entropy = ((token_entropy * token_mask_f).sum() / denom).detach()

        self._last_rl_metrics = {
            "train_action_logprob_mean": action_logprob_mean,
            "train_reward_mean": reward_mean,
            "train_reward_variance": reward_var,
            "train_advantage_mean": adv_mean.detach(),
            "train_advantage_variance": adv_var.detach(),
            "train_value_loss": critic_loss.detach(),
            "train_value_mean": values_mean.detach(),
            "train_value_explained_variance": explained_var.detach(),
            "train_policy_entropy": policy_entropy,
        }

        return reinforce_loss, valid_mask



    def _slice_inputs_by_mask(self, inputs: dict, mask: torch.Tensor) -> dict:
        sliced: dict = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == mask.size(0):
                sliced[key] = value[mask.to(device=value.device)]
            else:
                sliced[key] = value
        return sliced



    def _get_satisfactions_rows(self, formula_ids: torch.Tensor) -> torch.Tensor | None:
        if self.satisfactions_path is None:
            return None

        if self._satisfactions_mmap is None:
            self._satisfactions_mmap = torch.load(self.satisfactions_path, map_location="cpu", mmap=True)

        if formula_ids.ndim == 0:
            formula_ids = formula_ids.unsqueeze(0)
        if formula_ids.ndim != 1:
            return None

        formula_ids_cpu = formula_ids.detach().to(dtype=torch.long, device="cpu")
        return self._satisfactions_mmap.index_select(0, formula_ids_cpu)
    

    
    def _recompute_log_probs_and_hidden(
        self,
        *,
        model,
        sequences: torch.Tensor,
        generated_tokens: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        prefix_len: int,
        pad_id: int | None,
        require_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = generated_tokens.size(-1)
        teacher_inputs = sequences[:, : prefix_len + target_len].detach()
        shifted_inputs = teacher_inputs[:, :-1]

        grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with grad_ctx:
            outputs = model(
                input_ids=shifted_inputs,
                semantic_embeddings=semantic_embeddings,
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
            "critic_loss_coef": self.critic_loss_coef,
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