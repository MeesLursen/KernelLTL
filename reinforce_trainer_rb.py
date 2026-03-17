import torch
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
            eval_batch_size: int = 10240,
            satisfactions_path: str | None = None,
            **kwargs,
        ) -> None:
            self.processing_class = kwargs.pop("processing_class", None)
            if "tokenizer" not in kwargs:
                kwargs["tokenizer"] = tokenizer
            super().__init__(*args, **kwargs)
            self.kernel = kernel
            self.formula_tokenizer = tokenizer
            self.baseline_momentum = baseline_momentum
            self.reward_clip = reward_clip
            self.eval_batch_size = eval_batch_size
            self.satisfactions_path = satisfactions_path
            self._reward_baseline: float | None = None
            self._satisfactions_mmap: torch.Tensor | None = None
            self._sync_kernel_device(getattr(self.args, "device", None))


    def _sync_kernel_device(self, device: torch.device | str | None) -> None:
        if device is None:
            return
        if self.kernel is None:
            return
        self.kernel.set_device(device)

    
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
        return_outputs: bool = False,
    ):
        outputs = model(**inputs)

        # ----------- CE loss -----------
        ce_loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]

        # ----------- REINFORCE loss -----------
        semantic_embeddings = inputs.get("semantic_embeddings")
        # Safety sync
        self._sync_kernel_device(
            semantic_embeddings.device if semantic_embeddings is not None else ce_loss.device
        )
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

        # ----------- Combine losses -----------
        if (
            reinforce_loss is None
            or valid_mask is None
        ):
            loss = ce_loss if reinforce_loss is None else reinforce_loss
        else:
            valid_mask = valid_mask.to(device=ce_loss.device)
            invalid_mask = ~valid_mask
            loss_terms: list[torch.Tensor] = []
            batch_size = max(int(valid_mask.numel()), 1)
            n_valid = int(valid_mask.sum().item())
            n_invalid = int(invalid_mask.sum().item())

            if reinforce_loss is not None and bool(valid_mask.any()):
                rl_weight = n_valid / batch_size
                loss_terms.append(rl_weight * reinforce_loss)

            if bool(invalid_mask.any()):
                ce_invalid = self._compute_ce_invalid_global(
                    logits=getattr(outputs, "logits", None),
                    labels=inputs.get("labels"),
                    valid_mask=valid_mask,
                )
                if ce_invalid is not None:
                    ce_weight = n_invalid / batch_size
                    loss_terms.append(ce_weight * ce_invalid)

            if len(loss_terms) == 0:
                loss = ce_loss
            else:
                loss = sum(loss_terms)

        # ----------- Logging -----------
        logging_steps = getattr(self.args, "logging_steps", None)
        step = getattr(self.state, "global_step", None)
        if (
            logging_steps is not None
            and logging_steps > 0
            and step is not None
            and step % logging_steps == 0
        ):
            raise NotImplementedError

        if return_outputs:
            return loss, outputs
        return loss



    def _compute_ce_invalid_global(
        self,
        *,
        logits: torch.Tensor | None,
        labels: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if logits is None or labels is None:
            return None
        if logits.ndim != 3:
            return None

        labels = labels.to(logits.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.ndim != 2:
            return None

        if valid_mask.ndim != 1 or valid_mask.size(0) != labels.size(0):
            return None
        valid_mask = valid_mask.to(labels.device)
        if not bool(valid_mask.any()):
            return None

        seq_len = min(logits.size(1), labels.size(1))
        if seq_len <= 1:
            return None

        logits = logits[:, :seq_len, :].float().contiguous()
        labels = labels[:, :seq_len].contiguous()

        labels_invalid = labels.clone()
        labels_invalid[valid_mask] = -100

        # Match HF causal LM shifting convention: keep logits and shift labels by one.
        shift_labels = torch.nn.functional.pad(labels_invalid, (0, 1), value=-100)[..., 1:].contiguous()
        vocab_size = logits.size(-1)

        ce_sum = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )
        num_items = (shift_labels != -100).sum()
        if int(num_items.item()) == 0:
            return None

        return ce_sum / num_items.to(ce_sum.device)
    


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
        generated_tokens = generated_tokens.to(score_tensor.device)
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
            log_probs = torch.log_softmax(score_tensor, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)

        mask_dtype = token_log_probs.dtype
        if pad_id is not None:
            gen_mask = (generated_tokens != pad_id).to(mask_dtype)
        else:
            gen_mask = torch.ones_like(generated_tokens, dtype=mask_dtype)
        lengths = gen_mask.sum(dim=-1).clamp(min=1.0)
        seq_log_prob = (token_log_probs * gen_mask).sum(dim=-1) / lengths

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
                    generated_sats = self.kernel._evaluate_formula_on_traces(generated_formula, self.eval_batch_size, 0)
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
            return None, valid_mask

        seq_log_prob = seq_log_prob[valid_mask]

        reward_tensor = torch.stack(reward_values).to(device=device, dtype=seq_log_prob.dtype)
        reward_mean = reward_tensor.mean().item()

        if self._reward_baseline is None:
            self._reward_baseline = reward_mean
            self._reward_sq_mean = (reward_tensor ** 2).mean().item()
        else:
            self._reward_baseline = (
                self.baseline_momentum * self._reward_baseline
                + (1.0 - self.baseline_momentum) * reward_mean
            )
            self._reward_sq_mean = (
                self.baseline_momentum * self._reward_sq_mean
                + (1.0 - self.baseline_momentum) * (reward_tensor ** 2).mean().item()
            )

        self._reward_variance = max(self._reward_sq_mean - self._reward_baseline ** 2, 1e-8)
        
        # Normalize advantage (reduces variance significantly)
        baseline = torch.tensor(self._reward_baseline, device=device, dtype=reward_tensor.dtype)
        std = torch.tensor(max(self._reward_variance ** 0.5, 1e-4), device=device)
        advantage = ((reward_tensor - baseline) / std).detach()

        reinforce_loss = -(advantage * seq_log_prob).mean() 
    
        
        return reinforce_loss, valid_mask


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