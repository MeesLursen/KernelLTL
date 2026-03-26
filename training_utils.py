import json
import math
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from formula_utils import ParseError, str_to_formula
from kernel_class import LTLKernel
from dataset_class import LTLDataset
from tokenizer_pretrained_class import LTLTokenizer
from model_class import LTLModel
from ce_trainer import CETrainer
from reinforce_trainer import REINFORCETrainerGAE, REINFORCETrainerRB


class UnifiedMetricsLoggerCallback(TrainerCallback):
    """Centralized event-based metrics logger writing JSONL records."""

    def __init__(
        self,
        output_dir: str,
        stage_name: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.stage_name = stage_name or os.path.basename(os.path.normpath(output_dir))
        self.trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE = None
        self.trainer_kind: str = None
        self.logs_dir = os.path.join(output_dir, "logs")
        self.metrics_path = os.path.join(self.logs_dir, "metrics_history.jsonl")
        self._train_start_time: float | None = None
        self._best_semantic_distance: float | None = None
        self._best_epoch: int | None = None
        self._last_eval_metrics: dict[str, float] = {}
        self._stage_end_eval_metrics: dict[str, float] = {}
        self._metric_sums: dict[str, float | torch.Tensor] = {}
        self._metric_counts: dict[str, int] = defaultdict(int)

    def attach_trainer(self, trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE) -> None:
        self.trainer = trainer
        self.trainer_kind = trainer.trainer_kind

    def set_stage_end_eval_metrics(self, metrics: dict[str, float]) -> None:
        self._stage_end_eval_metrics = dict(metrics)

    def _is_main_process(self, args: TrainingArguments) -> bool:
        local_rank = getattr(args, "local_rank", -1)
        return local_rank in (-1, 0)

    def _format_mmss(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _append_record(self, record: dict) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _accumulate_metric(self, key: str, value: int | float | torch.Tensor) -> None:
        if isinstance(value, (int, float)):
            prev = self._metric_sums.get(key)
            if isinstance(prev, torch.Tensor):
                self._metric_sums[key] = prev + float(value)
            else:
                self._metric_sums[key] = float(prev) + float(value) if prev is not None else float(value)
            self._metric_counts[key] += 1
            return

        if torch.is_tensor(value) and value.numel() == 1:
            scalar = value.detach()
            prev = self._metric_sums.get(key)
            if isinstance(prev, torch.Tensor):
                self._metric_sums[key] = prev + scalar.to(device=prev.device, dtype=prev.dtype)
            elif isinstance(prev, (int, float)):
                self._metric_sums[key] = scalar + float(prev)
            else:
                self._metric_sums[key] = scalar
            self._metric_counts[key] += 1

    def _scalar_to_float(self, value: int | float | torch.Tensor) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if torch.is_tensor(value) and value.numel() == 1:
            return float(value.detach().to(dtype=torch.float32).cpu().item())
        return None

    def _base_record(
        self,
        event_type: str,
        split: str,
        state: TrainerState,
    ) -> dict:
        epoch_float = state.epoch if state.epoch is not None else -1.0
        epoch_idx = int(math.floor(epoch_float)) if epoch_float >= 0 else -1
        return {
            "event_type": event_type,
            "split": split,
            "stage_name": self.stage_name,
            "trainer_kind": self.trainer_kind,
            "global_step": int(state.global_step),
            "epoch": epoch_float,
            "epoch_index": epoch_idx,
            "wall_time_s": time.time(),
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._is_main_process(args):
            return
        self._train_start_time = time.time()
        os.makedirs(self.logs_dir, exist_ok=True)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._is_main_process(args):
            return

        step_metrics = {}
        if self.trainer is not None:
            step_metrics = getattr(self.trainer, "_last_train_metrics", {}) or {}
        else:
            raise AttributeError('Please call `attach_trainer()` before running the training loop.')

        for key, value in step_metrics.items():
            self._accumulate_metric(key, value)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if not self._is_main_process(args) or not logs:
            return

        grad_norm = logs.get("grad_norm")
        if isinstance(grad_norm, (int, float)):
            self._accumulate_metric("train_gradient_norm", grad_norm)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._is_main_process(args):
            return

        payload = {}
        for key, value_sum in self._metric_sums.items():
            count = self._metric_counts.get(key, 0)
            if count > 0:
                mean_value = self._scalar_to_float(value_sum / count)
                if mean_value is not None:
                    payload[f"{key}_mean"] = mean_value

        if payload:
            record = self._base_record("train_epoch_end", "train", state)
            record.update(payload)
            self._append_record(record)

        self._metric_sums.clear()
        self._metric_counts.clear()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ):
        if not self._is_main_process(args) or metrics is None:
            return

        payload = {}
        for key, value in metrics.items():
            scalar_value = self._scalar_to_float(value)
            if scalar_value is not None:
                payload[key] = scalar_value

        if not payload:
            return

        record = self._base_record("eval_epoch_end", "eval", state)
        record.update(payload)
        self._append_record(record)
        self._last_eval_metrics = payload

        semantic_distance = payload.get("eval_semantic_distance")
        if semantic_distance is not None:
            if self._best_semantic_distance is None or semantic_distance < self._best_semantic_distance:
                self._best_semantic_distance = semantic_distance
                epoch_float = state.epoch if state.epoch is not None else -1.0
                self._best_epoch = int(math.floor(epoch_float)) if epoch_float >= 0 else -1

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._is_main_process(args):
            return

        elapsed = 0.0
        if self._train_start_time is not None:
            elapsed = time.time() - self._train_start_time

        stage_record = self._base_record("train_stage_end", "train", state)
        stage_record["training_time_mmss"] = self._format_mmss(elapsed)
        stage_record["epochs_until_convergence"] = self._best_epoch if self._best_epoch is not None else -1
        self._append_record(stage_record)

        if self._last_eval_metrics:
            eval_record = self._base_record("eval_stage_end", "eval", state)
            eval_record.update(self._last_eval_metrics)
            if self._stage_end_eval_metrics:
                eval_record.update(self._stage_end_eval_metrics)
            self._append_record(eval_record)


class SemanticEvaluationCallback(TrainerCallback):
    """
    Custom callback for evaluating semantic similarity between generated and target formulas.
    Computes kernel embeddings of generated formulas and compares with target embeddings.
    """
    def __init__(self, 
                 tokenizer: LTLTokenizer,
                 top_k_stage_end: int = 5):
        """
        Args:
            tokenizer: LTLTokenizer for decoding generated sequences
            eval_dataset: LTLDataset to be used for model evaluation during training
            top_k_stage_end: Int that specifies the number of sequences to sample for end_of_stage metrics computation
        """
        
        self.tokenizer: LTLTokenizer = tokenizer
        self.top_k_stage_end = max(1, int(top_k_stage_end))
        self.trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE = None
        self.eval_dataset: LTLDataset = None
        self.kernel: LTLKernel = None
        self.trainer_kind: str = None
        self.semantic_eval_batch_size: int = None
        self.metrics_logger: UnifiedMetricsLoggerCallback | None = None

    def attach_trainer(self, trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE) -> None:
        self.trainer = trainer
        self.eval_dataset = trainer.eval_dataset
        self.kernel = trainer.kernel
        self.trainer_kind = trainer.trainer_kind
        self.semantic_eval_batch_size = trainer.semantic_eval_batch_size

    def attach_metrics_logger(self, metrics_logger: UnifiedMetricsLoggerCallback) -> None:
        self.metrics_logger = metrics_logger
    
    def _is_main_process(self, args: TrainingArguments) -> bool:
        local_rank = getattr(args, "local_rank", -1)
        return local_rank in (-1, 0)

    def _sentence_bleu(self, candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
        if not candidate:
            return 0.0
        if not references:
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

    def _extract_bleu_tokens_from_sequence(self, sequence: torch.Tensor) -> list[str]:
        token_ids = sequence.tolist()
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        start_idx = token_ids.index(bos_id) + 1
        end_idx = token_ids.index(eos_id, start_idx)

        content_ids = token_ids[start_idx:end_idx]
        filtered_ids = [token_id for token_id in content_ids if token_id != pad_id]

        return [str(token_id) for token_id in filtered_ids]

    def _compute_stage_end_metrics(
        self,
        *,
        args: TrainingArguments,
        state: TrainerState,
        model: LTLModel,
        eval_dataloader: DataLoader,
    ) -> dict[str, float]:
        if self.eval_dataset is None:
            return {}

        reference_model_path = None
        if self.trainer_kind is not None and self.trainer_kind != "ce":
            if self.trainer is not None:
                reference_model_path = getattr(self.trainer, "_ce_reference_model_path", None)

        reward_spread_total = 0.0
        reward_spread_count = 0
        self_bleu_total = 0.0
        self_bleu_count = 0
        entropy_num = 0.0
        entropy_den = 0.0
        action_lp_num = 0.0
        action_lp_den = 0.0
        kl_num = 0.0
        kl_den = 0.0

        kl_batches: list[dict[str, torch.Tensor]] = []

        original_device = next(model.parameters()).device
        original_training_mode = bool(model.training)

        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                semantic_embeddings = batch["semantic_embeddings"].to(model.device, non_blocking=True)
                target_satisfaction = batch.get("target_satisfaction")
                if target_satisfaction is not None:
                    target_satisfaction = target_satisfaction.to(self.kernel.device)

                gen_model = model.module if hasattr(model, 'module') else model
                k = self.top_k_stage_end
                generation = gen_model.generate(
                    semantic_embeddings=semantic_embeddings,
                    do_sample=True,
                    max_new_tokens=model.config.n_positions,
                    num_beams=1,
                    num_return_sequences=k,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=1.0,
                )

                sequences = generation.sequences
                sequences_cpu = sequences.detach().cpu()
                scores = list(generation.scores) if isinstance(generation.scores, tuple) else generation.scores
                if sequences is None or scores is None or len(scores) == 0:
                    continue

                total_steps = len(scores)
                seq_len = sequences.size(-1)
                prefix_len = max(0, seq_len - total_steps)
                generated_tokens = sequences[:, prefix_len:].long()
                if generated_tokens.size(-1) > total_steps:
                    generated_tokens = generated_tokens[:, :total_steps]

                score_tensor = torch.stack(scores, dim=0).transpose(0, 1)  # (B*k, T, V)
                score_log_probs = torch.log_softmax(score_tensor, dim=-1)
                score_probs = torch.exp(score_log_probs)
                token_entropy = -(score_probs * score_log_probs).sum(dim=-1)

                pad_id = self.tokenizer.pad_token_id
                token_mask = (generated_tokens != pad_id)
                token_mask_f = token_mask.to(dtype=score_tensor.dtype)
                entropy_num += float((token_entropy * token_mask_f).sum().detach().cpu().item())
                entropy_den += float(token_mask_f.sum().detach().cpu().item())

                token_log_probs = score_log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)
                lengths = token_mask_f.sum(dim=-1).clamp(min=1.0)
                seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths
                action_lp_num += float(seq_log_prob.sum().detach().cpu().item())
                action_lp_den += float(seq_log_prob.numel())

                if self.trainer_kind is not None and self.trainer_kind != "ce" and reference_model_path is not None:
                    shifted = sequences[:, :-1]
                    sem_rep = semantic_embeddings.repeat_interleave(k, dim=0)
                    kl_batches.append(
                        {
                            "shifted": shifted.detach().cpu(),
                            "semantic_embeddings": sem_rep.detach().cpu(),
                            "token_mask_f": token_mask_f.detach().cpu(),
                            "re_log_probs": score_log_probs.detach().cpu().to(dtype=torch.float32),
                        }
                    )

                generated_strs = self.tokenizer.batch_decode(generated_tokens.detach().cpu(), skip_special_tokens=True)
                batch_size = semantic_embeddings.size(0)

                grouped_rewards: list[list[float]] = [[] for _ in range(batch_size)]
                grouped_texts: list[list[str]] = [[] for _ in range(batch_size)]
                grouped_token_sequences: list[list[list[str]]] = [[] for _ in range(batch_size)]

                for idx, generated_str in enumerate(generated_strs):
                    b_idx = idx // k
                    grouped_texts[b_idx].append(generated_str)
                    grouped_token_sequences[b_idx].append(
                        self._extract_bleu_tokens_from_sequence(sequences_cpu[idx])
                    )
                    reward_val = 0.0
                    try:
                        generated_formula = str_to_formula(generated_str)
                        generated_sats = self.kernel._evaluate_formula_on_traces(
                            generated_formula,
                            self.semantic_eval_batch_size,
                        )
                        if target_satisfaction is not None:
                            target_sats = target_satisfaction[b_idx]
                        else:
                            reward_val = 0.0
                            grouped_rewards[b_idx].append(reward_val)
                            continue
                        hamming = torch.logical_xor(generated_sats, target_sats).to(torch.float32).mean().item()
                        reward_val = 1.0 - hamming
                    except ParseError:
                        reward_val = 0.0
                    except Exception as exc:
                        raise RuntimeError(
                            "Unexpected stage-end evaluation failure while scoring generated formulas."
                        ) from exc
                    grouped_rewards[b_idx].append(reward_val)

                for rewards in grouped_rewards:
                    if not rewards:
                        continue
                    reward_spread_total += float(max(rewards) - min(rewards))
                    reward_spread_count += 1

                for token_sequences in grouped_token_sequences:
                    if len(token_sequences) < 2:
                        continue
                    bleu_vals = []
                    for i, cand in enumerate(token_sequences):
                        refs = [r for j, r in enumerate(token_sequences) if j != i]
                        bleu_vals.append(self._sentence_bleu(cand, refs))
                    if bleu_vals:
                        self_bleu_total += float(sum(bleu_vals) / len(bleu_vals))
                        self_bleu_count += 1

        if self.trainer_kind is not None and self.trainer_kind != "ce" and reference_model_path is not None and kl_batches:
            reference_model = None
            try:
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                reference_model = LTLModel.from_pretrained(reference_model_path)
                reference_model.to(original_device)
                reference_model.eval()

                with torch.no_grad():
                    for kl_batch in kl_batches:
                        shifted = kl_batch["shifted"].to(original_device, non_blocking=True)
                        sem_rep = kl_batch["semantic_embeddings"].to(original_device, non_blocking=True)
                        token_mask_f = kl_batch["token_mask_f"].to(original_device, non_blocking=True)
                        re_log_probs = kl_batch["re_log_probs"].to(original_device, non_blocking=True)

                        t_steps = re_log_probs.size(1)
                        ce_logits = reference_model(
                            input_ids=shifted,
                            semantic_embeddings=sem_rep,
                        ).logits[:, -t_steps:, :]
                        ce_log_probs = torch.log_softmax(ce_logits, dim=-1)
                        re_probs = torch.exp(re_log_probs)
                        token_kl = (re_probs * (re_log_probs - ce_log_probs)).sum(dim=-1)
                        kl_num += float((token_kl * token_mask_f).sum().detach().cpu().item())
                        kl_den += float(token_mask_f.sum().detach().cpu().item())
            finally:
                if reference_model is not None:
                    del reference_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.to(original_device)
                if original_training_mode:
                    model.train()
                else:
                    model.eval()

        metrics: dict[str, float] = {}
        if reward_spread_count > 0:
            metrics["eval_stage_reward_spread"] = reward_spread_total / reward_spread_count
        if self_bleu_count > 0:
            metrics["eval_stage_self_bleu"] = self_bleu_total / self_bleu_count
        if entropy_den > 0.0:
            metrics["eval_stage_policy_entropy"] = entropy_num / entropy_den
        if self.trainer_kind is not None and self.trainer_kind != "ce" and action_lp_den > 0.0:
            metrics["eval_stage_action_logprob_mean"] = action_lp_num / action_lp_den
        if self.trainer_kind is not None and self.trainer_kind != "ce" and kl_den > 0.0:
            metrics["eval_stage_sequence_kl_mean"] = kl_num / kl_den

        return metrics

    def _compute_semantic_metrics(
        self,
        *,
        args: TrainingArguments,
        model: LTLModel,
        eval_dataloader: DataLoader,
    ) -> dict[str, float]:
        
        if self.eval_dataset is None:
            return {}

        total_distance = 0.0
        exact_string_matches = 0
        semantic_equivalent = 0
        incorrect = 0
        invalid = 0
        total_samples = 0
        generated_depth_sum = 0.0
        generated_length_sum = 0.0

        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'] 
                target_embeddings = batch['semantic_embeddings'].to(model.device, non_blocking=True)
                attention_mask = batch['attention_mask']
                target_formulas = batch.get('target_formulas')
                target_formula_strs = batch.get('target_formula_strs')
                target_satisfaction = batch.get('target_satisfaction')
                if target_satisfaction is not None:
                    target_satisfaction = target_satisfaction.to(self.kernel.device)
                
                if target_formula_strs is not None:
                    target_strs = target_formula_strs
                else:
                    target_strs = []
                    for ids, mask in zip(input_ids, attention_mask):
                        valid_ids = ids[mask.bool()].tolist()
                        target_strs.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))

                if target_formulas is None:
                    target_formulas = [str_to_formula(s) for s in target_strs]

                batch_size = target_embeddings.size(0)

                gen_model = model.module if hasattr(model, 'module') else model
                generated_ids = gen_model.generate(
                    semantic_embeddings=target_embeddings,
                    max_length=model.config.n_positions,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_strs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                per_sample_distance = torch.ones(size=(batch_size ,), device=self.kernel.device, dtype=torch.float32)
                per_sample_exact_str_match = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.bool)
                per_sample_semantic_equivalent = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.bool)
                per_sample_incorrect = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.bool)
                per_sample_invalid = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.bool)
                per_sample_generated_depth = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.float32)
                per_sample_generated_length = torch.zeros(size=(batch_size ,), device=self.kernel.device, dtype=torch.float32)

                for i in range(batch_size):
                    generated_str = generated_strs[i]
                    target_str = target_strs[i]
                    target_formula = target_formulas[i]

                    if target_satisfaction is not None:
                        target_sats = target_satisfaction[i]
                    else:
                        target_sats = self.kernel._evaluate_formula_on_traces(
                            formula=target_formula,
                            batch_size=self.semantic_eval_batch_size,
                        )

                    try:
                        generated_formula = str_to_formula(generated_str)
                        per_sample_generated_depth[i] = float(generated_formula.depth())
                        per_sample_generated_length[i] = float(len(generated_str))

                        if generated_str == target_str:
                            per_sample_exact_str_match[i] = True

                        generated_sats = self.kernel._evaluate_formula_on_traces(
                            formula=generated_formula,
                            batch_size=self.semantic_eval_batch_size,
                        )

                        xor = torch.logical_xor(target_sats, generated_sats)
                        distance = xor.to(dtype=torch.float32).mean().item()
                        per_sample_distance[i] = distance

                        if distance == 0.0:
                            per_sample_semantic_equivalent[i] = True
                        else:
                            per_sample_incorrect[i] = True
                    
                    except ParseError:
                        # Penalize for invalid formula by adding max distance
                        per_sample_invalid[i] = True
                    except Exception as exc:
                        raise RuntimeError(
                            "Unexpected semantic evaluation failure while scoring generated formulas."
                        ) from exc
                
                (gathered_per_sample_distance, 
                 gathered_per_sample_exact_str_match, 
                 gathered_per_sample_semantic_equivalent, 
                 gathered_per_sample_incorrect, 
                 gathered_per_sample_invalid, 
                 gathered_per_sample_generated_depth, 
                 gathered_per_sample_generated_length 
                 ) = self.trainer.accelerator.gather_for_metrics([
                     per_sample_distance, 
                     per_sample_exact_str_match, 
                     per_sample_semantic_equivalent, 
                     per_sample_incorrect, 
                     per_sample_invalid, 
                     per_sample_generated_depth, 
                     per_sample_generated_length
                 ])
                
                total_distance += float(gathered_per_sample_distance.sum().item())
                exact_string_matches += int(gathered_per_sample_exact_str_match.sum().item())
                semantic_equivalent += int(gathered_per_sample_semantic_equivalent.sum().item())
                incorrect += int(gathered_per_sample_incorrect.sum().item())
                invalid += int(gathered_per_sample_invalid.sum().item())
                total_samples += len(gathered_per_sample_distance)
                generated_depth_sum += float(gathered_per_sample_generated_depth.sum().item())
                generated_length_sum += float(gathered_per_sample_generated_length.sum().item())


        if total_samples <= 0:
            return {}
        
        
        avg_distance = total_distance / total_samples
        exact_string_match_rate = exact_string_matches / total_samples
        semantic_equivalent_rate = semantic_equivalent / total_samples
        incorrect_rate = incorrect / total_samples
        invalid_rate = invalid / total_samples
        generated_depth_mean = generated_depth_sum / max(total_samples - invalid, 1)
        generated_length_mean = generated_length_sum / max(total_samples - invalid, 1)

        return {
            "eval_semantic_distance": float(avg_distance),
            "eval_exact_match_rate": float(exact_string_match_rate),
            "eval_syntactic_equal_rate": float(exact_string_match_rate),
            "eval_semantic_equivalent_rate": float(semantic_equivalent_rate),
            "eval_incorrect_rate": float(incorrect_rate),
            "eval_invalid_rate": float(invalid_rate),
            "eval_generated_depth_mean": float(generated_depth_mean),
            "eval_generated_length_mean": float(generated_length_mean),
        }

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: LTLModel,
        metrics: dict | None = None,
        **kwargs,
    ):
        if self.trainer is None:
            raise AttributeError('Please call `attach_trainer()` before running the training loop.')

        eval_dataloader: DataLoader = self.trainer.get_eval_dataloader()
        eval_dataloader.collate_fn = lambda batch : self.tokenizer.collate_batch(batch, model.config.n_positions, include_metadata=True)
        prepared_eval_dataloader = self.trainer.accelerator.prepare(eval_dataloader)

        even_batches = getattr(self.trainer.accelerator, 'even_batches', None)
        if even_batches is not None:
            if self._is_main_process(args):
                print(f'even_batches = {self.trainer.accelerator.getattr('even_batches', None)}')
        else:
            if self._is_main_process(args):
                print("There was an issue retreiving the 'even_batches' argument.")    

        metric_values = self._compute_semantic_metrics(args=args, model=model, eval_dataloader=prepared_eval_dataloader)
        if not metric_values:
            return
        
        avg_distance = metric_values["eval_semantic_distance"]
        semantic_equiv_rate = metric_values["eval_semantic_equivalent_rate"]

        if metrics is not None:
            metrics.update(metric_values)

        if self._is_main_process(args):
            print(f"\n  Eval @ epoch {state.epoch} / step {state.global_step}:")
            print(f"  eval_semantic_distance: {avg_distance:.4f}")
            print(f"  eval_semantic_equivalent_rate: {semantic_equiv_rate:.4f}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: LTLModel,
        **kwargs,
    ):
        return

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: LTLModel,
        **kwargs,
    ):
        if self.trainer is None:
            raise AttributeError('Please call `attach_trainer()` before running the training loop.')
        
        eval_dataloader: DataLoader = self.trainer.get_eval_dataloader()
        eval_dataloader.collate_fn = lambda batch : self.tokenizer.collate_batch(batch, model.config.n_positions, include_metadata=True)
        prepared_eval_dataloader = self.trainer.accelerator.prepare(eval_dataloader)

        stage_end_metrics = self._compute_stage_end_metrics(args=args, state=state, model=model, eval_data_loader=prepared_eval_dataloader)
        if stage_end_metrics:
            self.metrics_logger.set_stage_end_eval_metrics(stage_end_metrics)


