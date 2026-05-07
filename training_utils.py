import json
import math
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
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
        debug_metrics: bool = False,
        debug_step_interval: int = 50,
    ) -> None:
        self.output_dir = output_dir
        self.stage_name = stage_name or os.path.basename(os.path.normpath(output_dir))
        self.trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE = None
        self.trainer_kind: str = None
        self.debug_metrics = bool(debug_metrics)
        self.debug_step_interval = max(1, int(debug_step_interval))
        self.logs_dir = os.path.join(output_dir, "logs")
        self.metrics_path = os.path.join(self.logs_dir, "metrics_history.jsonl")
        self._train_start_time: float | None = None
        self._best_semantic_distance: float | None = None
        self._best_epoch: int | None = None
        self._last_eval_metrics: dict[str, float] = {}
        self._best_eval_metrics: dict[str, float] = {}
        self._stage_end_eval_metrics: dict[str, float] = {}
        self._metric_sums: dict[str, float | torch.Tensor] = {}
        self._metric_counts: dict[str, int] = defaultdict(int)
        self._rl_stats: dict[str, float] = defaultdict(float)
        self._eval_metric_sums: dict[str, float] = defaultdict(float)
        self._eval_metric_counts: dict[str, int] = defaultdict(int)

    _RL_VECTOR_KEYS = {
        "token_count_per_sample",
        "token_entropy_sum",
        "train_action_log_prob_sum",
        "valid_formula_mask_per_sample",
        "reward_per_sample",
        "advantage_per_sample",
        "value_sum_per_sample",
        "returns_sum",
        "returns_sq_sum",
        "value_err_sq_sum",
        "value_err_sum",
        "mc_reward_err_sq_sum_per_sample",
        "mc_returns_sum_per_sample",
        "mc_returns_sq_sum_per_sample",
    }

    _RL_LOCAL_SCALAR_KEYS = set()

    _RL_LOCAL_VECTOR_KEYS = {
        "advantage_pos_mean",
        "value_pos_mean",
        "returns_pos_mean",
        "pos_counts",
    }

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

    def _should_debug_print_step(self, state: TrainerState) -> bool:
        if not self.debug_metrics:
            return False
        step = int(state.global_step)
        if step <= 3:
            return True
        if step % self.debug_step_interval == 0:
            return True
        epoch = state.epoch
        if epoch is not None:
            epoch_float = float(epoch)
            epoch_rounded = round(epoch_float)
            # Print at the last optimizer step of each epoch (epoch index becomes an integer).
            if epoch_rounded >= 1 and abs(epoch_float - float(epoch_rounded)) < 1e-8:
                return True
        return False

    def _debug_print(self, tag: str, payload: dict) -> None:
        if not self.debug_metrics:
            return
        print(f"[MetricsDebug][{tag}] {json.dumps(payload, sort_keys=True)}")

    def _gather_metric_tensor(
        self,
        value: int | float | torch.Tensor,
        *,
        metric_key: str,
        state: TrainerState,
    ) -> torch.Tensor:
        accelerator = self.trainer.accelerator
        device = getattr(accelerator, "device", torch.device("cpu"))

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            else:
                tensor = tensor.reshape(-1)
            gathered = accelerator.gather(tensor)
            if self._should_debug_print_step(state):
                self._debug_print( 
                    "step_gather_metric",
                    {
                        "metric_key": metric_key,
                        "pregather_numel": int(tensor.numel()),
                        "gathered_numel": int(gathered.numel()),
                        "pregather_shape": list(tensor.shape),
                        "gathered_shape": list(gathered.shape),
                        "dtype": str(tensor.dtype),
                        "global_step": int(state.global_step),
                        "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                    },
                )
            return gathered

        tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
        gathered = accelerator.gather(tensor)
        if self._should_debug_print_step(state):
            self._debug_print(
                "step_gather_metric",
                {
                    "metric_key": metric_key,
                    "pregather_numel": int(tensor.numel()),
                    "gathered_numel": int(gathered.numel()),
                    "pregather_shape": list(tensor.shape),
                    "gathered_shape": list(gathered.shape),
                    "dtype": str(tensor.dtype),
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                },
            )
        return gathered

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

    def _accumulate_scalar_step_metric(self, key: str, gathered_values: torch.Tensor) -> None:
        if gathered_values.numel() == 0:
            return
        step_mean = float(gathered_values.detach().to(dtype=torch.float32).mean().cpu().item())
        self._accumulate_metric(key, step_mean)

    def _accumulate_sum_count(
        self,
        prefix: str,
        numerator_values: torch.Tensor,
        denominator_values: torch.Tensor,
    ) -> None:
        if numerator_values.numel() == 0 or denominator_values.numel() == 0:
            return
        numerator_f = numerator_values.detach().to(dtype=torch.float32)
        denominator_f = denominator_values.detach().to(dtype=torch.float32)
        self._rl_stats[f"{prefix}_sum"] += float(numerator_f.sum().cpu().item())
        self._rl_stats[f"{prefix}_count"] += float(denominator_f.sum().cpu().item())

    def _accumulate_sum_sq_count(self, prefix: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values_f = values.detach().to(dtype=torch.float32)
        self._rl_stats[f"{prefix}_sum"] += float(values_f.sum().cpu().item())
        self._rl_stats[f"{prefix}_sq_sum"] += float((values_f * values_f).sum().cpu().item())
        self._rl_stats[f"{prefix}_count"] += float(values_f.numel())

    def _sample_variance_from_moments(self, prefix: str) -> float | None:
        count = self._rl_stats.get(f"{prefix}_count", 0.0)
        if count <= 1.0:
            return None
        total = self._rl_stats.get(f"{prefix}_sum", 0.0)
        total_sq = self._rl_stats.get(f"{prefix}_sq_sum", 0.0)
        numerator = total_sq - (total * total) / count
        return max(0.0, numerator / (count - 1.0))

    def _accumulate_sample_counts(self, token_count: torch.Tensor, valid_mask: torch.Tensor) -> None:
        sample_count = float(token_count.numel())
        valid_sample_count = float(valid_mask.sum().cpu().item())
        self._rl_stats["sample_count"] += sample_count
        self._rl_stats["valid_sample_count"] += valid_sample_count

    def _accumulate_value_moment_totals(
        self,
        *,
        returns_sum: torch.Tensor,
        returns_sq_sum: torch.Tensor,
        value_err_sum: torch.Tensor,
        value_err_sq_sum: torch.Tensor,
        token_count: torch.Tensor,
        valid_with_tokens: torch.Tensor,
    ) -> None:
        valid_returns_sum = returns_sum.detach().to(dtype=torch.float32)[valid_with_tokens]
        valid_returns_sq_sum = returns_sq_sum.detach().to(dtype=torch.float32)[valid_with_tokens]
        valid_value_err_sum = value_err_sum.detach().to(dtype=torch.float32)[valid_with_tokens]
        valid_value_err_sq_sum = value_err_sq_sum.detach().to(dtype=torch.float32)[valid_with_tokens]
        valid_token_count = token_count.detach().to(dtype=torch.float32)[valid_with_tokens]

        self._rl_stats["returns_sum_total"] += float(valid_returns_sum.sum().cpu().item())
        self._rl_stats["returns_sq_sum_total"] += float(valid_returns_sq_sum.sum().cpu().item())
        self._rl_stats["value_err_sum_total"] += float(valid_value_err_sum.sum().cpu().item())
        self._rl_stats["value_err_sq_sum_total"] += float(valid_value_err_sq_sum.sum().cpu().item())
        self._rl_stats["value_token_count_total"] += float(valid_token_count.sum().cpu().item())

    def _accumulate_position_means(
        self,
        prefix: str,
        pos_mean: torch.Tensor,
        pos_counts: torch.Tensor,
    ) -> None:
        if pos_mean.numel() == 0 or pos_counts.numel() == 0:
            return

        pos_mean_f = pos_mean.detach().to(dtype=torch.float32).cpu()
        pos_counts_f = pos_counts.detach().to(dtype=torch.float32).cpu()
        length = min(int(pos_mean_f.numel()), int(pos_counts_f.numel()))
        if length <= 0:
            return

        pos_mean_f = pos_mean_f[:length]
        pos_counts_f = pos_counts_f[:length]
        pos_sum_list = (pos_mean_f * pos_counts_f).tolist()
        pos_count_list = pos_counts_f.tolist()

        sum_key = f"{prefix}_sum_list"
        count_key = f"{prefix}_count_list"
        existing_sum = self._rl_stats.get(sum_key)
        existing_count = self._rl_stats.get(count_key)

        if not isinstance(existing_sum, list) or not isinstance(existing_count, list):
            self._rl_stats[sum_key] = pos_sum_list
            self._rl_stats[count_key] = pos_count_list
            return

        if len(existing_sum) < length:
            existing_sum.extend([0.0] * (length - len(existing_sum)))
        if len(existing_count) < length:
            existing_count.extend([0.0] * (length - len(existing_count)))

        for idx in range(length):
            existing_sum[idx] += pos_sum_list[idx]
            existing_count[idx] += pos_count_list[idx]

    def _position_means_from_stats(
        self,
        prefix: str,
    ) -> tuple[list[float] | None, list[float] | None]:
        sum_list = self._rl_stats.get(f"{prefix}_sum_list")
        count_list = self._rl_stats.get(f"{prefix}_count_list")
        if not isinstance(sum_list, list) or not isinstance(count_list, list):
            return None, None
        if not sum_list or not count_list:
            return None, None
        means = [s / c if c > 0 else 0.0 for s, c in zip(sum_list, count_list)]
        return means, count_list

    def _accumulate_rl_step_metrics(self, gathered_vectors: dict[str, torch.Tensor]) -> None:
        token_count = gathered_vectors.get("token_count_per_sample")
        if token_count is None or token_count.numel() == 0:
            return

        token_count_f = token_count.detach().to(dtype=torch.float32)
        has_tokens = token_count_f > 0.0 
        valid_mask_tensor = gathered_vectors.get("valid_formula_mask_per_sample")
        if valid_mask_tensor is None:
            valid_mask = torch.ones_like(token_count_f, dtype=torch.bool)
        else:
            valid_mask = valid_mask_tensor.detach().to(dtype=torch.bool)

        self._accumulate_sample_counts(token_count_f, valid_mask)

        token_count_nonzero = token_count_f[has_tokens]

        entropy_sum = gathered_vectors.get("token_entropy_sum")
        if entropy_sum is not None:
            entropy_sum_f = entropy_sum.detach().to(dtype=torch.float32)[has_tokens]
            self._accumulate_sum_count("policy_entropy", entropy_sum_f, token_count_nonzero)

        action_lp_sum = gathered_vectors.get("train_action_log_prob_sum")
        if action_lp_sum is not None:
            action_lp_sum_f = action_lp_sum.detach().to(dtype=torch.float32)[has_tokens]
            self._accumulate_sum_count("action_logprob", action_lp_sum_f, token_count_nonzero)

        valid_with_tokens = valid_mask & has_tokens

        reward = gathered_vectors.get("reward_per_sample")
        if reward is not None:
            reward_valid = reward.detach().to(dtype=torch.float32)[valid_mask]
            self._accumulate_sum_sq_count("reward", reward_valid)

        advantage = gathered_vectors.get("advantage_per_sample")
        if advantage is not None:
            advantage_f = advantage.detach().to(dtype=torch.float32)
            if self.trainer_kind == "gae":
                advantage_values = (advantage_f[valid_with_tokens] / token_count_f[valid_with_tokens])
            else:
                advantage_values = advantage_f[valid_mask]
            self._accumulate_sum_sq_count("advantage", advantage_values)

        value_sum = gathered_vectors.get("value_sum_per_sample")
        if value_sum is not None:
            value_mean_per_sample = value_sum.detach().to(dtype=torch.float32)[valid_with_tokens] / token_count_f[valid_with_tokens]
            self._accumulate_sum_sq_count("value", value_mean_per_sample)

        mc_err_sq_vec = gathered_vectors.get("mc_reward_err_sq_sum_per_sample")
        mc_ret_sum_vec = gathered_vectors.get("mc_returns_sum_per_sample")
        mc_ret_sq_vec = gathered_vectors.get("mc_returns_sq_sum_per_sample")
        if mc_err_sq_vec is not None and mc_ret_sum_vec is not None and mc_ret_sq_vec is not None and valid_with_tokens.any():
            vwt = valid_with_tokens
            self._rl_stats["mc_token_count_total"] += float(token_count_f[vwt].sum().cpu().item())
            self._rl_stats["mc_err_sq_sum_total"] += float(mc_err_sq_vec.detach().to(dtype=torch.float32)[vwt].sum().cpu().item())
            self._rl_stats["mc_ret_sum_total"] += float(mc_ret_sum_vec.detach().to(dtype=torch.float32)[vwt].sum().cpu().item())
            self._rl_stats["mc_ret_sq_sum_total"] += float(mc_ret_sq_vec.detach().to(dtype=torch.float32)[vwt].sum().cpu().item())
            if value_sum is not None:
                self._rl_stats["mc_value_sum_total"] += float(value_sum.detach().to(dtype=torch.float32)[vwt].sum().cpu().item())

        returns_sum = gathered_vectors.get("returns_sum")
        returns_sq_sum = gathered_vectors.get("returns_sq_sum")
        value_err_sum = gathered_vectors.get("value_err_sum")
        value_err_sq_sum = gathered_vectors.get("value_err_sq_sum")
        if (
            returns_sum is not None
            and returns_sq_sum is not None
            and value_err_sum is not None
            and value_err_sq_sum is not None
        ):
            self._accumulate_value_moment_totals(
                returns_sum=returns_sum,
                returns_sq_sum=returns_sq_sum,
                value_err_sum=value_err_sum,
                value_err_sq_sum=value_err_sq_sum,
                token_count=token_count_f,
                valid_with_tokens=valid_with_tokens,
            )

        pos_counts = gathered_vectors.get("pos_counts")
        if pos_counts is not None:
            advantage_pos = gathered_vectors.get("advantage_pos_mean")
            if advantage_pos is not None:
                self._accumulate_position_means("advantage_pos", advantage_pos, pos_counts)
            value_pos = gathered_vectors.get("value_pos_mean")
            if value_pos is not None:
                self._accumulate_position_means("value_pos", value_pos, pos_counts)
            returns_pos = gathered_vectors.get("returns_pos_mean")
            if returns_pos is not None:
                self._accumulate_position_means("returns_pos", returns_pos, pos_counts)

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
        if self.trainer is None:
            raise AttributeError('Please call `attach_trainer()` before running the training loop.')

        step_metrics = getattr(self.trainer, "_last_train_metrics", None)
        if not step_metrics:
            return

        gathered_scalars: dict[str, torch.Tensor] = {}
        gathered_vectors: dict[str, torch.Tensor] = {}

        if self._should_debug_print_step(state):
            self._debug_print(
                "step_begin",
                {
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                    "trainer_kind": self.trainer_kind,
                    "local_rank": int(getattr(args, "local_rank", -1)),
                    "is_main_process": bool(self._is_main_process(args)),
                    "step_metric_keys": sorted(list(step_metrics.keys())),
                },
            )

        for key, value in step_metrics.items():
            if key in self._RL_VECTOR_KEYS:
                gathered_vectors[key] = self._gather_metric_tensor(value, metric_key=key, state=state)
            elif key in self._RL_LOCAL_VECTOR_KEYS:
                if self._is_main_process(args) and torch.is_tensor(value):
                    tensor = value.detach()
                    if tensor.ndim == 0:
                        tensor = tensor.reshape(1)
                    else:
                        tensor = tensor.reshape(-1)
                    gathered_vectors[key] = tensor
            elif key in self._RL_LOCAL_SCALAR_KEYS:
                if self._is_main_process(args):
                    scalar_value = self._scalar_to_float(value)
                    if scalar_value is not None:
                        self._accumulate_metric(key, scalar_value)
            else:
                gathered_scalars[key] = self._gather_metric_tensor(value, metric_key=key, state=state)

        if not self._is_main_process(args):
            return

        for key, values in gathered_scalars.items():
            self._accumulate_scalar_step_metric(key, values)

        self._accumulate_rl_step_metrics(gathered_vectors)

        if self._should_debug_print_step(state):
            debug_lengths = {k: int(v.numel()) for k, v in gathered_vectors.items()}
            self._debug_print(
                "step_post_accumulate",
                {
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                    "gathered_vector_numel": debug_lengths,
                    "rl_stats_snapshot": {
                        "sample_count": self._rl_stats.get("sample_count", 0.0),
                        "valid_sample_count": self._rl_stats.get("valid_sample_count", 0.0),
                        "policy_entropy_sum": self._rl_stats.get("policy_entropy_sum", 0.0),
                        "policy_entropy_count": self._rl_stats.get("policy_entropy_count", 0.0),
                        "action_logprob_sum": self._rl_stats.get("action_logprob_sum", 0.0),
                        "action_logprob_count": self._rl_stats.get("action_logprob_count", 0.0),
                        "reward_count": self._rl_stats.get("reward_count", 0.0),
                        "advantage_count": self._rl_stats.get("advantage_count", 0.0),
                        "value_count": self._rl_stats.get("value_count", 0.0),
                        "value_token_count_total": self._rl_stats.get("value_token_count_total", 0.0),
                    },
                },
            )

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

        sample_count = self._rl_stats.get("sample_count", 0.0)
        valid_sample_count = self._rl_stats.get("valid_sample_count", 0.0)
        if sample_count > 0.0:
            payload["train_valid_formula_ratio"] = valid_sample_count / sample_count

        entropy_count = self._rl_stats.get("policy_entropy_count", 0.0)
        if entropy_count > 0.0:
            payload["train_policy_entropy"] = self._rl_stats.get("policy_entropy_sum", 0.0) / entropy_count

        action_lp_count = self._rl_stats.get("action_logprob_count", 0.0)
        if action_lp_count > 0.0:
            payload["train_action_logprob_avg"] = self._rl_stats.get("action_logprob_sum", 0.0) / action_lp_count

        reward_count = self._rl_stats.get("reward_count", 0.0)
        if reward_count > 0.0:
            reward_mean = self._rl_stats.get("reward_sum", 0.0) / reward_count
            payload["train_reward_avg"] = reward_mean
            reward_var = self._sample_variance_from_moments("reward")
            if reward_var is not None:
                payload["train_reward_variance"] = reward_var

        advantage_count = self._rl_stats.get("advantage_count", 0.0)
        if advantage_count > 0.0:
            advantage_mean = self._rl_stats.get("advantage_sum", 0.0) / advantage_count
            payload["train_advantage_avg"] = advantage_mean
            advantage_var = self._sample_variance_from_moments("advantage")
            if advantage_var is not None:
                payload["train_advantage_variance"] = advantage_var

        value_count = self._rl_stats.get("value_count", 0.0)
        if value_count > 0.0:
            value_mean = self._rl_stats.get("value_sum", 0.0) / value_count
            payload["train_value_avg"] = value_mean
            value_var = self._sample_variance_from_moments("value")
            if value_var is not None:
                payload["train_value_variance"] = value_var

        value_token_count = self._rl_stats.get("value_token_count_total", 0.0)
        if value_token_count > 0.0:
            returns_sum_total = self._rl_stats.get("returns_sum_total", 0.0)
            returns_sq_sum_total = self._rl_stats.get("returns_sq_sum_total", 0.0)
            value_err_sum_total = self._rl_stats.get("value_err_sum_total", 0.0)
            value_err_sq_sum_total = self._rl_stats.get("value_err_sq_sum_total", 0.0)

            returns_mean = returns_sum_total / value_token_count
            returns_var = max(0.0, (returns_sq_sum_total / value_token_count) - (returns_mean * returns_mean))

            value_err_mean = value_err_sum_total / value_token_count
            value_err_centered_var = max(0.0, (value_err_sq_sum_total / value_token_count) - (value_err_mean * value_err_mean))

            payload["train_value_centered_residual_var"] = value_err_centered_var
            payload["train_value_explained_variance"] = 1.0 - (value_err_centered_var / max(returns_var, 1e-8))

        mc_token_count = self._rl_stats.get("mc_token_count_total", 0.0)
        if mc_token_count > 0.0:
            mc_err_sq_sum = self._rl_stats.get("mc_err_sq_sum_total", 0.0)
            mc_ret_sum = self._rl_stats.get("mc_ret_sum_total", 0.0)
            mc_ret_sq_sum = self._rl_stats.get("mc_ret_sq_sum_total", 0.0)
            mc_val_sum = self._rl_stats.get("mc_value_sum_total", 0.0)

            mc_ret_mean = mc_ret_sum / mc_token_count
            mc_ret_var = max(0.0, mc_ret_sq_sum / mc_token_count - mc_ret_mean * mc_ret_mean)

            mc_err_mean = (mc_ret_sum - mc_val_sum) / mc_token_count
            mc_err_var = max(0.0, mc_err_sq_sum / mc_token_count - mc_err_mean * mc_err_mean)

            payload["train_mc_value_ev"] = 1.0 - (mc_err_var / max(mc_ret_var, 1e-8))

        adv_pos_mean, pos_counts = self._position_means_from_stats("advantage_pos")
        if adv_pos_mean is not None:
            payload["train_advantage_pos_mean"] = adv_pos_mean
            if pos_counts is not None:
                payload["train_pos_counts"] = pos_counts

        value_pos_mean, pos_counts = self._position_means_from_stats("value_pos")
        if value_pos_mean is not None:
            payload["train_value_pos_mean"] = value_pos_mean
            if "train_pos_counts" not in payload and pos_counts is not None:
                payload["train_pos_counts"] = pos_counts

        returns_pos_mean, pos_counts = self._position_means_from_stats("returns_pos")
        if returns_pos_mean is not None:
            payload["train_returns_pos_mean"] = returns_pos_mean
            if "train_pos_counts" not in payload and pos_counts is not None:
                payload["train_pos_counts"] = pos_counts

        if payload:
            record = self._base_record("train_epoch_end", "train", state)
            record.update(payload)
            self._append_record(record)
            if self._is_main_process(args) and self.trainer.trainer_kind == 'gae':
                ev = payload.get("train_value_explained_variance", "N/A")
                mc_ev = payload.get("train_mc_value_ev", "N/A")
                print(f'EV = {ev}, MC_EV = {mc_ev}, critic_loss = {payload["train_critic_loss_mean"]}')

        if self.debug_metrics:
            self._debug_print(
                "epoch_end_payload",
                {
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                    "payload_keys": sorted(list(payload.keys())),
                    "payload": payload,
                    "scalar_metric_counts": dict(self._metric_counts),
                    "rl_stats": dict(self._rl_stats),
                },
            )

        self._metric_sums.clear()
        self._metric_counts.clear()
        self._rl_stats.clear()

    def on_prediction_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.trainer is None:
            return
        batch_losses = getattr(self.trainer, "_last_eval_batch_losses", {})
        if not batch_losses:
            return
        gathered: dict[str, torch.Tensor] = {}
        for key, value in batch_losses.items():
            gathered[key] = self._gather_metric_tensor(value, metric_key=key, state=state)
        if not self._is_main_process(args):
            return
        for key, values in gathered.items():
            if values.numel() > 0:
                step_mean = float(values.detach().to(dtype=torch.float32).mean().cpu().item())
                self._eval_metric_sums[key] += step_mean
                self._eval_metric_counts[key] += 1

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

        if self._eval_metric_counts:
            eval_means = {
                f"eval_{k}_mean": self._eval_metric_sums[k] / self._eval_metric_counts[k]
                for k in self._eval_metric_counts
            }
            payload.update(eval_means)
            print("[Eval] " + "  ".join(f"{k}: {v:.4f}" for k, v in eval_means.items()))
            self._eval_metric_sums.clear()
            self._eval_metric_counts.clear()

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
                self._best_eval_metrics = dict(payload)

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
        stage_record["best_semantic_distance"] = self._best_semantic_distance if self._best_semantic_distance is not None else 1.0
        self._append_record(stage_record)

        use_best = bool(getattr(args, "load_best_model_at_end", False)) and bool(self._best_eval_metrics)
        eval_metrics = self._best_eval_metrics if use_best else self._last_eval_metrics
        if eval_metrics:
            eval_record = self._base_record("eval_stage_end", "eval", state)
            eval_record.update(eval_metrics)
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
                 top_k_stage_end: int = 5,
                 enable_train_end_eval: bool = True,
                 debug_metrics: bool = False,
                 debug_step_interval: int = 50):
        """
        Args:
            tokenizer: LTLTokenizer for decoding generated sequences
            eval_dataset: LTLDataset to be used for model evaluation during training
            top_k_stage_end: Int that specifies the number of sequences to sample for end_of_stage metrics computation
        """
        
        self.tokenizer: LTLTokenizer = tokenizer
        self.top_k_stage_end = max(1, int(top_k_stage_end))
        self.enable_train_end_eval = bool(enable_train_end_eval)
        self.debug_metrics = bool(debug_metrics)
        self.debug_step_interval = max(1, int(debug_step_interval))
        self.trainer: CETrainer | REINFORCETrainerRB | REINFORCETrainerGAE = None
        self.eval_dataset: LTLDataset = None
        self.kernel: LTLKernel = None
        self.trainer_kind: str = None
        self.semantic_eval_batch_size: int = None
        self.metrics_logger: UnifiedMetricsLoggerCallback | None = None

    def _should_debug_print_step(self, state: TrainerState) -> bool:
        if not self.debug_metrics:
            return False
        step = int(state.global_step)
        if step <= 3:
            return True
        if step % self.debug_step_interval == 0:
            return True
        epoch = state.epoch
        if epoch is not None:
            epoch_float = float(epoch)
            epoch_rounded = round(epoch_float)
            # Print at the last optimizer step of each epoch (epoch index becomes an integer).
            if epoch_rounded >= 1 and abs(epoch_float - float(epoch_rounded)) < 1e-8:
                return True
        return False

    def _debug_print(self, tag: str, payload: dict) -> None:
        if not self.debug_metrics:
            return
        print(f"[MetricsDebug][{tag}] {json.dumps(payload, sort_keys=True)}")

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

        if bos_id in token_ids:
            start_idx = token_ids.index(bos_id) + 1
        else:
            start_idx = 0

        try:
            end_idx = token_ids.index(eos_id, start_idx)
        except ValueError:
            end_idx = len(token_ids)

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

        is_dist = dist.is_available() and dist.is_initialized()

        reference_model_path = None
        if self.trainer is not None:
            reference_model_path = getattr(self.trainer, "_ce_reference_model_path", None)
        
        reward_variance_total = 0.0
        reward_mean_total = 0.0
        reward_count = 0
        self_bleu_total = 0.0
        self_bleu_sq_total = 0.0
        self_bleu_count = 0
        entropy_num = 0.0
        entropy_den = 0.0
        action_lp_num = 0.0
        action_lp_den = 0.0
        kl_num = 0.0
        kl_den = 0.0

        kl_batches: list[dict[str, torch.Tensor]] = []

        gen_model = model.module if hasattr(model, "module") else model
        original_device = next(gen_model.parameters()).device
        original_training_mode = bool(gen_model.training)

        gen_model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                encoder_hidden_states = batch["encoder_hidden_states"].to(original_device, non_blocking=True)
                target_satisfaction = batch.get("target_satisfaction")
                if target_satisfaction is not None:
                    target_satisfaction = target_satisfaction.to(original_device)

                batch_size = encoder_hidden_states.size(0)
                k = self.top_k_stage_end

                generation = gen_model.generate(
                    encoder_hidden_states=encoder_hidden_states,
                    do_sample=True,
                    max_new_tokens=gen_model.config.n_positions,
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

                score_tensor = torch.stack(scores, dim=0).transpose(0, 1) # (B*k, T, V)
                score_log_probs = torch.log_softmax(score_tensor, dim=-1)
                score_probs = torch.exp(score_log_probs)
                token_entropy = -(score_probs * score_log_probs).sum(dim=-1)  # (B*k, T)

                pad_id = self.tokenizer.pad_token_id
                token_mask = (generated_tokens != pad_id)                     # (B*k, T)
                token_mask_f = token_mask.to(dtype=score_tensor.dtype)          # (B*k, T)

                token_log_probs = score_log_probs.gather(
                    dim=-1, index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)                                                    # (B*k, T)
                lengths = token_mask_f.sum(dim=-1).clamp(min=1.0)           # (B*k,)
                seq_log_prob = (token_log_probs * token_mask_f).sum(dim=-1) / lengths  # (B*k,)


                entropy_bkt = (token_entropy * token_mask_f).reshape(batch_size, k, -1)  # (B, k, T)
                mask_bkt = token_mask_f.reshape(batch_size, k, -1)                    # (B, k, T)
                per_sample_entropy_num = entropy_bkt.sum(dim=(1, 2))  # (B,)
                per_sample_entropy_den = mask_bkt.sum(dim=(1, 2))     # (B,)

                seq_lp_bk = seq_log_prob.reshape(batch_size, k)  # (B, k)
                per_sample_action_lp = seq_lp_bk.sum(dim=1)                 # (B,)
                per_sample_action_den = torch.full((batch_size,), float(k), dtype=torch.float32, device=original_device)

                per_sample_reward_variance = torch.zeros(batch_size, dtype=torch.float32, device=original_device)
                per_sample_reward_mean = torch.zeros(batch_size, dtype=torch.float32, device=original_device)
                per_sample_has_reward = torch.zeros(batch_size, dtype=torch.bool, device=original_device)
                per_sample_self_bleu = torch.zeros(batch_size, dtype=torch.float32, device=original_device)
                per_sample_has_bleu = torch.zeros(batch_size, dtype=torch.bool, device=original_device)

                generated_strs = self.tokenizer.batch_decode(
                    generated_tokens.detach().cpu(), skip_special_tokens=True
                )

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
                            generated_formula, self.semantic_eval_batch_size
                        )
                        if target_satisfaction is not None:
                            target_sats = target_satisfaction[b_idx]
                        else:
                            grouped_rewards[b_idx].append(0.0)
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

                for b_idx, rewards in enumerate(grouped_rewards):
                    if rewards:
                        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=original_device)
                        reward_mean = float(rewards_t.mean().item())
                        reward_var = float(torch.var(rewards_t, unbiased=False).item())
                        per_sample_reward_mean[b_idx] = reward_mean
                        per_sample_reward_variance[b_idx] = reward_var
                        per_sample_has_reward[b_idx] = True

                for b_idx, token_sequences in enumerate(grouped_token_sequences):
                    if len(token_sequences) >= 2:
                        bleu_vals = []
                        for i, cand in enumerate(token_sequences):
                            refs = [r for j, r in enumerate(token_sequences) if j != i]
                            bleu_vals.append(self._sentence_bleu(cand, refs))
                        if bleu_vals:
                            per_sample_self_bleu[b_idx] = float(sum(bleu_vals) / len(bleu_vals))
                            per_sample_has_bleu[b_idx] = True

                (
                    gathered_entropy_num,
                    gathered_entropy_den,
                    gathered_action_lp,
                    gathered_action_den,
                    gathered_reward_variance,
                    gathered_reward_mean,
                    gathered_has_reward,
                    gathered_self_bleu,
                    gathered_has_bleu,
                ) = self.trainer.accelerator.gather_for_metrics((
                    per_sample_entropy_num.to(original_device),
                    per_sample_entropy_den.to(original_device),
                    per_sample_action_lp.to(original_device),
                    per_sample_action_den.to(original_device),
                    per_sample_reward_variance.to(original_device),
                    per_sample_reward_mean.to(original_device),
                    per_sample_has_reward.to(original_device),
                    per_sample_self_bleu.to(original_device),
                    per_sample_has_bleu.to(original_device),
                ))

                if self._should_debug_print_step(state):
                    self._debug_print(
                        "stage_end_gather",
                        {
                            "global_step": int(state.global_step),
                            "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                            "pregather_numel": {
                                "per_sample_entropy_num": int(per_sample_entropy_num.numel()),
                                "per_sample_entropy_den": int(per_sample_entropy_den.numel()),
                                "per_sample_action_lp": int(per_sample_action_lp.numel()),
                                "per_sample_action_den": int(per_sample_action_den.numel()),
                                "per_sample_reward_variance": int(per_sample_reward_variance.numel()),
                                "per_sample_reward_mean": int(per_sample_reward_mean.numel()),
                                "per_sample_has_reward": int(per_sample_has_reward.numel()),
                                "per_sample_self_bleu": int(per_sample_self_bleu.numel()),
                                "per_sample_has_bleu": int(per_sample_has_bleu.numel()),
                            },
                            "gathered_numel": {
                                "gathered_entropy_num": int(gathered_entropy_num.numel()),
                                "gathered_entropy_den": int(gathered_entropy_den.numel()),
                                "gathered_action_lp": int(gathered_action_lp.numel()),
                                "gathered_action_den": int(gathered_action_den.numel()),
                                "gathered_reward_variance": int(gathered_reward_variance.numel()),
                                "gathered_reward_mean": int(gathered_reward_mean.numel()),
                                "gathered_has_reward": int(gathered_has_reward.numel()),
                                "gathered_self_bleu": int(gathered_self_bleu.numel()),
                                "gathered_has_bleu": int(gathered_has_bleu.numel()),
                            },
                        },
                    )

                entropy_num += float(gathered_entropy_num.sum().item())
                entropy_den += float(gathered_entropy_den.sum().item())
                action_lp_num += float(gathered_action_lp.sum().item())
                action_lp_den += float(gathered_action_den.sum().item())
                reward_variance_total += float(gathered_reward_variance[gathered_has_reward].sum().item())
                reward_mean_total += float(gathered_reward_mean[gathered_has_reward].sum().item())
                reward_count += int(gathered_has_reward.sum().item())
                valid_self_bleu = gathered_self_bleu[gathered_has_bleu].to(dtype=torch.float32)
                self_bleu_total += float(valid_self_bleu.sum().item())
                self_bleu_sq_total += float((valid_self_bleu * valid_self_bleu).sum().item())
                self_bleu_count += int(gathered_has_bleu.sum().item())

                if reference_model_path is not None:
                    shifted = sequences[:, :-1]
                    sem_rep = encoder_hidden_states.repeat_interleave(k, dim=0)
                    kl_batches.append({
                        "shifted": shifted.detach().cpu(),
                        "shifted_attention_mask": (shifted != self.tokenizer.pad_token_id).to(dtype=torch.long).detach().cpu(),
                        "encoder_hidden_states": sem_rep.detach().cpu(),
                        "token_mask_f": token_mask_f.detach().cpu(),
                        "re_log_probs": score_log_probs.detach().cpu().to(dtype=torch.float32),
                    })

        if reference_model_path is not None and kl_batches:
            reference_model = None
            try:
                gen_model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                reference_model = LTLModel.from_pretrained(reference_model_path)
                reference_model.to(original_device)
                reference_model.eval()

                with torch.no_grad():
                    for kl_batch in kl_batches:
                        shifted = kl_batch["shifted"].to(original_device, non_blocking=True)
                        shifted_attention_mask = kl_batch["shifted_attention_mask"].to(original_device, non_blocking=True)
                        sem_rep = kl_batch["encoder_hidden_states"].to(original_device, non_blocking=True)
                        token_mask_f = kl_batch["token_mask_f"].to(original_device, non_blocking=True)
                        re_log_probs = kl_batch["re_log_probs"].to(original_device, non_blocking=True)

                        t_steps = re_log_probs.size(1)
                        ce_logits = reference_model(
                            input_ids=shifted,
                            attention_mask=shifted_attention_mask,
                            encoder_hidden_states=sem_rep,
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
                gen_model.to(original_device)
                if original_training_mode:
                    gen_model.train()
                else:
                    gen_model.eval()

            if is_dist:
                kl_tensor = torch.tensor(
                    [kl_num, kl_den], dtype=torch.float64, device=original_device
                )
                dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
                kl_num, kl_den = kl_tensor[0].item(), kl_tensor[1].item()

        metrics: dict[str, float] = {}
        if reward_count > 0:
            metrics["eval_stage_reward_variance"] = reward_variance_total / reward_count
            metrics["eval_stage_reward_mean"] = reward_mean_total / reward_count
        if self_bleu_count > 0:
            self_bleu_mean = self_bleu_total / self_bleu_count
            metrics["eval_stage_self_bleu"] = self_bleu_mean
            self_bleu_var = max(0.0, (self_bleu_sq_total / self_bleu_count) - (self_bleu_mean * self_bleu_mean))
            metrics["eval_stage_self_bleu_variance"] = self_bleu_var
        if entropy_den > 0.0:
            metrics["eval_stage_policy_entropy"] = entropy_num / entropy_den
        if action_lp_den > 0.0:
            metrics["eval_stage_action_logprob_mean"] = action_lp_num / action_lp_den
        if kl_den > 0.0:
            metrics["eval_stage_sequence_kl_mean"] = kl_num / kl_den

        return metrics

    def _compute_semantic_metrics(
        self,
        *,
        args: TrainingArguments,
        state: TrainerState,
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

        gen_model = model.module if hasattr(model, "module") else model
        original_training_mode = bool(gen_model.training)
        gen_model.eval()
        try:
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch['input_ids'] 
                    target_embeddings = batch['encoder_hidden_states'].to(gen_model.device, non_blocking=True)
                    attention_mask = batch['attention_mask']
                    target_formulas = batch.get('target_formulas')
                    target_formula_strs = batch.get('target_formula_strs')
                    target_satisfaction = batch.get('target_satisfaction')
                    if target_satisfaction is not None:
                        target_satisfaction = target_satisfaction.to(gen_model.device)
                    
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

                    generated_ids = gen_model.generate(
                        encoder_hidden_states=target_embeddings,
                        max_length=gen_model.config.n_positions,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_strs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    per_sample_distance = torch.ones(size=(batch_size ,), device=gen_model.device, dtype=torch.float32)
                    per_sample_exact_str_match = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.bool)
                    per_sample_semantic_equivalent = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.bool)
                    per_sample_incorrect = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.bool)
                    per_sample_invalid = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.bool)
                    per_sample_generated_depth = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.float32)
                    per_sample_generated_length = torch.zeros(size=(batch_size ,), device=gen_model.device, dtype=torch.float32)

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
                    ) = self.trainer.accelerator.gather_for_metrics((
                        per_sample_distance, 
                        per_sample_exact_str_match, 
                        per_sample_semantic_equivalent, 
                        per_sample_incorrect, 
                        per_sample_invalid, 
                        per_sample_generated_depth, 
                        per_sample_generated_length
                    ))

                    if self._should_debug_print_step(state):
                        self._debug_print(
                            "semantic_eval_gather",
                            {
                                "global_step": int(state.global_step),
                                "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                                "pregather_numel": {
                                    "per_sample_distance": int(per_sample_distance.numel()),
                                    "per_sample_exact_str_match": int(per_sample_exact_str_match.numel()),
                                    "per_sample_semantic_equivalent": int(per_sample_semantic_equivalent.numel()),
                                    "per_sample_incorrect": int(per_sample_incorrect.numel()),
                                    "per_sample_invalid": int(per_sample_invalid.numel()),
                                    "per_sample_generated_depth": int(per_sample_generated_depth.numel()),
                                    "per_sample_generated_length": int(per_sample_generated_length.numel()),
                                },
                                "gathered_numel": {
                                    "gathered_per_sample_distance": int(gathered_per_sample_distance.numel()),
                                    "gathered_per_sample_exact_str_match": int(gathered_per_sample_exact_str_match.numel()),
                                    "gathered_per_sample_semantic_equivalent": int(gathered_per_sample_semantic_equivalent.numel()),
                                    "gathered_per_sample_incorrect": int(gathered_per_sample_incorrect.numel()),
                                    "gathered_per_sample_invalid": int(gathered_per_sample_invalid.numel()),
                                    "gathered_per_sample_generated_depth": int(gathered_per_sample_generated_depth.numel()),
                                    "gathered_per_sample_generated_length": int(gathered_per_sample_generated_length.numel()),
                                },
                            },
                        )
                    
                    total_distance += float(gathered_per_sample_distance.sum().item())
                    exact_string_matches += int(gathered_per_sample_exact_str_match.sum().item())
                    semantic_equivalent += int(gathered_per_sample_semantic_equivalent.sum().item())
                    incorrect += int(gathered_per_sample_incorrect.sum().item())
                    invalid += int(gathered_per_sample_invalid.sum().item())
                    total_samples += len(gathered_per_sample_distance)
                    generated_depth_sum += float(gathered_per_sample_generated_depth.sum().item())
                    generated_length_sum += float(gathered_per_sample_generated_length.sum().item())
        finally:
            if original_training_mode:
                gen_model.train()


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

        metric_values = self._compute_semantic_metrics(args=args, state=state, model=model, eval_dataloader=prepared_eval_dataloader)
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
            if self.debug_metrics:
                self._debug_print(
                    "on_evaluate_metrics",
                    {
                        "global_step": int(state.global_step),
                        "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                        "metric_values": metric_values,
                    },
                )

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
        if not self.enable_train_end_eval:
            return

        if self.trainer is None:
            raise AttributeError('Please call `attach_trainer()` before running the training loop.')
        
        eval_dataloader: DataLoader = self.trainer.get_eval_dataloader()
        eval_dataloader.collate_fn = lambda batch : self.tokenizer.collate_batch(batch, model.config.n_positions, include_metadata=True)
        prepared_eval_dataloader = self.trainer.accelerator.prepare(eval_dataloader)

        stage_end_metrics = self._compute_stage_end_metrics(args=args, state=state, model=model, eval_dataloader=prepared_eval_dataloader)
        if stage_end_metrics and self.metrics_logger is not None:
            self.metrics_logger.set_stage_end_eval_metrics(stage_end_metrics)
            if self.debug_metrics:
                self._debug_print(
                    "on_train_end_stage_metrics",
                    {
                        "global_step": int(state.global_step),
                        "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                        "stage_end_metrics": stage_end_metrics,
                    },
                )
