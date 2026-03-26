import torch
from transformers import Trainer
from kernel_class import LTLKernel


class CETrainer(Trainer):
    """Custom CE trainer exposing low-overhead per-step metrics for epoch aggregation."""

    def __init__(self,
                 *args,
                 kernel: LTLKernel,
                 semantic_eval_batch_size: int = 10240,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.trainer_kind = 'ce'
        self.semantic_eval_batch_size = semantic_eval_batch_size
        self._last_train_metrics: dict[str, float | torch.Tensor] = {}
        self._sync_kernel_device(getattr(self.args, "device", None))

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]

        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        policy_entropy = None
        if logits is not None and labels is not None:
            token_log_probs = torch.log_softmax(logits, dim=-1)
            token_probs = torch.exp(token_log_probs)
            token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
            token_mask = labels.ne(-100)
            token_mask_f = token_mask.to(token_entropy.dtype)
            denom = token_mask_f.sum().clamp(min=1.0)
            policy_entropy = (token_entropy * token_mask_f).sum() / denom

        self._last_train_metrics = {
            "train_loss": loss.detach(),
        }
        if policy_entropy is not None:
            self._last_train_metrics["train_policy_entropy"] = policy_entropy.detach()

        if return_outputs:
            return loss, outputs
        return loss

    def _sync_kernel_device(self, device: torch.device | str | None) -> None:
        if device is None:
            return
        if self.kernel is None:
            return
        self.kernel.set_device(device)