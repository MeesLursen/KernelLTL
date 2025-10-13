# tests
from typing import Any, Callable
from training_utils import SemanticEvaluationCallback
from kernel_class import LTLKernel
from tokenizer_pretrained_class import LTLTokenizer

def attach_kwargs_logger(callback: SemanticEvaluationCallback) -> Callable[..., Any]:
    """Wrap the callback’s on_epoch_end to log kwargs before delegating."""
    original = callback.on_epoch_end

    def logged_on_epoch_end(*args, **kwargs):
        print("on_epoch_end kwargs:", kwargs)
        return original(*args, **kwargs)

    callback.on_epoch_end = logged_on_epoch_end  # type: ignore[method-assign]
    return original

kernel = LTLKernel(20,5,1)
tokenizer = LTLTokenizer(n_ap=5)
callback = SemanticEvaluationCallback(kernel=kernel, tokenizer=tokenizer)

attach_kwargs_logger(callback)