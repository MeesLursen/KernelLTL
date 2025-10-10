from transformers import GPT2Config
from tokenizer_class import LTLTokenizer

class LTLConfig(GPT2Config):
    """
    Convenience subclass of GPT2Config pre-populated with sensible defaults for LTL task.
    You can still pass any GPT2Config parameter via kwargs.
    """

    model_type = "gpt2"   # keep compatible with HF Auto classes

    def __init__(
        self,
        tokenizer: LTLTokenizer,
        n_positions: int = 512,
        n_embd: int = 1024, # must equal the size of the anchor set
        n_layer: int = 12,
        n_head: int = 16,
        add_cross_attention: bool = True,
        # pass-through kwargs to GPT2Config
        **kwargs
    ):
        kwargs.setdefault("add_cross_attention", add_cross_attention)

        super().__init__(
            vocab_size=tokenizer.vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            **kwargs,
        )