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
        vocab_size: int = 19,
        n_positions: int = 512,
        n_embd: int = 1024, # must equal the size of the anchor set
        n_layer: int = 12,
        n_head: int = 16,
        add_cross_attention: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        # pass-through kwargs to GPT2Config
        **kwargs
    ):
        kwargs.setdefault("add_cross_attention", add_cross_attention)

        self.vocab_size=vocab_size
        self.n_positions=n_positions
        self.n_embd=n_embd
        self.n_layer=n_layer
        self.n_head=n_head
        self.add_cross_attention=add_cross_attention
        self.bos_token_id=bos_token_id
        self.eos_token_id=eos_token_id
        self.pad_token_id=pad_token_id

        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )