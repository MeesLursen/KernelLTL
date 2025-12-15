from transformers import GPT2Config

class LTLConfig(GPT2Config):
    """
    Convenience subclass of GPT2Config pre-populated with sensible defaults for LTL task.
    """

    model_type = "gpt2"  # class attribute for HF Auto detection

    def __init__(
        self,
        vocab_size: int = 19,
        n_positions: int = 512,
        n_embd: int = 1024,
        n_layer: int = 12,
        n_head: int = 16,
        add_cross_attention: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            add_cross_attention=add_cross_attention,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        # Force instance attribute so it appears in config.json
        self.model_type = "gpt2"