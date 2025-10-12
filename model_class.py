import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from config_class import LTLConfig

class LTLModel(nn.Module):
    """
    Wrapper that:
      - constructs an AutoModelForCausalLM using an LTLConfig
      - optionally projects semantic embeddings to model hidden dim and uses them as encoder_hidden_states
    """
    def __init__(self, config: LTLConfig, semantic_emb_dim: int = None):
        """
        Args:
            config: LTLConfig instanc
            semantic_emb_dim: dimensionality of your kernel embeddings (if provided, used to build projection)
            device: torch.device (optional)
        """
        super().__init__()

        self.config = config

        # instantiate HF causal LM (random init)
        self.base: GPT2LMHeadModel = GPT2LMHeadModel(config)

        # projection from semantic embedding to model hidden size (if needed)
        self.semantic_emb_dim = semantic_emb_dim
        if semantic_emb_dim is not None and semantic_emb_dim != self.config.n_embd:
            self.encoder_proj = nn.Linear(semantic_emb_dim, self.config.n_embd)
        else:
            self.encoder_proj = None



    @property
    def device(self) -> torch.device:
        """Current device of the underlying language model."""
        return next(self.base.parameters()).device



    def build_encoder_states(self, semantic_embeddings: torch.Tensor | None) -> torch.Tensor:
        """
        Convert kernel output (B, semantic_emb_dim) -> encoder_hidden_states (B, 1, n_embd).
        If encoder_proj is None and semantic_emb_dim==n_embd, this is effectively an unsqueeze.
        """
        if semantic_embeddings is None:
            return None

        device = self.device
        x = semantic_embeddings.to(device, non_blocking=True).float()  # (B, E_sem)
        if self.encoder_proj is not None:
            x = self.encoder_proj(x)  # (B, n_embd)
        if x.ndim == 2:  # (B, n_embd)
            return x.unsqueeze(1)   # (B, 1, n_embd)
        elif x.ndim == 3:
            return x                # assume already (B, 1, n_embd)
        else:
            raise ValueError("semantic_embeddings must be (B, E) or (B, 1, E) after projection")



    def forward(self, 
                input_ids: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                semantic_embeddings: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None,
                **kwargs):
        """
        Build encoder_hidden_states from semantic_embeddings and delegate to HF model.
        Extra kwargs are passed to HF model (use_cache, output_attentions, etc).
        """
        # build encoder hidden states
        enc_states = None
        if semantic_embeddings is not None:
            enc_states = self.build_encoder_states(semantic_embeddings)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    enc_states.size(0),
                    enc_states.size(1),
                    dtype=torch.long,
                    device=self.device
                )

        # move inputs to device
        if input_ids is not None:
            input_ids = input_ids.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device, non_blocking=True)

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    


    def generate(self, *args, semantic_embeddings: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, **kwargs):
        """
        Same arguments semantics as HF generate but accept semantic_embeddings (B, E) and will inject encoder_hidden_states.
        Example: generated = model.generate(input_ids=..., semantic_embeddings=sem, max_length=..., num_beams=...)
        """
        enc_states = None
        if semantic_embeddings is not None:
            enc_states = self.build_encoder_states(semantic_embeddings)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    enc_states.size(0),
                    enc_states.size(1),
                    dtype=torch.long,
                    device=self.device
                )

        # pass through to HF generate, include encoder_hidden_states kwargs
        return self.base.generate(*args, encoder_hidden_states=enc_states, encoder_attention_mask=encoder_attention_mask, **kwargs)



    # convenience saving/loading (saves projector separately)
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.base.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)
        if self.encoder_proj is not None:
            torch.save(self.encoder_proj.state_dict(), os.path.join(save_directory, "encoder_proj.pt"))



    @classmethod
    def from_pretrained(cls, load_directory: str, device: torch.device = None):
        """
        Load HF model from save_directory and attach a projection (expected to be saved at encoder_proj.pt).
        """
        cfg = LTLConfig.from_pretrained(load_directory)
        proj_path = os.path.join(load_directory, "encoder_proj.pt")
        if not os.path.exists(proj_path):
            semantic_emb_dim = None
        else:
            semantic_emb_dim = torch.load(proj_path, map_location='cpu')['weight'].size(dim=1)
            
        inst = cls(cfg, semantic_emb_dim=semantic_emb_dim)
        # load HF weights
        inst.base = AutoModelForCausalLM.from_pretrained(load_directory)
        if device is not None:
            inst.to(device)
        # load projector if present
        if inst.encoder_proj is not None:
            inst.encoder_proj.load_state_dict(torch.load(proj_path, map_location=inst.device))
        return inst
