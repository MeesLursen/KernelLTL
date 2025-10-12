import json
import os
import torch
from formula_class import Formula

from transformers import PreTrainedTokenizer

from tokenizer_class import LTLTokenizer as _LegacyLTLTokenizer


class LTLTokenizer(PreTrainedTokenizer):
    """Hugging Face compatible tokenizer for LTL formulas.

    This class wraps the existing project-specific tokenizer so we retain
    deterministic tokenisation, decoding and batching while exposing the
    standard ``PreTrainedTokenizer`` interface for downstream tooling.
    """

    vocab_files_names: dict[str, str] = {"vocab_file": "vocab.json"}
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    max_model_input_sizes: dict[str, int] = {}

    def __init__(
        self,
        vocab_file: str | None = None,
        n_ap: str | None = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        **kwargs,
    ) -> None:
        if vocab_file is None and n_ap is None:
            raise ValueError("Provide either `vocab_file` or `n_ap` when initialising LTLTokenizer.")

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )

        if vocab_file is not None:
            if not os.path.isfile(vocab_file):
                raise FileNotFoundError(f"The vocabulary file '{vocab_file}' could not be found.")
            legacy_tokenizer = _LegacyLTLTokenizer.load_vocab(vocab_file)
        else:
            legacy_tokenizer = _LegacyLTLTokenizer(
                n_ap=n_ap,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
            )

        self._legacy = legacy_tokenizer
        self._all_tokens: list[str] = list(legacy_tokenizer.all_tokens)
        self._token_to_id: dict[str, int] = dict(legacy_tokenizer.token_to_id)
        self._id_to_token: dict[int, str] = dict(legacy_tokenizer.id_to_token)
        self.vocab_file = vocab_file

        # Ensure the base class knows about our special tokens and their ids
        self.add_special_tokens(
            {
                "pad_token": pad_token,
                "bos_token": bos_token,
                "eos_token": eos_token,
                "unk_token": unk_token,
            }
        )
        self._sync_special_token_ids()

    def _sync_special_token_ids(self) -> None:
        """Keep legacy and HF views aligned for special token ids."""
        self._legacy.pad_id = self.pad_token_id
        self._legacy.bos_id = self.bos_token_id
        self._legacy.eos_id = self.eos_token_id
        self._legacy.unk_id = self.unk_token_id

    # ---------------------------------------------------------------------
    # Vocabulary helpers
    # ---------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self._all_tokens)

    def get_vocab(self) -> dict[str, int]:  # type: ignore[override]
        return dict(self._token_to_id)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:  # type: ignore[override]
        os.makedirs(save_directory, exist_ok=True)
        filename = "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        save_path = os.path.join(save_directory, filename)

        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(self._all_tokens, fp, ensure_ascii=False, indent=2)

        return (save_path,)

    # ---------------------------------------------------------------------
    # Core tokenisation logic (delegate to the legacy implementation)
    # ---------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:  # type: ignore[override]
        return self._legacy.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:  # type: ignore[override]
        return self._token_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:  # type: ignore[override]
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:  # type: ignore[override]
        token_ids = [self._convert_token_to_id(t) for t in tokens]
        return self._legacy.decode(token_ids, skip_special_tokens=True)

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:  # type: ignore[override]
        # `clean_up_tokenization_spaces` is ignored because the legacy decoder already
        # applies the desired formatting for parentheses and spacing.
        return self._legacy.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: torch.Tensor | list[list[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> list[str]:  # type: ignore[override]
        return self._legacy.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )

    # ---------------------------------------------------------------------
    # Special token handling
    # ---------------------------------------------------------------------
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:  # type: ignore[override]
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for LTL tokenization.")

        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:  # type: ignore[override]
        if already_has_special_tokens:
            return [1 if token_id in {self.bos_token_id, self.eos_token_id, self.pad_token_id} else 0 for token_id in token_ids_0]

        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for LTL tokenization.")

        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:  # type: ignore[override]
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for LTL tokenization.")

        total_length = len(token_ids_0) + self.num_special_tokens_to_add(pair=False)
        return [0] * total_length

    def num_special_tokens_to_add(self, pair: bool = False) -> int:  # type: ignore[override]
        if pair:
            raise NotImplementedError("Pair sequences are not supported for LTL tokenization.")
        return 2  # BOS and EOS

    # ---------------------------------------------------------------------
    # Add/remove tokens
    # ---------------------------------------------------------------------
    def _add_tokens(self, new_tokens: list[str], special_tokens: bool = False) -> int:  # type: ignore[override]
        added = 0
        for token in new_tokens:
            if token in self._token_to_id:
                continue
            index = len(self._all_tokens)
            self._all_tokens.append(token)
            self._token_to_id[token] = index
            self._id_to_token[index] = token
            added += 1

        if added:
            self._legacy.all_tokens = list(self._all_tokens)
            self._legacy.ops_and_props = self._all_tokens[4:]
            self._legacy.token_to_id = dict(self._token_to_id)
            self._legacy.id_to_token = dict(self._id_to_token)
            self._legacy.vocab_size = len(self._all_tokens)
            self._sync_special_token_ids()

        return added

    # ---------------------------------------------------------------------
    # Collation helpers (delegated to the existing implementation)
    # ---------------------------------------------------------------------
    def collate_batch(self, batch: list[tuple[Formula, torch.Tensor]], max_len: int):
        return self._legacy.collate_batch(batch, max_len=max_len)

    # ---------------------------------------------------------------------
    # Convenience constructors
    # ---------------------------------------------------------------------
    @classmethod
    def from_token_count(
        cls,
        n_ap: int,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        **kwargs,
    ) -> "LTLTokenizer":
        return cls(
            vocab_file=None,
            n_ap=n_ap,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )

