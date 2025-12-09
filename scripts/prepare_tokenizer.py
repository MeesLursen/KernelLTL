"""Utility for instantiating and persisting an LTL tokenizer for curriculum stages."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from tokenizer_pretrained_class import LTLTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or copy a tokenizer vocabulary so it can be reused between curriculum stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", required=True, help="Directory where the tokenizer will be saved (via save_pretrained)")
    parser.add_argument("--n-ap", type=int, default=None, help="Number of atomic propositions. Required unless --kernel-dir or --vocab-file is given.")
    parser.add_argument("--kernel-dir", default=None, help="Optional kernel directory to infer n_ap from metadata.json")
    parser.add_argument("--vocab-file", default=None, help="Existing vocab.json to load instead of creating a fresh tokenizer")
    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--bos-token", default="<bos>")
    parser.add_argument("--eos-token", default="<eos>")
    parser.add_argument("--unk-token", default="<unk>")

    return parser.parse_args()


def _infer_n_ap(args: argparse.Namespace) -> int:
    if args.n_ap is not None:
        return int(args.n_ap)
    if args.kernel_dir:
        metadata_path = os.path.join(args.kernel_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Could not find kernel metadata at {metadata_path} to infer n_ap")
        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata: dict[str, Any] = json.load(fp)
        if "AP" not in metadata:
            raise KeyError("Kernel metadata missing 'AP' field")
        return int(metadata["AP"])
    raise ValueError("Provide either --n-ap, --kernel-dir, or --vocab-file to configure the tokenizer")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.vocab_file:
        tokenizer = LTLTokenizer(vocab_file=args.vocab_file)
    else:
        n_ap = _infer_n_ap(args)
        tokenizer = LTLTokenizer.from_token_count(
            n_ap=n_ap,
            pad_token=args.pad_token,
            bos_token=args.bos_token,
            eos_token=args.eos_token,
            unk_token=args.unk_token,
        )

    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
