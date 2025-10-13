"""Ad-hoc sanity checks for the Hugging Face-compatible LTL tokenizer.

Run manually with `python test.py` to view printed diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import torch

from tokenizer_pretrained_class import LTLTokenizer
from formula_utils import sample_formulas


def main() -> None:
	seed = 1
	device = "cpu"
	rng = torch.Generator(device=device).manual_seed(seed)

	n_ap = 5
	formulas = sample_formulas(
		n_formula=5,
		p_leaf=0.45,
		max_depth=6,
		n_ap=n_ap,
		force_tree=True,
		rng=rng,
		device=device,
	)

	tokenizer = LTLTokenizer(n_ap=n_ap)

	print("=== Tokenizer metadata ===")
	print(f"Vocab size: {tokenizer.vocab_size}")
	print(
		"Special tokens: "
		f"pad={tokenizer.pad_token_id}, "
		f"bos={tokenizer.bos_token_id}, "
		f"eos={tokenizer.eos_token_id}, "
		f"unk={tokenizer.unk_token_id}"
	)
	print()

	encoded_cases = []
	for idx, formula in enumerate(formulas, start=1):
		text = str(formula)
		print(f"--- Formula {idx} ---")
		print(f"Raw: {text}")

		tokens = tokenizer.tokenize(text)
		print(f"Tokens ({len(tokens)}): {tokens}")

		encoded = tokenizer.encode(text, add_special_tokens=True)
		print(f"Token IDs ({len(encoded)}): {encoded}")

		encoded_cases.append(encoded)

		decoded = tokenizer.decode(encoded, skip_special_tokens=True)
		print(f"Decoded: {decoded}")

		if text.strip() == decoded.strip():
			print("Round-trip check: ✅ matches original (ignoring whitespace)")
		else:
			print("Round-trip check: ⚠️ differs from original")

		print()

	print("=== Batch encode/decode ===")
	batch_texts = [str(f) for f in formulas]
	batch_outputs = tokenizer(batch_texts, padding=True, return_tensors="pt")
	print("Input IDs shape:", tuple(batch_outputs["input_ids"].shape))
	print("Attention mask shape:", tuple(batch_outputs["attention_mask"].shape))
	print("Batch input IDs (first two rows):")
	print(batch_outputs["input_ids"][:2])
	decoded_batch = tokenizer.batch_decode(
		batch_outputs["input_ids"], skip_special_tokens=True
	)
	print("Batch decoded texts:")
	for idx, text in enumerate(decoded_batch, start=1):
		print(f"  {idx}. {text}")
	print()

	print("=== Collate helper ===")
	# Fake embeddings to mimic dataset output ((Formula, semantic_embedding))
	embedding_dim = 1024
	max_sequence_length = max(len(case) for case in encoded_cases)
	dummy_batch = [
		(formulas[i], torch.randn(embedding_dim))
		for i in range(len(formulas))
	]
	collated = tokenizer.collate_batch(dummy_batch, max_len=max_sequence_length)
	print("Collate keys:", list(collated.keys()))
	print("labels shape:", tuple(collated["labels"].shape))
	print("attention_mask shape:", tuple(collated["attention_mask"].shape))
	print("semantic_embeddings shape:", tuple(collated["semantic_embeddings"].shape))
	print()

	print("=== Save & reload round-trip ===")
	tmp_dir = Path("./tmp_tokenizer_test")
	if tmp_dir.exists():
		shutil.rmtree(tmp_dir)
	tmp_dir.mkdir(parents=True, exist_ok=True)

	tokenizer.save_pretrained(tmp_dir)
	reloaded = LTLTokenizer.from_pretrained(tmp_dir)
	sample_text = str(formulas[2])
	assert tokenizer.decode(tokenizer.encode(sample_text)) == reloaded.decode(
		reloaded.encode(sample_text)
	)
	print(f"Reloaded tokenizer matches original on sample formula: {sample_text}")

	# Clean up temporary directory
	shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
