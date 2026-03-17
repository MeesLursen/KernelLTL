import json
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from formula_class import Formula

class LTLTokenizer:
    def __init__(self,
                 n_ap: int,
                 pad_token='<pad>', bos_token='<bos>', eos_token='<eos>', unk_token='<unk>'):
        
        base_tokens = [pad_token, bos_token, eos_token, unk_token]
        op_tokens = ['~', 'AND', 'OR', '->', 'X', 'F', 'G', 'U', '(', ')']
        prop_tokens = [f"p_{i}" for i in range(n_ap)]
        all_tokens = base_tokens + op_tokens + prop_tokens

        self.base_tokens: list[str]     = base_tokens
        self.ops_and_props: list[str]   = op_tokens + prop_tokens
        self.all_tokens: list[str]      = all_tokens
        self.vocab_size: int            = len(self.all_tokens)
        self.token_to_id: dict[str,int] = {t:i for i,t in enumerate(all_tokens)}
        self.id_to_token: dict[str,int] = {i:t for t,i in self.token_to_id.items()}
        self.n_ap: int                   = n_ap

        self.pad_token, self.bos_token, self.eos_token, self.unk_token = pad_token, bos_token, eos_token, unk_token

        self.pad_token_id: int                = self.token_to_id[self.pad_token]
        self.bos_token_id: int                = self.token_to_id[self.bos_token]
        self.eos_token_id: int                = self.token_to_id[self.eos_token]
        self.unk_token_id: int                = self.token_to_id[self.unk_token]



    def save_state(self, path: str) -> str:
        data: dict[str, Any] = {
            "version": 1,
            "tokens": self.all_tokens,
            "base_tokens": self.base_tokens,
            "ops_and_props": self.ops_and_props,
            "n_ap": self.n_ap,
            "special_tokens": {
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
            },
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path



    def save_vocab(self, path: str):
        self.save_state(path)



    @classmethod
    def load_state(cls, path: str) -> "LTLTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return cls._from_token_list(data)

        tokens: list[str] = data["tokens"]
        n_ap = data.get("n_ap")
        if n_ap is None:
            n_ap = sum(1 for tok in tokens if tok.startswith('p_'))

        special_tokens = data.get("special_tokens", {})
        pad_token = special_tokens.get("pad_token", "<pad>")
        bos_token = special_tokens.get("bos_token", "<bos>")
        eos_token = special_tokens.get("eos_token", "<eos>")
        unk_token = special_tokens.get("unk_token", "<unk>")

        obj = cls(n_ap,
                  pad_token=pad_token,
                  bos_token=bos_token,
                  eos_token=eos_token,
                  unk_token=unk_token)

        obj.base_tokens = data.get("base_tokens", obj.base_tokens)
        obj.ops_and_props = data.get("ops_and_props", tokens[len(obj.base_tokens):])
        obj.all_tokens = tokens
        obj.token_to_id = {t: i for i, t in enumerate(tokens)}
        obj.id_to_token = {i: t for t, i in obj.token_to_id.items()}
        obj.vocab_size = len(tokens)
        obj.n_ap = n_ap
        obj.pad_token_id = obj.token_to_id[obj.pad_token]
        obj.bos_token_id = obj.token_to_id[obj.bos_token]
        obj.eos_token_id = obj.token_to_id[obj.eos_token]
        obj.unk_token_id = obj.token_to_id[obj.unk_token]

        return obj



    @classmethod
    def _from_token_list(cls, tokens: list[str]) -> "LTLTokenizer":
        prop_tokens = [t for t in tokens if t.startswith('p_')]
        num_props = len(prop_tokens)
        obj = cls(num_props)
        obj.all_tokens = tokens
        obj.ops_and_props = tokens[4:]
        obj.token_to_id = {t:i for i,t in enumerate(tokens)}
        obj.id_to_token = {i:t for t,i in obj.token_to_id.items()}
        obj.pad_token_id = obj.token_to_id['<pad>']
        obj.bos_token_id = obj.token_to_id['<bos>']
        obj.eos_token_id = obj.token_to_id['<eos>']
        obj.unk_token_id = obj.token_to_id['<unk>']
        obj.vocab_size = len(tokens)
        obj.n_ap = num_props
        return obj



    @classmethod
    def load_vocab(cls, path: str) -> "LTLTokenizer":
        return cls.load_state(path)



    def tokenize(self, canonical_formula_str: str) -> list[str]:
        s = canonical_formula_str.strip()
        tokens: list[str] = []
        i = 0
        while i < len(s):
            # skip whitespace
            if s[i].isspace():
                i += 1
                continue

            # rest_of_tokens
            matched = False
            for tok in self.ops_and_props:
                if s.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue

            raise ValueError(f"Unrecognized token sequence starting at position {i}: '{s[i:]}'")

        return tokens



    def encode(self, canonical_formula_str: str, max_length: int) -> list[int]:
        tokens = [self.bos_token] + self.tokenize(canonical_formula_str) + [self.eos_token]
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        if len(ids) >= max_length:
            return ids[:max_length]

        return ids



    def decode(self, token_ids: list[int], skip_special_tokens = True) -> str:
        tokens = [self.id_to_token[i] for i in token_ids]

        s = ""
        for t in tokens:
            if (t == "<bos>" or t == "<eos>" or t == "<pad>"):
                if not skip_special_tokens:
                    s += t
                else:
                    continue  # skip special tokens
            if t == ")":
                # remove trailing space before )
                s = s.rstrip() + ")"
            elif t == "(":
                # append ( directly without trailing space
                s += "("
            else:
                # regular token: prepend a space if not at start or after '('
                if len(s) > 0 and s[-1] not in " (":  
                    s += " "
                s += f"{t} "
        return s
    

    def batch_decode(self, sequences: torch.Tensor | list[list[int]], skip_special_tokens=True):
        """
        sequences: List[List[int]] or torch.Tensor of shape (B, L)
        Returns List[str] of decoded strings
        """
        # if it's a tensor, convert to list
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]



    def collate_batch(self, 
                      batch: list[dict[str, torch.Tensor | Formula | str]],
                      max_len: int,
                      include_metadata: bool = False):

        semantic_embeddings_ls = []
        input_ids_ls = []
        formulas_ls: list[Formula] = []
        formula_strs_ls: list[str] = []
        satisfactions_ls: list[torch.Tensor] = []
        formula_ids_list: list[int] = []

        for sample in batch:
            formula: Formula = sample["formula"]
            emb: torch.Tensor = sample["embedding"]
            idx: int = sample["formula_id"]

            formula_str = sample.get("formula_str")
            if formula_str is None:
                formula_str = str(formula)

            ids = torch.tensor(self.encode(formula_str, max_length=max_len), dtype=torch.long)
            input_ids_ls.append(ids)
            semantic_embeddings_ls.append(emb)
            formulas_ls.append(formula)
            formula_strs_ls.append(formula_str)
            formula_ids_list.append(idx)

            if include_metadata and "satisfaction" in sample:
                satisfactions_ls.append(sample["satisfaction"])

        input_ids = pad_sequence(input_ids_ls, batch_first=True, padding_value=self.pad_token_id)  # (B, L)
        attention_mask = (input_ids != self.pad_token_id).long()

        loss_labels = input_ids.clone()
        loss_labels[loss_labels == self.pad_token_id] = -100

        semantic_embs = torch.stack(semantic_embeddings_ls, dim=0).to(dtype=torch.float32)  # (B, m)

        formula_ids = torch.tensor(formula_ids_list, dtype=torch.long)

        batch_dict: dict[str, torch.Tensor | list[str] | list[Formula]] = {
            "labels": loss_labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "semantic_embeddings": semantic_embs,
            "formula_ids": formula_ids,
        }

        if include_metadata:
            batch_dict["target_formulas"] = formulas_ls
            batch_dict["target_formula_strs"] = formula_strs_ls
            if len(satisfactions_ls) != len(batch):
                raise ValueError("include_metadata=True but some samples lack 'satisfaction'")
            batch_dict["target_satisfaction"] = torch.stack(satisfactions_ls, dim=0)

        return batch_dict
