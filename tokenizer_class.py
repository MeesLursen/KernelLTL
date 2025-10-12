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

        self.pad_token, self.bos_token, self.eos_token, self.unk_token = pad_token, bos_token, eos_token, unk_token

        self.pad_id: int                = self.token_to_id[self.pad_token]
        self.bos_id: int                = self.token_to_id[self.bos_token]
        self.eos_id: int                = self.token_to_id[self.eos_token]
        self.unk_id: int                = self.token_to_id[self.unk_token]



    def save_vocab(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.all_tokens, f)



    # TODO: fix this to be more flexible, don't store vocab.json as just a list of tokens. Might want to add aditional tokens during tokenizer init.
    @classmethod
    def load_vocab(cls, path: str):
        import json
        with open(path, 'r') as f:
            tokens: list[str] = json.load(f)
        # reconstruct object: infer num_props from tokens
        prop_tokens = [t for t in tokens if t.startswith('p_')]
        num_props = len(prop_tokens)
        obj = cls(num_props)
        obj.all_tokens = tokens
        obj.ops_and_props = tokens[4:]
        obj.token_to_id = {t:i for i,t in enumerate(tokens)}
        obj.id_to_token = {i:t for t,i in obj.token_to_id.items()}
        obj.pad_id = obj.token_to_id['<pad>']
        obj.bos_id = obj.token_to_id['<bos>']
        obj.eos_id = obj.token_to_id['<eos>']
        obj.unk_id = obj.token_to_id['<unk>']
        return obj



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
            for tok in self.ops_and_props:
                if s.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue

        return tokens



    def encode(self, canonical_formula_str: str, max_length: int) -> list[int]:
        tokens = [self.bos_token] + self.tokenize(canonical_formula_str) + [self.eos_token]
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
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
                      batch: list[tuple[Formula, torch.Tensor]],
                      max_len: int):

        input_embeddings = []
        labels = []

        for formula, emb in batch:
            s = str(formula)
            ids = torch.tensor(self.encode(s, max_length=max_len), dtype=torch.long)
            labels.append(ids)
            input_embeddings.append(emb)

        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)  # (B, L)
        attention_mask = (labels != self.pad_id).long()

        loss_labels = labels.clone()
        loss_labels[loss_labels == self.pad_id] = -100

        encoder_embs = torch.stack(input_embeddings, dim=0).to(dtype=torch.float32)  # (B, m)
        if encoder_embs.device.type == "cpu" and torch.cuda.is_available():
            encoder_embs = encoder_embs.pin_memory()

        return {
            "labels": loss_labels,
            "input_ids": labels,
            "attention_mask": attention_mask,
            "semantic_embeddings": encoder_embs
        } 
