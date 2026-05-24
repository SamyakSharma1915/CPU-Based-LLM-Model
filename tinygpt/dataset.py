# tinygpt/dataset.py
import json
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, tokenizer, corpus_path="data/corpus.txt", val_fraction=0.05, seq_len=256, split="train"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        data = torch.tensor(ids, dtype=torch.long)
        n = data.size(0)
        split_index = int(n * (1 - val_fraction))
        if split == "train":
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]
        self.seq_len = seq_len
        self.n_samples = max(0, (self.data.size(0) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y

class SFTDataset(Dataset):
    def __init__(self, tokenizer, chat_path="data/chat_finetune.jsonl", seq_len=256):
        self.examples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(chat_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {chat_path} at line {line_num}: "
                        "each non-empty line must be a JSON object like "
                        '{"prompt": "...", "response": "..."}'
                    ) from exc
                prompt = obj.get("prompt", "")
                response = obj.get("response", "")
                input_ids, target_ids = tokenizer.encode_chat(prompt, response)
                if len(input_ids) > seq_len:
                    input_ids = input_ids[:seq_len]
                    target_ids = target_ids[:seq_len]
                else:
                    pad = seq_len - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_id] * pad
                    target_ids = target_ids + [-1] * pad
                x = [0 if t == -1 else t for t in input_ids]
                self.examples.append((torch.tensor(x, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))

        if not self.examples:
            raise ValueError(
                f"No fine-tuning examples were loaded from {chat_path}. "
                "Add at least one JSONL row with prompt/response fields."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
