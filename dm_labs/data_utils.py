import random

import torch
from torch.utils.data import IterableDataset


DEFAULT_PROMPT = "Write a short story."


def format_as_chat(story_text: str, user_prompt: str = DEFAULT_PROMPT) -> str:
    story_text = story_text.strip()
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n{story_text}\n<|end|>\n"


class TokenBlockDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len, shuffle=False, seed=0, user_prompt: str = DEFAULT_PROMPT):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.user_prompt = user_prompt

    def __iter__(self):
        indices = list(range(len(self.hf_ds)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        buffer = []
        for idx in indices:
            text = format_as_chat(self.hf_ds[idx]["text"], user_prompt=self.user_prompt)
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(block, dtype=torch.long)


def collate_blocks(batch, pad_id: int):
    input_ids = torch.stack(batch, dim=0)
    attention_mask = input_ids != pad_id
    return {"input_ids": input_ids, "attention_mask": attention_mask}
