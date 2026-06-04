import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from datasets import DatasetDict

from src.data.preprocess import get_clean_dataset


def get_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _tokenize(dataset: DatasetDict, tokenizer: GPT2TokenizerFast, cfg: dict) -> DatasetDict:
    tok_cfg = cfg["tokenizer"]
    max_len = tok_cfg["max_length"]
    stride = tok_cfg["stride"]
    col = cfg["data"]["text_column"]

    def tokenize_fn(batch):
        out = tokenizer(
            batch[col],
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors=None,
        )
        out.pop("overflow_to_sample_mapping", None)
        return out

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )


class GPT2Dataset(Dataset):
    def __init__(self, tokenized_split):
        self.data = tokenized_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_datasets(cfg: dict):
    tokenizer = get_tokenizer()
    clean_ds = get_clean_dataset(cfg)
    tokenized = _tokenize(clean_ds, tokenizer, cfg)
    return (
        GPT2Dataset(tokenized["train"]),
        GPT2Dataset(tokenized["validation"]),
        GPT2Dataset(tokenized["test"]),
    )
