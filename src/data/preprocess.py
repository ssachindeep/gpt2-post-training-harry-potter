import re
import unicodedata
from datasets import DatasetDict

from src.data.download import get_dataset


def _clean(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\x80-\xFF]", "", text)
    return text


def get_clean_dataset(cfg: dict) -> DatasetDict:
    ds = get_dataset(cfg)
    col = cfg["data"]["text_column"]
    return ds.map(lambda ex: {col: _clean(ex[col])}, desc="Cleaning")
