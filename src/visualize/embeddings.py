from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import DatasetDict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.data.download import get_dataset
from src.data.preprocess import get_clean_dataset


def get_frequent_token_ids(cfg: dict, tokenizer: GPT2TokenizerFast) -> list[int]:
    n = cfg["visualization"]["embeddings"]["most_frequent_tokens"]
    ds = get_clean_dataset(cfg)
    col = cfg["data"]["text_column"]

    counter: Counter = Counter()
    for row in ds["train"]:
        ids = tokenizer.encode(row[col], add_special_tokens=False)
        counter.update(ids)

    return [tok_id for tok_id, _ in counter.most_common(n)]


def _reduce_2d(embeddings: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(embeddings)
        except Exception:
            pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def plot_embeddings(
    model: GPT2LMHeadModel,
    token_ids: list[int],
    tokenizer: GPT2TokenizerFast,
    output_path: str,
    title: str,
    method: str = "umap",
) -> None:
    weight = model.transformer.wte.weight.detach().cpu().numpy()
    vecs = weight[token_ids]

    coords = _reduce_2d(vecs, method)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.6)

    for i, tok_id in enumerate(token_ids[:200]):
        label = tokenizer.decode([tok_id]).strip()
        if label:
            ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
