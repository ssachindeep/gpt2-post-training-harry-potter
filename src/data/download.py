from datasets import load_dataset, DatasetDict


def get_dataset(cfg: dict) -> DatasetDict:
    data_cfg = cfg["data"]
    return load_dataset(
        data_cfg["dataset_name"],
        cache_dir=data_cfg["cache_dir"],
    )


def get_row(dataset: DatasetDict, split: str, index: int) -> dict:
    return dataset[split][index]
