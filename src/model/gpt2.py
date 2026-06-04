from transformers import GPT2LMHeadModel


def load_model(cfg: dict) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]["name"])
    return model


def param_counts(model: GPT2LMHeadModel) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_param_counts(model: GPT2LMHeadModel) -> None:
    total, trainable = param_counts(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
