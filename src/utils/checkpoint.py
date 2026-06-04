import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def save_checkpoint(
    output_dir: str,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    tokenizer: GPT2TokenizerFast,
) -> str:
    ckpt_dir = Path(output_dir) / f"checkpoint-epoch{epoch}-step{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "checkpoint.pt",
    )
    tokenizer.save_pretrained(str(ckpt_dir))
    return str(ckpt_dir)


def load_checkpoint(
    ckpt_dir: str,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    path = Path(ckpt_dir) / "checkpoint.pt"
    state = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "val_loss": state["val_loss"],
    }
