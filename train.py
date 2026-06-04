import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import get_config_parser, load_config
from src.utils.logger import get_logger, log_metrics
from src.utils.checkpoint import save_checkpoint
from src.model.gpt2 import load_model, log_param_counts
from src.data.dataset import get_datasets, get_tokenizer


def parse_args():
    parser = get_config_parser()
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline and exit")
    return parser.parse_args()


def dry_run(cfg: dict, logger) -> None:
    logger.info("=== Dry Run ===")

    logger.info("Step 1-2: Downloading and loading training set...")
    train_ds, _, _ = get_datasets(cfg)
    logger.info(f"Train samples: {len(train_ds)}")

    logger.info("Step 3-4: Tokenizing one batch...")
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"])
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    logger.info(f"Batch input_ids shape: {input_ids.shape}")

    logger.info("Step 5-6: Loading GPT-2 and embedding tokens...")
    model = load_model(cfg)
    model.eval()
    log_param_counts(model)
    embeddings = model.transformer.wte(input_ids)
    logger.info(f"Token embeddings shape: {embeddings.shape}")

    logger.info("Step 7-8: Running forward pass through 12 MHSA transformer blocks...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logger.info(f"Logits shape: {logits.shape}")

    logger.info("Step 9: Converting logits to probabilities via softmax...")
    probs = F.softmax(logits, dim=-1)
    logger.info(f"Probabilities shape: {probs.shape}")

    logger.info("Step 10: Calculating cross-entropy loss...")
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    logger.info(f"Cross-entropy loss: {loss.item():.4f}")
    logger.info("=== Dry Run Complete: Pipeline OK ===")


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


def train(cfg: dict, logger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds, val_ds, test_ds = get_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    tokenizer = get_tokenizer()
    model = load_model(cfg)
    model.to(device)
    log_param_counts(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    grad_accum_steps = cfg["training"]["gradient_accumulation_steps"]
    save_steps = cfg["training"]["save_steps"]
    epochs = cfg["training"]["epochs"]
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % save_steps == 0:
                    ckpt_dir = save_checkpoint(output_dir, model, optimizer, epoch, global_step, float("inf"), tokenizer)
                    logger.info(f"Step checkpoint saved: {ckpt_dir}")

            if (step + 1) % 100 == 0:
                logger.info(f"Epoch {epoch} | Step {step+1} | Loss {epoch_loss / (step+1):.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        log_metrics(output_dir, {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        ckpt_dir = save_checkpoint(output_dir, model, optimizer, epoch, global_step, avg_val_loss, tokenizer)
        logger.info(f"Epoch checkpoint saved: {ckpt_dir}")

    avg_test_loss = evaluate(model, test_loader, device)
    logger.info(f"Final Test Loss: {avg_test_loss:.4f}")
    log_metrics(output_dir, {"test_loss": avg_test_loss})


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    logger = get_logger()

    if args.dry_run:
        dry_run(cfg, logger)
    else:
        train(cfg, logger)


if __name__ == "__main__":
    main()
