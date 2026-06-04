import torch
from torch.utils.data import DataLoader

from src.utils.config import get_config_parser, load_config
from src.utils.logger import get_logger, log_metrics
from src.utils.checkpoint import load_checkpoint
from src.model.gpt2 import load_model
from src.data.dataset import get_datasets, get_tokenizer
from src.utils.generate import generate

FIXED_PROMPTS = [
    "Harry looked at",
    "Hermione opened her book and",
    "Dumbledore said calmly,",
    "The Sorting Hat announced",
    "Ron whispered to Harry,",
]


def parse_args():
    parser = get_config_parser()
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint directory (omit for base GPT-2)")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    return parser.parse_args()


def compute_test_loss(model, test_loader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(test_loader)


def run_qualitative(model, tokenizer, cfg, max_new_tokens, device, label) -> None:
    s = cfg["sampling"]
    print(f"\n{'='*60}")
    print(f"Qualitative samples — {label}")
    print(f"{'='*60}")
    for prompt in FIXED_PROMPTS:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=s["temperature"],
            top_k=s["top_k"],
            top_p=s["top_p"],
            repetition_penalty=s["repetition_penalty"],
            do_sample=s["do_sample"],
            seed=42,
            device=device,
        )
        print(f"\nPrompt : {prompt}")
        print(f"Output : {result.text}")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    logger = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _, _, test_ds = get_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])
    tokenizer = get_tokenizer()

    # --- Base pretrained GPT-2 ---
    logger.info("Loading base pretrained GPT-2...")
    base_model = load_model(cfg)
    base_model.to(device)

    base_loss = compute_test_loss(base_model, test_loader, device)
    logger.info(f"Base GPT-2 | Test Loss: {base_loss:.4f}")
    log_metrics(cfg["training"]["output_dir"], {"model": "base", "test_loss": base_loss})

    run_qualitative(base_model, tokenizer, cfg, args.max_new_tokens, device, label="Base GPT-2")

    # --- Fine-tuned checkpoint (if provided) ---
    if args.checkpoint:
        logger.info(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        ft_model = load_model(cfg)
        meta = load_checkpoint(args.checkpoint, ft_model, device=device)
        ft_model.to(device)
        logger.info(f"Checkpoint: epoch={meta['epoch']} step={meta['global_step']} val_loss={meta['val_loss']:.4f}")

        ft_loss = compute_test_loss(ft_model, test_loader, device)
        logger.info(f"Fine-tuned | Test Loss: {ft_loss:.4f}")
        log_metrics(cfg["training"]["output_dir"], {"model": "fine-tuned", "checkpoint": args.checkpoint, "test_loss": ft_loss})

        run_qualitative(ft_model, tokenizer, cfg, args.max_new_tokens, device, label="Fine-tuned GPT-2")

        print(f"\n{'='*60}")
        print(f"Test Loss Comparison")
        print(f"  Base GPT-2   : {base_loss:.4f}")
        print(f"  Fine-tuned   : {ft_loss:.4f}")
        print(f"  Improvement  : {base_loss - ft_loss:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
