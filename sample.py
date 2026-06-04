import torch

from src.utils.config import get_config_parser, load_config
from src.utils.checkpoint import load_checkpoint
from src.model.gpt2 import load_model
from src.data.dataset import get_tokenizer
from src.utils.generate import generate


def parse_args():
    parser = get_config_parser()
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to load (optional)")
    parser.add_argument("--prompt", default="Harry looked at", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding (overrides do_sample)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top_n_print", type=int, default=5, help="Top-N probs to print per step")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    s = cfg["sampling"]

    max_new_tokens = args.max_new_tokens or s["max_new_tokens"]
    temperature = args.temperature if args.temperature is not None else s["temperature"]
    top_k = args.top_k if args.top_k is not None else s["top_k"]
    top_p = args.top_p if args.top_p is not None else s["top_p"]
    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else s["repetition_penalty"]
    do_sample = False if args.greedy else s["do_sample"]
    seed = args.seed if args.seed is not None else s["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    model = load_model(cfg)

    if args.checkpoint:
        meta = load_checkpoint(args.checkpoint, model, device=device)
        print(f"Loaded checkpoint: epoch={meta['epoch']} step={meta['global_step']} val_loss={meta['val_loss']:.4f}")

    model.to(device)
    model.eval()

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        seed=seed,
        top_n_record=cfg["visualization"]["distributions"]["top_n_tokens"],
        device=device,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {result.text}\n")

    print(f"--- Per-step top-{args.top_n_print} probabilities ---")
    for rec in result.steps:
        top = list(zip(rec.top_tokens[:args.top_n_print], rec.top_probs[:args.top_n_print]))
        chosen = f">> {repr(rec.token_str)}"
        pairs = "  ".join(f"{repr(t)}:{p:.3f}" for t, p in top)
        print(f"Step {rec.step:3d} {chosen:20s} | {pairs}")


if __name__ == "__main__":
    main()
