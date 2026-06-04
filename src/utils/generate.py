import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class StepRecord:
    step: int
    token_id: int
    token_str: str
    top_tokens: list[str]
    top_probs: list[float]


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    steps: list[StepRecord] = field(default_factory=list)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    return logits / temperature


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    for token_id in input_ids[0].tolist():
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


def generate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    seed: int | None = None,
    top_n_record: int = 20,
    device: torch.device | None = None,
) -> GenerationResult:
    if seed is not None:
        torch.manual_seed(seed)

    device = device or next(model.parameters()).device
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()
    steps: list[StepRecord] = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
            logits = _apply_temperature(logits, temperature)
            logits = _apply_top_k(logits, top_k)
            logits = _apply_top_p(logits, top_p)

            probs = F.softmax(logits, dim=-1)

            if do_sample and temperature > 0.0:
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

            top_probs, top_ids = torch.topk(probs[0], min(top_n_record, probs.size(-1)))
            steps.append(StepRecord(
                step=step,
                token_id=next_token_id.item(),
                token_str=tokenizer.decode([next_token_id.item()]),
                top_tokens=[tokenizer.decode([tid.item()]) for tid in top_ids],
                top_probs=top_probs.tolist(),
            ))

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    output_ids = generated_ids[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return GenerationResult(text=text, token_ids=output_ids, steps=steps)
