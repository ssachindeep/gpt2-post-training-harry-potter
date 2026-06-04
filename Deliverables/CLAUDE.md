# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role
Principal ML Engineer who write minimal yet effective code. 

## Project Overview

This is an IISc Deep Learning assignment: fine-tune GPT-2 small on the Harry Potter books dataset (`WutYee/HarryPotter_books_1to7`), expose flexible sampling controls, and visualize token embeddings and per-step generation probability distributions.

## Entry Points

| Script | Purpose |
|--------|---------|
| `train.py` | Fine-tune GPT-2 on the Harry Potter dataset |
| `eval.py` | Evaluate on test split; report loss and qualitative samples |
| `sample.py` | Run inference from CLI with sampling controls |
| `app.py` | Streamlit local inference app with per-step prob visualization |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (local dry run)
python train.py --config config/default.yaml

# Evaluate with a checkpoint
python eval.py --config config/default.yaml --checkpoint ./checkpoints/<name>

# Sample
python sample.py --prompt "Harry looked at" --max_new_tokens 100

# Launch app
streamlit run app.py
```

## Configuration

All runtime settings live in `config/default.yaml`. CLI overrides use argparse dot notation (e.g. `--training.epochs 5`). Do not hardcode hyperparameters in source files — all tunable values belong in the config.

## Architecture

```
src/
  data/
    download.py      # loads dataset via HuggingFace datasets
    preprocess.py    # text cleaning (NFC, whitespace, control chars)
    dataset.py       # fixed-length causal LM chunks with labels=-100 at padding
  model/
    gpt2.py          # thin wrapper around GPT2LMHeadModel.from_pretrained("gpt2")
  utils/
    config.py        # loads/merges YAML config with CLI overrides
    logger.py        # training loop logging
    checkpoint.py    # save/load checkpoints
    generate.py      # manual step-by-step generation (exposes per-step logits)
  visualize/
    embeddings.py    # UMAP/t-SNE of model.transformer.wte.weight
    distributions.py # per-step top-N probability bar charts + optional GIF
```

Key design constraint: generation is implemented manually in `src/utils/generate.py` (not using `model.generate()`) so every step exposes raw logits for visualization. The generation pipeline is: logits → temperature → top-k → top-p → repetition penalty → softmax → sample/argmax → record top-N probs.

## Tokenizer

Always use `GPT2TokenizerFast.from_pretrained("gpt2")`. Never retrain or replace the tokenizer — pretrained embeddings depend on the original GPT-2 token IDs.

## Visualization Outputs

- `Deliverables/Plots/` — embedding plots (pretrained vs fine-tuned), per-step distribution charts
- `visualization/` — intermediate artifacts during a run

## Kaggle Training Workflow

Write and test all code locally (dry run), then upload to Kaggle for GPU training. After downloading the checkpoint, run `eval.py`, `sample.py`, and `app.py` locally. The checkpoint path returned by Kaggle is passed via `--checkpoint`.

## Deliverables Directory

`Deliverables/` must contain: updated `SPEC.md`, `AGENTS.md` (or `CLAUDE.md`), Claude `/usage` output, `Plots/`, and `Reflection.md` (written by the student, not generated).

## Done
1. Review config/default.yaml 
2. Read the project structure and only create the files, I will verify if files are created or not.
3. From config and spec file, fill requirements txt file with the dependencies.
4. Implement default.yaml configuration to config.py file which would be used by code.
5. Implement download.py and I should be able to access any random row data from the downloaded data.
6. Preprocessing of the data. It should be able to load the data, maintina the dataset split of train,validate and test. Clean the data
7. Implement dataset.py: load GPT-2 tokenizer, set pad token to EOS, tokenize cleaned text with GPT-2 BPE, Validate with `python3 -m py_compile src/data/dataset.py` and print one train batch shape.
8. Implement gpt2.py: load gpt2, print/log total parameter count and trainable parameter count, and do not modify GPT-2 architecture. Validate with `python3 -m py_compile src/model/gpt2.py` and a sample  check that loads the model and prints parameter counts.
9. Implement logger: create logging and a helper to append training/evaluation metrics as JSON for the output directory
10. Implement `train.py`: load config, dataset, tokenizer/dataloader, GPT-2 model, di a dry-run (steps given in SPEC.md) exits without optimizer step or checkpoint save. Validate with `python3 -m py_compile train.py` and `.venv/bin/python train.py --config config/default.yaml --dry-run`.
11. Extend train.py from dry-run to real training: add AdamW, gradient accumulation, validation loss evaluation, checkpoint saving at 'save_steps', and final test loss reporting; keep dry-run. Validate with compile train.py
12. Implement checkpoint.py: add helpers to save and load GPT-2 checkpoints containing model state, optimizer state, model state, tokenizer files, epoch data and validation loss data.
13. Use checkpoint.py in train.py for periodic saving, include tokenizer saving, and keep dry-run mode from saving anything.
14. Implement generate.py: create the manual token-by-token generation loop, model forward pass, temperature, top-k, top-p, softmax, sample/argmax, and a per-step probability logging for reflection and for visualization. 
15. Implement sample.py: load config, tokenizer, GPT-2 model or checkpoint, call generate.py, add CLI arguments, print generated text, and print top-N per-step probabilities. Validate with the prompt `Harry looked at`.
16. Implement eval.pt: load config, test split, tokenizer/dataloader, base GPT-2, compute average test cross-entropy loss, and generate fixed qualitative samples for before/after comparison. 
17. Implement distributions.py: use the recorded per-step probabilities generated to save top-N token probability bar charts, highlight the sampled token, include generation step number
18. Implement embeddings.py: load pretrained GPT-2, select the most frequent Harry Potter training-corpus tokens, reduce embeddings to 2D with UMAP/PCA fallback, and save the plot to Deliverables/Plots/
19. Implement `app.py`: build a minimal Streamlit app that loads GPT-2 and a checkpoint, accepts a prompt, exposes sampling controls (`max_new_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, greedy vs sampling), generates text using `src/utils/generate.py`, and displays per-step top-N probability charts if available side by side for GPT2 Gnerated and checkpoint generated. do not spend time polishing UI. Run this in background, as it is independent of pipeline
20. Create a Kaggle-ready notebook file `kaggle_train.ipynb` that runs the existing project: install requirements, run full GPU training, zip checkpoints, and print the checkpoint zip path. Do not modify README, SPEC, CLAUDE, or any markdown file for this task.
21. I Unzipped the downloaded Kaggle checkpoint into ./checkpoints/, find the actual checkpoint-epoch1-step450 directory, then run sample.py twice with the same prompt "Harry looked at": first base GPT-2 without --checkpoint, then fine-tuned GPT-2 with --checkpoint <checkpoint-dir>. Print both outputs and per-step top probabilities for comparison. Do not run full eval.py test loss because it is slow.
22. Create `Deliverables/usage.txt` with evidence from the dry run we just ran, Kaggle GPU training, checkpoint creation, and local base-vs-fine-tuned inference comparison. Include the dry-run shapes/loss, Kaggle loss progression, checkpoint path, and difference where fine-tuned GPT-2 generated Harry Potter terms like Hagrid while base GPT-2 stayed generic.
23. Run the Streamlit app locally with `streamlit run app.py`, load the fine-tuned checkpoint from checkpoint-epoch1-step450, generate from the same prompt `"Harry looked at"` for both base and fine-tuned GPT-2, and confirm the app shows side-by-side outputs plus per-step probability charts. Do not modify code unless the app fails to load.
24. Create the post-trained embedding plot. Use the fine-tuned checkpoint at `checkpoints/checkpoint-epoch1-step450`, plot the most frequent Harry Potter tokens using the checkpoint's embedding matrix, and save it as `Deliverables/Plots/embeddings_posttrained.png`.