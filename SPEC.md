# GPT-2 Post-Training Specification

> **Status:** Minimal implementation spec  
> **Dataset:** `WutYee/HarryPotter_books_1to7`  
> **Dataset URL:** https://huggingface.co/datasets/WutYee/HarryPotter_books_1to7  
> **Target model:** GPT-2 small (`gpt2`)  
> **Goal:** Fine-tune GPT-2 small on the Harry Potter books dataset, expose flexible sampling controls, and visualize both token embeddings and per-step generation probability distributions.

---

## 1. Project Layout

```text
gpt2-post-training-ssachindeep/
├── SPEC.md
├── README.md
├── AGENTS.md
├── requirements.txt
├── config/
│   └── default.yaml
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── model/
│   │   └── gpt2.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   └── generate.py
│   └── visualize/
│       ├── embeddings.py
│       └── distributions.py
├── train.py
├── eval.py
├── sample.py
├── app.py
└── Deliverables/
    ├── SPEC.md
    ├── AGENTS.md
    ├── usage.txt
    ├── Reflection.md
    └── Plots/
```

---

## 2. Configuration

All runtime settings live in `config/default.yaml`. CLI overrides use argparse dot notation.

---

## 3. Data Pipeline

Load the dataset with:

```python
load_dataset("WutYee/HarryPotter_books_1to7")
```

Use only the `text` column. Preserve the native splits:

- `train`
- `validation`
- `test`

Clean text lightly:

- strip leading and trailing whitespace
- normalize Unicode to NFC
- collapse repeated blank lines
- remove control characters except newline and tab

The test split must not be used during training or validation.

---

## 4. Tokenizer And Dataset

Use the original GPT-2 tokenizer:

```python
GPT2TokenizerFast.from_pretrained("gpt2")
```

Do not train or replace the tokenizer. GPT-2 pretrained embeddings depend on the original GPT-2 token IDs.

Create fixed-length causal language modeling chunks:

- `input_ids`: token IDs
- `attention_mask`: real-token vs padding mask
- `labels`: same values as `input_ids`
- padding positions in `labels`: `-100`

---

## 5. Model And Training

Use:

```python
GPT2LMHeadModel.from_pretrained("gpt2")
```

Training objective:

```text
predict token_t from token_<t
```

Use cross-entropy loss over non-padding positions.

Training must support:

- AdamW
- gradient calculation
- validation loss tracking
- checkpoint saving

---

## 6. Sampling And Inference

Implement sampling manually in `src/utils/generate.py` so every generation step can expose probabilities.

Expose these controls:

- prompt
- max new tokens
- temperature
- top-k
- top-p
- greedy vs sampling
- seed

At each generation step:

1. run model forward pass
2. get next-token logits
3. apply temperature
4. apply top-k
5. apply top-p
6. apply repetition penalty
7. convert logits to probabilities
8. sample or choose argmax
9. record top-N probabilities for visualization

---

## 7. Visualization

### 7.1 Token Embeddings

Visualize `model.transformer.wte.weight`.

Create plots in `./Deliverables/Plots/` for:

- pretrained GPT-2 embeddings
- fine-tuned GPT-2 embeddings
- Plot most frequent tokens in harry potter corpus

### 7.2 Per-Step Generation Probabilities

During inference, save probability chart for each of the generated token.

Each chart should show:

- top-N candidate tokens
- probability of each candidate
- sampled token highlighted
- generation step number

---

## 8. Evaluation

Evaluate on the test split after training.

Report:

- average cross-entropy loss
- qualitative generated samples from fixed prompts

Run the same prompts two times, on before and one after fine-tuning for comparison.

---

## 9. Local App

Build a local inference app in `app.py`.

The app must show:

- prompt input
- sampling controls
- generated text
- per-step probability visualization

The app loads the downloaded fine-tuned checkpoint and also the retrained model data for side by side comparison.

---

## 10. Kaggle Workflow

Local machine:

1. write all Python files locally
2. run dry run locally
3. upload code to Kaggle

Dry run:
1. Able to download the data
2. load the training set
3. Load the BPE tokenizer (same as gpt2 so token id are not having mismatches)
4. Tokenize one batch of training set split
5. Embed the token with standard embedding matrix
6. Load gpt2 model
7. run across the 12 MMHA transformer blocks
8. Store the logits
9. Using softmax convert the logits to probabiliities
10. calculate the cross entropy loss, print the loss. 
If we are able to do all that, it means our pipeline is working fine and can be taken to Kaggle 

Kaggle notebook:

1. install requirements
2. run `train.py`
3. save checkpoint
4. zip checkpoint
5. download checkpoint locally
6. save the checkpoint in ./checkpoints

Local machine after Kaggle:

1. run `eval.py`
2. run `sample.py`
3. run `app.py`
4. generate final plots

---

## 11. Dependencies

```text
torch
transformers
datasets
tokenizers
pyyaml
tqdm
numpy
matplotlib
scikit-learn
umap-learn
pillow
imageio
streamlit
```

---

## 12. Deliverables

Create `Deliverables/` containing:

- updated `SPEC.md`
- Codex export
- `AGENTS.md`
- Codex status output
- `Plots/`
- `Reflection.md` under 300 words (do not write this, I will write myself)