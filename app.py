import base64
from pathlib import Path

import torch
import streamlit as st
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.model.gpt2 import load_model
from src.data.dataset import get_tokenizer
from src.utils.checkpoint import load_checkpoint
from src.utils.generate import generate, StepRecord

CFG_PATH = "config/default.yaml"
DEFAULT_AUDIO_PATH = "assets/background.mp3"


@st.cache_resource
def load_base_model():
    cfg = load_config(CFG_PATH)
    tokenizer = get_tokenizer()
    model = load_model(cfg)
    model.eval()
    return cfg, tokenizer, model


@st.cache_resource
def load_finetuned_model(ckpt_path: str):
    cfg = load_config(CFG_PATH)
    model = load_model(cfg)
    load_checkpoint(ckpt_path, model)
    model.eval()
    return model


def prob_chart(rec: StepRecord):
    tokens = rec.top_tokens
    probs = rec.top_probs
    colors = ["#e74c3c" if t == rec.token_str else "#3498db" for t in tokens]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(tokens[::-1], probs[::-1], color=colors[::-1])
    ax.set_title(f"Step {rec.step} → {repr(rec.token_str)}", fontsize=9)
    ax.set_xlabel("Probability")
    plt.tight_layout()
    return fig


def run_generation(model, tokenizer, cfg, prompt, max_new_tokens, temperature, top_k, top_p, rep_penalty, greedy, seed):
    device = next(model.parameters()).device
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        do_sample=not greedy,
        seed=seed if seed >= 0 else None,
        top_n_record=cfg["visualization"]["distributions"]["top_n_tokens"],
        device=device,
    )


def render_looping_audio(audio_path: str):
    path = Path(audio_path)
    if not path.exists():
        st.sidebar.warning(f"Audio file not found: {audio_path}")
        return

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.sidebar.markdown(
        f"""
        <audio controls autoplay loop style="width: 100%;">
            <source src="data:audio/mpeg;base64,{encoded}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )


st.title("GPT-2 Harry Potter — Inference App")

cfg, tokenizer, base_model = load_base_model()

# Sidebar — checkpoint + sampling controls
st.sidebar.header("Model")
ckpt_path = st.sidebar.text_input("Fine-tuned checkpoint dir (optional)", value="")

st.sidebar.header("Background Music")
play_music = st.sidebar.checkbox("Play looping background music", value=False)
audio_path = st.sidebar.text_input("Audio file path", value=DEFAULT_AUDIO_PATH)
if play_music:
    render_looping_audio(audio_path)

st.sidebar.header("Sampling Controls")
prompt = st.sidebar.text_area("Prompt", value="Harry looked at")
max_new_tokens = st.sidebar.slider("Max new tokens", 10, 300, cfg["sampling"]["max_new_tokens"])
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, float(cfg["sampling"]["temperature"]), 0.05)
top_k = st.sidebar.slider("Top-k", 0, 200, int(cfg["sampling"]["top_k"]))
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, float(cfg["sampling"]["top_p"]), 0.01)
rep_penalty = st.sidebar.slider("Repetition penalty", 1.0, 2.0, float(cfg["sampling"]["repetition_penalty"]), 0.05)
greedy = st.sidebar.checkbox("Greedy decoding", value=False)
seed = st.sidebar.number_input("Seed (-1 = random)", value=42, step=1)
show_charts = st.sidebar.checkbox("Show per-step probability charts", value=True)
max_charts = st.sidebar.slider("Max steps to chart", 1, 30, 5)

if st.sidebar.button("Generate"):
    col_base, col_ft = st.columns(2)

    with col_base:
        st.subheader("Base GPT-2")
        with st.spinner("Generating..."):
            base_result = run_generation(base_model, tokenizer, cfg, prompt, max_new_tokens,
                                         temperature, top_k, top_p, rep_penalty, greedy, int(seed))
        st.markdown(f"**Prompt:** {prompt}")
        st.markdown(f"**Output:** {base_result.text}")

        if show_charts:
            st.markdown("**Per-step probabilities:**")
            for rec in base_result.steps[:max_charts]:
                st.pyplot(prob_chart(rec))
                plt.close("all")

    with col_ft:
        st.subheader("Fine-tuned GPT-2")
        if not ckpt_path.strip():
            st.info("No checkpoint provided. Enter a checkpoint directory in the sidebar.")
        else:
            try:
                ft_model = load_finetuned_model(ckpt_path.strip())
                with st.spinner("Generating..."):
                    ft_result = run_generation(ft_model, tokenizer, cfg, prompt, max_new_tokens,
                                               temperature, top_k, top_p, rep_penalty, greedy, int(seed))
                st.markdown(f"**Prompt:** {prompt}")
                st.markdown(f"**Output:** {ft_result.text}")

                if show_charts:
                    st.markdown("**Per-step probabilities:**")
                    for rec in ft_result.steps[:max_charts]:
                        st.pyplot(prob_chart(rec))
                        plt.close("all")
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
