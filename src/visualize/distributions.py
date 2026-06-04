from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.utils.generate import StepRecord


def plot_step(rec: StepRecord, output_path: str) -> None:
    tokens = rec.top_tokens
    probs = rec.top_probs
    colors = ["#e74c3c" if t == rec.token_str else "#3498db" for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(tokens[::-1], probs[::-1], color=colors[::-1])
    ax.set_xlabel("Probability")
    ax.set_title(f"Step {rec.step} — sampled: {repr(rec.token_str)}")
    ax.set_xlim(0, max(probs) * 1.15)

    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{prob:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def save_step_charts(steps: list[StepRecord], output_dir: str) -> list[str]:
    paths = []
    for rec in steps:
        path = str(Path(output_dir) / f"step_{rec.step:03d}.png")
        plot_step(rec, path)
        paths.append(path)
    return paths


def save_gif(image_paths: list[str], gif_path: str, fps: int = 2) -> None:
    frames = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(gif_path, frames, fps=fps)
