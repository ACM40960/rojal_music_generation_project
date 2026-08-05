import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})

import config

MODEL_COLORS = {"mlp": "#8899aa", "rnn": "#e07a5f", "lstm": "#3d5a80", "gru": "#81b29a"}
VOICE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Soprano (Blue), Alto (Orange), Tenor (Green), Bass (Red)


def load_histories(model_names=config.MODEL_NAMES):
    histories = {}
    for name in model_names:
        path = os.path.join(config.OUTPUT_DIR, f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)
    return histories


def plot_loss_curves(histories, out_path=os.path.join(config.PLOTS_DIR, "loss_curves.png")):
    names = list(histories.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        hist = histories[name]
        ax.plot(hist["train_loss"], label="train", color="#3d5a80", linewidth=1.8)
        ax.plot(hist["val_loss"], label="val", color="#e07a5f", linewidth=1.8)
        best_epoch = int(np.argmin(hist["val_loss"]))
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.scatter([best_epoch], [hist["val_loss"][best_epoch]], color="#e07a5f", zorder=5, s=35)
        ax.set_title(f"Model: {name.upper()}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray", fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle=":")

    axes[0].set_ylabel("Loss (Summed Cross-Entropy)")
    fig.suptitle("Training vs Validation Loss (Dashed Line = Best Checkpoint)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_val_loss_comparison(histories, out_path=os.path.join(config.PLOTS_DIR, "val_loss_comparison.png")):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    best_losses = []
    for name, hist in histories.items():
        ax.plot(hist["val_loss"], label=name.upper(), color=MODEL_COLORS.get(name), linewidth=2.0)
        best_losses.append(min(hist["val_loss"]))
    ax.set_ylim(min(best_losses) * 0.95, min(best_losses) * 1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison Across Models (Clipped)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray", loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_curves(histories, out_path=os.path.join(config.PLOTS_DIR, "accuracy_curves.png")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, hist in histories.items():
        color = MODEL_COLORS.get(name)
        axes[0].plot(hist["val_pitch_acc"], label=name.upper(), color=color, linewidth=2.0)
        axes[1].plot(hist["val_dur_acc"], label=name.upper(), color=color, linewidth=2.0)
    axes[0].set_title("Validation Pitch Accuracy", fontweight="bold")
    axes[1].set_title("Validation Duration Accuracy", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray", loc="lower right")
        ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(test_results, out_path=os.path.join(config.PLOTS_DIR, "model_comparison.png")):
    names = list(test_results.keys())
    pitch_acc = [test_results[n]["test_pitch_acc"] for n in names]
    dur_acc = [test_results[n]["test_dur_acc"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, pitch_acc, width, label="Pitch Accuracy", color="#3d5a80")
    ax.bar(x + width / 2, dur_acc, width, label="Duration Accuracy", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], fontweight="bold")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Set Accuracy", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray", loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_muspy_comparison(muspy_results, out_path=os.path.join(config.PLOTS_DIR, "muspy_comparison.png")):
    metrics = ["pitch_class_entropy", "scale_consistency", "empty_beat_rate"]
    names = list(muspy_results.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    for i, metric in enumerate(metrics):
        gen_vals = [muspy_results[n]["generated"][metric] for n in names]
        real_vals = [muspy_results[n]["real_bach"][metric] for n in names]
        x = np.arange(len(names))
        width = 0.35
        axes[i].bar(x - width / 2, gen_vals, width, label="Generated", color="#e07a5f")
        axes[i].bar(x + width / 2, real_vals, width, label="Real Bach", color="#3d5a80")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([n.upper() for n in names], fontweight="bold")
        axes[i].set_title(metric.replace("_", " ").title(), fontweight="bold")
        axes[i].grid(True, alpha=0.3, linestyle=":")
        if i == 0:
            axes[i].legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray", loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_piano_roll(ax, pitch_idx, dur_idx, pitch_vocab, dur_vocab, prompt_len=None):
    """Plots SATB piano roll line segments per voice across synchronized timesteps t."""
    for v in range(config.NUM_VOICES):
        for t in range(pitch_idx.shape[0]):
            p_tok = pitch_vocab.decode(int(pitch_idx[t, v]))
            if p_tok not in (config.REST_TOKEN, config.PAD_TOKEN):
                ax.plot([t, t + 0.9], [int(p_tok), int(p_tok)], color=VOICE_COLORS[v],
                        linewidth=2.2, label=config.VOICE_NAMES[v] if t == 0 else None)
    
    if prompt_len is not None:
        ax.axvline(x=prompt_len - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.85)

    ax.set_ylabel("MIDI Pitch", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")


def plot_multi_panel_piano_roll_comparison(
    model_results, prompt_pitch_idx, prompt_dur_idx, actual_pitch_idx, actual_dur_idx,
    pitch_vocab, dur_vocab, out_path=os.path.join(config.PLOTS_DIR, "multi_panel_piano_roll.png")
):
    """
    Generates a 5-panel stacked piano roll comparison matching the demo image format:
    1. Ground Truth Chorale (Prompt + Ground Truth Continuation)
    2. Model Continuation: MLP
    3. Model Continuation: VANILLA_RNN
    4. Model Continuation: LSTM
    5. Model Continuation: GRU
    """
    model_order = ["mlp", "rnn", "lstm", "gru"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True, sharey=True)

    prompt_len = prompt_pitch_idx.shape[0]

    # Panel 1: Ground Truth
    gt_pitch = np.vstack([prompt_pitch_idx, actual_pitch_idx])
    gt_dur = np.vstack([prompt_dur_idx, actual_dur_idx])
    _draw_piano_roll(axes[0], gt_pitch, gt_dur, pitch_vocab, dur_vocab, prompt_len=prompt_len)
    axes[0].set_title("Ground Truth Chorale (Prompt + Ground Truth Continuation)", fontsize=10, fontweight="bold", pad=4)

    # Panels 2-5: Models
    model_titles = {
        "mlp": "Model Continuation: MLP",
        "rnn": "Model Continuation: VANILLA_RNN",
        "lstm": "Model Continuation: LSTM",
        "gru": "Model Continuation: GRU",
    }

    for idx, name in enumerate(model_order):
        ax = axes[idx + 1]
        gen_p = model_results[name]["generated_pitch_idx"]
        gen_d = model_results[name]["generated_dur_idx"]
        full_p = np.vstack([prompt_pitch_idx, gen_p])
        full_d = np.vstack([prompt_dur_idx, gen_d])

        _draw_piano_roll(ax, full_p, full_d, pitch_vocab, dur_vocab, prompt_len=prompt_len)
        ax.set_title(model_titles[name], fontsize=10, fontweight="bold", pad=4)

def plot_piano_roll_single_comparison(
    prompt_pitch_idx, prompt_dur_idx, generated_pitch_idx, generated_dur_idx,
    actual_pitch_idx, actual_dur_idx, pitch_vocab, dur_vocab, title, out_path
):
    """
    Generates a 2-panel stacked piano roll comparison:
    1. Ground Truth Chorale (Prompt + Ground Truth Continuation)
    2. Model Continuation (Prompt + Generated Continuation)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, sharey=True)
    prompt_len = prompt_pitch_idx.shape[0]

    # Panel 1: Ground Truth
    gt_pitch = np.vstack([prompt_pitch_idx, actual_pitch_idx])
    gt_dur = np.vstack([prompt_dur_idx, actual_dur_idx])
    _draw_piano_roll(axes[0], gt_pitch, gt_dur, pitch_vocab, dur_vocab, prompt_len=prompt_len)
    axes[0].set_title("Ground Truth Chorale (Prompt + Ground Truth Continuation)", fontsize=10, fontweight="bold", pad=4)

    # Panel 2: Predicted Continuation
    pred_pitch = np.vstack([prompt_pitch_idx, generated_pitch_idx])
    pred_dur = np.vstack([prompt_dur_idx, generated_dur_idx])
    _draw_piano_roll(axes[1], pred_pitch, pred_dur, pitch_vocab, dur_vocab, prompt_len=prompt_len)
    axes[1].set_title(f"Model Continuation: {title}", fontsize=10, fontweight="bold", pad=4)

    axes[-1].set_xlabel("Synchronized Event Timestep (t)", fontsize=10, fontweight="bold")
    fig.suptitle(f"Piano Roll Comparison — {title}", fontsize=12, fontweight="bold", x=0.01, ha="left", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path