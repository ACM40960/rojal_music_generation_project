import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import config

MODEL_COLORS = {"mlp": "#8899aa", "rnn": "#e07a5f", "lstm": "#3d5a80", "gru": "#81b29a"}


def load_histories(model_names=config.MODEL_NAMES):
    histories = {}
    for name in model_names:
        path = os.path.join(config.OUTPUT_DIR, f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)
    return histories


def plot_loss_curves(histories, out_path=os.path.join(config.PLOTS_DIR, "loss_curves.png")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, hist in histories.items():
        color = MODEL_COLORS.get(name)
        axes[0].plot(hist["train_loss"], label=name.upper(), color=color)
        axes[1].plot(hist["val_loss"], label=name.upper(), color=color)
    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (summed CE, 4 voices)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_accuracy_curves(histories, out_path=os.path.join(config.PLOTS_DIR, "accuracy_curves.png")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, hist in histories.items():
        color = MODEL_COLORS.get(name)
        axes[0].plot(hist["val_pitch_acc"], label=name.upper(), color=color)
        axes[1].plot(hist["val_dur_acc"], label=name.upper(), color=color)
    axes[0].set_title("Validation Pitch Accuracy")
    axes[1].set_title("Validation Duration Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_model_comparison(test_results, out_path=os.path.join(config.PLOTS_DIR, "model_comparison.png")):
    names = list(test_results.keys())
    pitch_acc = [test_results[n]["test_pitch_acc"] for n in names]
    dur_acc = [test_results[n]["test_dur_acc"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, pitch_acc, width, label="Pitch Accuracy", color="#3d5a80")
    ax.bar(x + width / 2, dur_acc, width, label="Duration Accuracy", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_muspy_comparison(muspy_results, out_path=os.path.join(config.PLOTS_DIR, "muspy_comparison.png")):
    """muspy_results: {model_name: {"generated": {...}, "real_bach": {...}}}"""
    metrics = ["pitch_class_entropy", "scale_consistency", "empty_beat_rate"]
    names = list(muspy_results.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for i, metric in enumerate(metrics):
        gen_vals = [muspy_results[n]["generated"][metric] for n in names]
        real_vals = [muspy_results[n]["real_bach"][metric] for n in names]
        x = np.arange(len(names))
        width = 0.35
        axes[i].bar(x - width / 2, gen_vals, width, label="Generated", color="#e07a5f")
        axes[i].bar(x + width / 2, real_vals, width, label="Real Bach", color="#3d5a80")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([n.upper() for n in names])
        axes[i].set_title(metric.replace("_", " ").title())
        if i == 0:
            axes[i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_piano_roll(pitch_idx, dur_idx, pitch_vocab, dur_vocab, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#3d5a80", "#81b29a", "#e07a5f", "#8899aa"]
    for v in range(config.NUM_VOICES):
        t_cursor = 0.0
        for t in range(pitch_idx.shape[0]):
            pitch_token = pitch_vocab.decode(int(pitch_idx[t, v]))
            dur_val = dur_vocab.decode(int(dur_idx[t, v]))
            if isinstance(dur_val, str):
                t_cursor += 0.5
                continue
            if pitch_token not in (config.REST_TOKEN, config.PAD_TOKEN):
                ax.barh(int(pitch_token), dur_val, left=t_cursor, height=0.8,
                        color=colors[v], alpha=0.8, label=config.VOICE_NAMES[v] if t == 0 else None)
            t_cursor += dur_val
    ax.set_xlabel("Time (quarter lengths)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
