import os
import matplotlib.pyplot as plt
import numpy as np
import config


def set_plot_style():
    """Applies modern publication-quality styling."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_training_history(histories):
    """Plots training/validation loss and accuracy curves for all architectures."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    colors = {"MLP": "#1f77b4", "VanillaRNN": "#ff7f0e", "LSTM": "#2ca02c", "GRU": "#d62728"}

    # Plot 1: Validation Loss
    for model_name, hist in histories.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        c = colors.get(model_name, "#333333")
        axes[0].plot(epochs, hist["val_loss"], label=f"{model_name}", color=c, linewidth=2)
    
    axes[0].set_title("Validation Loss Progression")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Cross-Entropy)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Validation Joint Accuracy
    for model_name, hist in histories.items():
        epochs = range(1, len(hist["val_joint_acc"]) + 1)
        c = colors.get(model_name, "#333333")
        axes[1].plot(epochs, hist["val_joint_acc"], label=f"{model_name}", color=c, linewidth=2)

    axes[1].set_title("Validation Joint Event Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_model_comparison_bar(test_results, histories):
    """Bar chart comparing Test Loss, Joint Accuracy, Training Time, and Latency."""
    set_plot_style()
    models = list(test_results.keys())
    losses = [test_results[m]["test_loss"] for m in models]
    accuracies = [test_results[m]["joint_acc"] for m in models]
    train_times = [histories[m]["train_time_sec"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(models)]

    axes[0].bar(models, losses, color=colors, width=0.5)
    axes[0].set_title("Test Loss (Lower is Better)")
    axes[0].set_ylabel("Cross-Entropy Loss")

    axes[1].bar(models, accuracies, color=colors, width=0.5)
    axes[1].set_title("Joint Event Accuracy (Higher is Better)")
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(models, train_times, color=colors, width=0.5)
    axes[2].set_title("Training Duration (Seconds)")
    axes[2].set_ylabel("Time (s)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5, axis='y')

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "model_comparison_bar.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved model comparison bar chart to {save_path}")


def plot_muspy_metrics_comparison(muspy_summary_dict):
    """Bar chart comparing MusPy metrics across Original Bach vs Neural Models."""
    set_plot_style()
    categories = list(muspy_summary_dict.keys())  # Bach_GroundTruth, MLP, RNN, LSTM, GRU, Ablation
    pc_entropies = [muspy_summary_dict[c]["pitch_class_entropy"] for c in categories]
    consistencies = [muspy_summary_dict[c]["scale_consistency"] * 100.0 for c in categories]
    p_entropies = [muspy_summary_dict[c]["pitch_entropy"] for c in categories]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"][:len(categories)]

    axes[0].bar(categories, pc_entropies, color=colors, width=0.5)
    axes[0].set_title("MusPy Pitch Class Entropy")
    axes[0].set_ylabel("Entropy (Bits)")

    axes[1].bar(categories, consistencies, color=colors, width=0.5)
    axes[1].set_title("MusPy Scale Consistency (%)")
    axes[1].set_ylabel("Percentage (%)")

    axes[2].bar(categories, p_entropies, color=colors, width=0.5)
    axes[2].set_title("MusPy Pitch Entropy")
    axes[2].set_ylabel("Entropy (Bits)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=15)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "muspy_metrics_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved MusPy metrics comparison plot to {save_path}")


def plot_piano_rolls(ground_truth_p, continuations_dict, pitch_vocab):
    """Renders piano roll visualizations comparing ground truth chorale with continuations."""
    set_plot_style()
    models = list(continuations_dict.keys())
    fig, axes = plt.subplots(len(models) + 1, 1, figsize=(12, 3 * (len(models) + 1)), sharex=True, dpi=300)

    gt_pitches = []
    for row in ground_truth_p:
        for p_id in row:
            tok = pitch_vocab.decode(int(p_id))
            if tok not in [config.PAD_TOKEN, config.REST_TOKEN]:
                try:
                    gt_pitches.append(int(tok))
                except ValueError:
                    pass

    axes[0].scatter(range(len(gt_pitches)), gt_pitches, c='#2ca02c', s=15, alpha=0.8)
    axes[0].set_title("Ground Truth Bach Chorale")
    axes[0].set_ylabel("MIDI Pitch")

    for idx, (m_name, p_mat) in enumerate(continuations_dict.items(), start=1):
        m_pitches = []
        for row in p_mat:
            for p_id in row:
                tok = pitch_vocab.decode(int(p_id))
                if tok not in [config.PAD_TOKEN, config.REST_TOKEN]:
                    try:
                        m_pitches.append(int(tok))
                    except ValueError:
                        pass
        axes[idx].scatter(range(len(m_pitches)), m_pitches, c='#1f77b4', s=15, alpha=0.8)
        axes[idx].set_title(f"Continuation: {m_name}")
        axes[idx].set_ylabel("MIDI Pitch")

    axes[-1].set_xlabel("Event Step")
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "piano_roll_continuations.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved piano roll comparison plot to {save_path}")
