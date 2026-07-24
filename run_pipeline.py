import os
import pandas as pd
import numpy as np
import torch
import config
from dataset import get_dataloaders
from models import MLPBaseline, VanillaRNN, LSTMMusic, GRUMusic, PitchOnlyBaseline
from train import train_model, evaluate_test_set, set_seed
from generate import generate_chorale_continuation, matrix_to_music21_score, save_continuation_midi
from music_metrics import compute_muspy_metrics_from_midi
from visualize import (plot_training_history, plot_model_comparison_bar,
                       plot_muspy_metrics_comparison, plot_piano_rolls)


def main():
    print("=" * 70)
    print("M.Tech Thesis Pipeline: Symbolic Music Completion Using Bach Chorales")
    print("Evaluation: MusPy Metrics Suite & 5-Piece Cut-in-Half Validation")
    print(f"Device: {config.DEVICE} | Seed: {config.SEED}")
    print("=" * 70)

    set_seed(config.SEED)

    # 1. Load Data & Create Splits / Vocabs / Loaders
    train_loader, val_loader, test_loader, pitch_vocab, dur_vocab, raw_splits = get_dataloaders()
    test_chorales = raw_splits["test"]
    num_test_pieces = min(5, len(test_chorales))
    test_samples = test_chorales[:num_test_pieces]

    num_pitches = len(pitch_vocab)
    num_durations = len(dur_vocab)

    # 2. Compute Original Bach Ground Truth MusPy Baseline across the 5 test pieces
    bach_gt_metrics_list = []
    for idx, piece in enumerate(test_samples):
        gt_enc_p = [[pitch_vocab.encode(p) for p in row] for row in piece["pitch_matrix"]]
        gt_enc_d = [[dur_vocab.encode(d) for d in row] for row in piece["dur_matrix"]]
        gt_score = matrix_to_music21_score(gt_enc_p, gt_enc_d, pitch_vocab, dur_vocab)
        midi_path = save_continuation_midi(gt_score, f"Piece_{idx+1}_Original_GroundTruth.mid")
        m_res = compute_muspy_metrics_from_midi(midi_path)
        bach_gt_metrics_list.append(m_res)

    bach_gt_avg = {
        "pitch_class_entropy": float(np.mean([m["pitch_class_entropy"] for m in bach_gt_metrics_list])),
        "scale_consistency": float(np.mean([m["scale_consistency"] for m in bach_gt_metrics_list])),
        "pitch_entropy": float(np.mean([m["pitch_entropy"] for m in bach_gt_metrics_list])),
        "empty_beat_rate": float(np.mean([m["empty_beat_rate"] for m in bach_gt_metrics_list]))
    }

    # 3. Model Definitions
    models_to_train = {
        "MLP": MLPBaseline(num_pitches, num_durations),
        "VanillaRNN": VanillaRNN(num_pitches, num_durations),
        "LSTM": LSTMMusic(num_pitches, num_durations),
        "GRU": GRUMusic(num_pitches, num_durations),
        "PitchOnlyAblation": PitchOnlyBaseline(num_pitches)
    }

    histories = {}
    test_results = {}
    muspy_metrics_summary = {"Bach_GroundTruth": bach_gt_avg}
    sample_continuations_matrices = {}

    # 4. Unified Training & Multi-Piece Cut-in-Half Validation Loop
    for model_name, model in models_to_train.items():
        print(f"\n>>> Training Model: {model_name} <<<")
        history = train_model(model, train_loader, val_loader, model_name=model_name, epochs=config.EPOCHS)
        histories[model_name] = history

        # Load Best Model for Evaluation
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint_path))

        # Evaluate on Unseen Test Set
        test_res = evaluate_test_set(model, test_loader, model_name=model_name)
        test_res["train_time_sec"] = history["train_time_sec"]
        test_res["avg_inf_ms"] = history["avg_inference_time_ms"]
        test_results[model_name] = test_res

        # 5-Piece Cut-in-Half Continuation Validation
        model_muspy_list = []
        for idx, piece in enumerate(test_samples):
            enc_p, enc_d, prompt_len = generate_chorale_continuation(
                model, piece, pitch_vocab, dur_vocab
            )

            # Store first piece continuation for piano roll visualization
            if idx == 0:
                sample_continuations_matrices[model_name] = enc_p

            score = matrix_to_music21_score(enc_p, enc_d, pitch_vocab, dur_vocab)
            midi_file = f"Piece_{idx+1}_{model_name}_continuation.mid"
            midi_path = save_continuation_midi(score, midi_file)

            m_res = compute_muspy_metrics_from_midi(midi_path)
            model_muspy_list.append(m_res)

        avg_muspy = {
            "pitch_class_entropy": float(np.mean([m["pitch_class_entropy"] for m in model_muspy_list])),
            "scale_consistency": float(np.mean([m["scale_consistency"] for m in model_muspy_list])),
            "pitch_entropy": float(np.mean([m["pitch_entropy"] for m in model_muspy_list])),
            "empty_beat_rate": float(np.mean([m["empty_beat_rate"] for m in model_muspy_list]))
        }
        muspy_metrics_summary[model_name] = avg_muspy

    # 5. Save Summary Tables (CSV & Markdown)
    df_perf = pd.DataFrame.from_dict(test_results, orient='index')
    perf_csv = os.path.join(config.METRICS_DIR, "test_performance_summary.csv")
    df_perf.to_csv(perf_csv)

    muspy_rows = []
    for name, metrics in muspy_metrics_summary.items():
        muspy_rows.append({
            "Model": name,
            "Pitch Class Entropy": f"{metrics['pitch_class_entropy']:.4f}",
            "Scale Consistency (%)": f"{metrics['scale_consistency'] * 100.0:.2f}%",
            "Pitch Entropy": f"{metrics['pitch_entropy']:.4f}",
            "Empty Beat Rate (%)": f"{metrics['empty_beat_rate'] * 100.0:.2f}%"
        })

    df_muspy = pd.DataFrame(muspy_rows)
    muspy_csv = os.path.join(config.METRICS_DIR, "muspy_metrics_summary.csv")
    df_muspy.to_csv(muspy_csv, index=False)

    print("\n" + "=" * 70)
    print("THESIS RESULTS SUMMARY: MACHINE LEARNING METRICS")
    print("=" * 70)
    print(df_perf[["test_loss", "pitch_acc", "dur_acc", "joint_acc", "train_time_sec", "avg_inf_ms"]].to_string())

    print("\n" + "=" * 70)
    print("THESIS RESULTS SUMMARY: MUSPY MUSIC METRICS (5-PIECE AVERAGE)")
    print("=" * 70)
    print(df_muspy.to_string(index=False))

    # 6. Generate Visualizations & Publication Plots
    plot_training_history({k: v for k, v in histories.items() if k != "PitchOnlyAblation"})
    plot_model_comparison_bar(test_results, histories)
    plot_muspy_metrics_comparison(muspy_metrics_summary)
    
    first_piece_gt_p = [[pitch_vocab.encode(p) for p in row] for row in test_samples[0]["pitch_matrix"]]
    plot_piano_rolls(first_piece_gt_p, sample_continuations_matrices, pitch_vocab)

    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {config.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
