import os
import json
import pickle
import argparse
import numpy as np
import torch

import config
import dataset
import models
import generate
import music_metrics
import visualize

# ==============================================================================
# DEMONSTRATION CHORALE SELECTION
# ==============================================================================
# To select a different unseen test chorale for demonstration:
#   Option 1: Change the integer value of CHORALE_INDEX below (e.g. 0, 1, 2...).
#   Option 2: Run via command line with the --index flag:
#             python demo_generate.py --index 3
#   Option 3: Run with --list to view all available unseen test set chorales:
#             python demo_generate.py --list
# ==============================================================================
CHORALE_INDEX = 0


def main(chorale_index=CHORALE_INDEX, temperature=0.9, list_chorales=False):
    config.set_seed()

    # 1. Load cached dataset & vocabularies
    data_cache_path = os.path.join(config.OUTPUT_DIR, "data_cache.pkl")
    if not os.path.exists(data_cache_path):
        raise FileNotFoundError(
            f"Cached dataset not found at '{data_cache_path}'. Please run dataset.py first."
        )

    with open(data_cache_path, "rb") as f:
        data = pickle.load(f)

    test_pieces = data["test_pieces"]
    pitch_vocab = data["pitch_vocab"]
    dur_vocab = data["dur_vocab"]

    if list_chorales:
        print("Available Unseen Test Set Chorales:")
        for idx, p in enumerate(test_pieces):
            T = p["pitch"].shape[0]
            print(f"  [{idx:2d}] {p['id']} (Total Steps: {T})")
        return

    if chorale_index < 0 or chorale_index >= len(test_pieces):
        raise IndexError(
            f"Invalid CHORALE_INDEX {chorale_index}. Must be between 0 and {len(test_pieces) - 1}."
        )

    # 2. Load trained LSTM checkpoint
    ckpt_path = os.path.join(config.CKPT_DIR, "lstm_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{ckpt_path}'. Please run training first."
        )

    model = models.build_model("lstm", len(pitch_vocab), len(dur_vocab)).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    # 3. Select target chorale
    piece = test_pieces[chorale_index]
    total_timesteps = piece["pitch"].shape[0]

    # 4. Perform cut-half autoregressive generation (50% prompt / 50% continuation)
    result = generate.cut_half_generate(model, piece, pitch_vocab, dur_vocab, temperature=temperature)

    prompt_length = result["prompt_pitch_idx"].shape[0]
    continuation_length = result["generated_pitch_idx"].shape[0]

    # 5. Prepare output directory
    demo_dir = os.path.join(config.OUTPUT_DIR, "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # 6. Generate required MIDI files
    # 1. original_full.mid (GT First Half + GT Second Half)
    gt_full_result = {
        "prompt_pitch_idx": result["prompt_pitch_idx"],
        "prompt_dur_idx": result["prompt_dur_idx"],
        "generated_pitch_idx": result["actual_pitch_idx"],
        "generated_dur_idx": result["actual_dur_idx"],
    }
    orig_mid_path = os.path.join(demo_dir, "original_full.mid")
    generate.write_continuation_midi(
        gt_full_result, pitch_vocab, dur_vocab, orig_mid_path, title=f"Real Bach — {piece['id']}"
    )

    # 2. predicted_full.mid (GT First Half + Generated Second Half)
    pred_mid_path = os.path.join(demo_dir, "predicted_full.mid")
    generate.write_continuation_midi(
        result, pitch_vocab, dur_vocab, pred_mid_path, title=f"LSTM Continuation — {piece['id']}"
    )

    # 3. ground_truth_second_half.mid (GT Second Half Only)
    empty_prompt = np.zeros((0, config.NUM_VOICES), dtype=np.int64)
    gt_half_result = {
        "prompt_pitch_idx": empty_prompt,
        "prompt_dur_idx": empty_prompt,
        "generated_pitch_idx": result["actual_pitch_idx"],
        "generated_dur_idx": result["actual_dur_idx"],
    }
    gt_half_path = os.path.join(demo_dir, "ground_truth_second_half.mid")
    generate.write_continuation_midi(
        gt_half_result, pitch_vocab, dur_vocab, gt_half_path, title=f"Real Bach 2nd Half — {piece['id']}"
    )

    # 4. predicted_second_half.mid (Generated Second Half Only)
    pred_half_result = {
        "prompt_pitch_idx": empty_prompt,
        "prompt_dur_idx": empty_prompt,
        "generated_pitch_idx": result["generated_pitch_idx"],
        "generated_dur_idx": result["generated_dur_idx"],
    }
    pred_half_path = os.path.join(demo_dir, "predicted_second_half.mid")
    generate.write_continuation_midi(
        pred_half_result, pitch_vocab, dur_vocab, pred_half_path, title=f"LSTM 2nd Half — {piece['id']}"
    )

    # 7. Generate piano roll comparison figure
    piano_roll_path = os.path.join(demo_dir, "piano_roll.png")
    visualize.plot_piano_roll_single_comparison(
        result["prompt_pitch_idx"],
        result["prompt_dur_idx"],
        result["generated_pitch_idx"],
        result["generated_dur_idx"],
        result["actual_pitch_idx"],
        result["actual_dur_idx"],
        pitch_vocab,
        dur_vocab,
        f"LSTM — {piece['id']}",
        piano_roll_path,
    )

    # 8. Compute official thesis evaluation metrics
    musical_eval = music_metrics.evaluate_generation_musically(result, pitch_vocab, dur_vocab)

    # Load test set overall results if available
    overall_test_results = {}
    test_res_path = os.path.join(config.OUTPUT_DIR, "test_results.json")
    if os.path.exists(test_res_path):
        with open(test_res_path) as f:
            overall_test_results = json.load(f)

    lstm_test_results = overall_test_results.get("lstm", {})

    metrics_payload = {
        "chorale_info": {
            "title": piece["id"],
            "total_timesteps": total_timesteps,
            "prompt_length": prompt_length,
            "continuation_length": continuation_length,
        },
        "overall_test_set_metrics": lstm_test_results,
        "muspy_metrics": musical_eval,
    }

    # Save metrics JSON
    metrics_json_path = os.path.join(demo_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save clean summary text file
    summary_txt_path = os.path.join(demo_dir, "summary.txt")
    summary_text = (
        f"======================================================================\n"
        f"DEMONSTRATION SUMMARY: {piece['id']}\n"
        f"======================================================================\n"
        f"Selected Chorale     : J.S. Bach — {piece['id']}\n"
        f"Total Timesteps      : {total_timesteps}\n"
        f"Prompt Length        : {prompt_length}\n"
        f"Continuation Length : {continuation_length}\n\n"
        f"----------------------------------------------------------------------\n"
        f"OVERALL TEST SET PERFORMANCE (LSTM MODEL)\n"
        f"----------------------------------------------------------------------\n"
        f"Test Loss            : {lstm_test_results.get('test_loss', 'N/A')}\n"
        f"Pitch Accuracy       : {lstm_test_results.get('test_pitch_acc', 0) * 100:.2f}%\n"
        f"Duration Accuracy    : {lstm_test_results.get('test_dur_acc', 0) * 100:.2f}%\n\n"
        f"----------------------------------------------------------------------\n"
        f"MUSPY EVALUATION (GENERATED VS GROUND TRUTH)\n"
        f"----------------------------------------------------------------------\n"
        f"Metric                   Generated (LSTM)     Real Bach (GT)\n"
        f"Pitch Class Entropy    : {musical_eval['generated']['pitch_class_entropy']:.4f}               {musical_eval['real_bach']['pitch_class_entropy']:.4f}\n"
        f"Scale Consistency      : {musical_eval['generated']['scale_consistency']:.4f}               {musical_eval['real_bach']['scale_consistency']:.4f}\n"
        f"Empty Beat Rate        : {musical_eval['generated']['empty_beat_rate']:.4f}               {musical_eval['real_bach']['empty_beat_rate']:.4f}\n"
        f"======================================================================\n"
    )

    with open(summary_txt_path, "w") as f:
        f.write(summary_text)

    # Print summary to console
    print(summary_text)
    print(f"Demonstration artifacts successfully generated in '{demo_dir}':")
    print(f"  - Original Full MIDI      : {orig_mid_path}")
    print(f"  - Predicted Full MIDI     : {pred_mid_path}")
    print(f"  - GT 2nd Half MIDI        : {gt_half_path}")
    print(f"  - Pred 2nd Half MIDI      : {pred_half_path}")
    print(f"  - Piano Roll Figure       : {piano_roll_path}")
    print(f"  - Metrics JSON            : {metrics_json_path}")
    print(f"  - Summary Text File       : {summary_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Thesis Demonstration Script")
    parser.add_argument(
        "--index",
        type=int,
        default=CHORALE_INDEX,
        help="Index of the unseen test chorale to demonstrate (default: 0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Autoregressive sampling temperature (default: 0.9)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available unseen test set chorales and exit",
    )
    args = parser.parse_args()
    main(chorale_index=args.index, temperature=args.temperature, list_chorales=args.list)
