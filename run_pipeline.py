import os
import json
import argparse

import config
import dataset
import train
import generate
import music_metrics
import visualize


def run(n_cut_half_pieces=15, temperature=0.9):
    config.set_seed()

    print("=== preparing data ===")
    bundle = dataset.prepare_data()
    print(f"pieces: {len(bundle['pieces'])} "
          f"(train {len(bundle['train_pieces'])} / val {len(bundle['val_pieces'])} / test {len(bundle['test_pieces'])})")
    print(f"pitch vocab: {len(bundle['pitch_vocab'])}, duration vocab: {len(bundle['dur_vocab'])}")

    trained_models = {}
    test_results = {}

    for name in config.MODEL_NAMES:
        print(f"\n=== training {name} ===")
        model, history = train.train_model(name, bundle)
        trained_models[name] = model
        test_results[name] = train.evaluate_test_set(model, bundle)
        print(f"{name} test results: {test_results[name]}")

    with open(os.path.join(config.OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n=== cut-half generation + musical evaluation ===")
    eval_pieces = [p for p in bundle["test_pieces"] if p["pitch"].shape[0] >= 2 * config.WINDOW_SIZE][:n_cut_half_pieces]
    muspy_results = {}

    for name, model in trained_models.items():
        gen_metrics_list, real_metrics_list, note_accs = [], [], []
        for i, piece in enumerate(eval_pieces):
            result = generate.cut_half_generate(model, piece, bundle["pitch_vocab"], bundle["dur_vocab"], temperature)
            note_accs.append(generate.continuation_note_accuracy(result))
            musical = music_metrics.evaluate_generation_musically(result, bundle["pitch_vocab"], bundle["dur_vocab"])
            gen_metrics_list.append(musical["generated"])
            real_metrics_list.append(musical["real_bach"])

            if i == 0:
                midi_path = os.path.join(config.MIDI_DIR, f"{name}_continuation_example.mid")
                generate.write_continuation_midi(result, bundle["pitch_vocab"], bundle["dur_vocab"], midi_path,
                                                  title=f"{name.upper()} continuation of {piece['id']}")
                visualize.plot_piano_roll(
                    result["generated_pitch_idx"], result["generated_dur_idx"],
                    bundle["pitch_vocab"], bundle["dur_vocab"],
                    f"{name.upper()} generated continuation — {piece['id']}",
                    os.path.join(config.PLOTS_DIR, f"{name}_piano_roll.png"),
                )

        muspy_results[name] = {
            "generated": music_metrics.average_metrics(gen_metrics_list),
            "real_bach": music_metrics.average_metrics(real_metrics_list),
        }
        avg_pitch_match = sum(a["pitch_match"] for a in note_accs) / len(note_accs)
        avg_dur_match = sum(a["duration_match"] for a in note_accs) / len(note_accs)
        muspy_results[name]["continuation_note_accuracy"] = {
            "pitch_match": avg_pitch_match, "duration_match": avg_dur_match
        }
        print(f"{name} cut-half musical eval: {muspy_results[name]}")

    with open(os.path.join(config.OUTPUT_DIR, "musical_eval_results.json"), "w") as f:
        json.dump(muspy_results, f, indent=2)

    print("\n=== plots ===")
    histories = visualize.load_histories()
    visualize.plot_loss_curves(histories)
    visualize.plot_accuracy_curves(histories)
    visualize.plot_model_comparison(test_results)
    visualize.plot_muspy_comparison(muspy_results)

    print("\ndone. see outputs/ for checkpoints, midi, plots, and result jsons.")
    return test_results, muspy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-half-pieces", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()
    run(n_cut_half_pieces=args.cut_half_pieces, temperature=args.temperature)
