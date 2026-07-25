import os
import tempfile
import muspy
import numpy as np

import config
import generate


def _matrix_to_midi(pitch_idx, dur_idx, pitch_vocab, dur_vocab, tmp_path):
    score_result = {
        "prompt_pitch_idx": np.zeros((0, config.NUM_VOICES), dtype=np.int64),
        "prompt_dur_idx": np.zeros((0, config.NUM_VOICES), dtype=np.int64),
        "generated_pitch_idx": pitch_idx,
        "generated_dur_idx": dur_idx,
    }
    generate.write_continuation_midi(score_result, pitch_vocab, dur_vocab, tmp_path)
    return tmp_path


def muspy_metrics_for_matrix(pitch_idx, dur_idx, pitch_vocab, dur_vocab):
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        _matrix_to_midi(pitch_idx, dur_idx, pitch_vocab, dur_vocab, tmp_path)
        music = muspy.read_midi(tmp_path)
        if len(music.tracks) == 0 or all(len(t.notes) == 0 for t in music.tracks):
            return {"pitch_class_entropy": None, "scale_consistency": None, "empty_beat_rate": None}
        return {
            "pitch_class_entropy": float(muspy.pitch_class_entropy(music)),
            "scale_consistency": float(muspy.scale_consistency(music)),
            "empty_beat_rate": float(muspy.empty_beat_rate(music)),
        }
    finally:
        os.remove(tmp_path)


def evaluate_generation_musically(result, pitch_vocab, dur_vocab):
    """Compares MusPy metrics between the real continuation and the generated one for a
    single cut-half trial."""
    generated_metrics = muspy_metrics_for_matrix(
        result["generated_pitch_idx"], result["generated_dur_idx"], pitch_vocab, dur_vocab
    )
    real_metrics = muspy_metrics_for_matrix(
        result["actual_pitch_idx"], result["actual_dur_idx"], pitch_vocab, dur_vocab
    )
    return {"generated": generated_metrics, "real_bach": real_metrics}


def average_metrics(metric_dicts):
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts if d[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out
