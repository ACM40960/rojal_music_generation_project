import os
import numpy as np
import torch
from music21 import stream, note, metadata

import config
import dataset


def _sample_with_temperature(logits, temperature=0.9, top_k=5):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, top_k)
        min_topk = v[-1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_continuation(model, seed_pitch_idx, seed_dur_idx, n_steps, temperature=0.9, device=config.DEVICE):
    """seed_pitch_idx / seed_dur_idx: (window_size, 4) numpy arrays. Returns generated (n_steps, 4) arrays."""
    model.eval()
    window_pitch = seed_pitch_idx.copy()
    window_dur = seed_dur_idx.copy()
    gen_pitch, gen_dur = [], []

    for _ in range(n_steps):
        xp = torch.from_numpy(window_pitch).unsqueeze(0).to(device)
        xd = torch.from_numpy(window_dur).unsqueeze(0).to(device)
        _, _, sampled_pitch, sampled_dur = model(xp, xd, y_pitch=None, y_dur=None, temperature=temperature)

        next_pitch = sampled_pitch[0].cpu().numpy()
        next_dur = sampled_dur[0].cpu().numpy()

        gen_pitch.append(next_pitch)
        gen_dur.append(next_dur)
        window_pitch = np.vstack([window_pitch[1:], next_pitch])
        window_dur = np.vstack([window_dur[1:], next_dur])

    return np.array(gen_pitch), np.array(gen_dur)



def cut_half_generate(model, piece, pitch_vocab, dur_vocab, temperature=0.9):
    """Split a piece at 50%, prompt on first half, generate the rest."""
    pitch_idx, dur_idx = dataset.encode_piece(piece, pitch_vocab, dur_vocab)
    T = pitch_idx.shape[0]
    mid = T // 2
    if mid < config.WINDOW_SIZE:
        raise ValueError(f"piece too short for window size {config.WINDOW_SIZE}")

    seed_pitch = pitch_idx[mid - config.WINDOW_SIZE:mid]
    seed_dur = dur_idx[mid - config.WINDOW_SIZE:mid]
    n_steps = T - mid

    gen_pitch_idx, gen_dur_idx = generate_continuation(model, seed_pitch, seed_dur, n_steps, temperature)
    actual_pitch_idx = pitch_idx[mid:]
    actual_dur_idx = dur_idx[mid:]

    return {
        "generated_pitch_idx": gen_pitch_idx,
        "generated_dur_idx": gen_dur_idx,
        "actual_pitch_idx": actual_pitch_idx,
        "actual_dur_idx": actual_dur_idx,
        "prompt_pitch_idx": pitch_idx[:mid],
        "prompt_dur_idx": dur_idx[:mid],
    }


def continuation_note_accuracy(result):
    pitch_match = (result["generated_pitch_idx"] == result["actual_pitch_idx"]).mean()
    dur_match = (result["generated_dur_idx"] == result["actual_dur_idx"]).mean()
    return {"pitch_match": float(pitch_match), "duration_match": float(dur_match)}


def _idx_matrix_to_stream(pitch_idx_matrix, dur_idx_matrix, pitch_vocab, dur_vocab, voice):
    s = stream.Part(id=config.VOICE_NAMES[voice])
    for t in range(pitch_idx_matrix.shape[0]):
        pitch_token = pitch_vocab.decode(int(pitch_idx_matrix[t, voice]))
        dur_val = dur_vocab.decode(int(dur_idx_matrix[t, voice]))
        if dur_val == config.PAD_TOKEN:
            continue
        if pitch_token in (config.REST_TOKEN, config.PAD_TOKEN):
            s.append(note.Rest(quarterLength=dur_val))
        else:
            s.append(note.Note(int(pitch_token), quarterLength=dur_val))
    return s


def write_continuation_midi(result, pitch_vocab, dur_vocab, out_path, title="chorale continuation"):
    """Writes prompt + generated continuation as one score, one part per voice."""
    full_pitch = np.vstack([result["prompt_pitch_idx"], result["generated_pitch_idx"]])
    full_dur = np.vstack([result["prompt_dur_idx"], result["generated_dur_idx"]])

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    for v in range(config.NUM_VOICES):
        score.append(_idx_matrix_to_stream(full_pitch, full_dur, pitch_vocab, dur_vocab, v))
    score.write("midi", fp=out_path)
    return out_path
