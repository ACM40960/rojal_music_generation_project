import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from music21 import corpus

import config


def _voice_events(part):
    events = []
    for n in part.flatten().notesAndRests:
        pitch = n.pitch.midi if n.isNote else None
        events.append({"offset": round(float(n.offset), 6), "pitch": pitch, "dur": float(n.quarterLength)})
    events.sort(key=lambda e: e["offset"])
    return events


def _align_to_grid(events, grid):
    pitches, durs = [], []
    cursor = 0
    last = len(events) - 1
    for g in grid:
        while cursor < last and events[cursor + 1]["offset"] <= g + 1e-6:
            cursor += 1
        pitches.append(events[cursor]["pitch"])
        durs.append(events[cursor]["dur"])
    return pitches, durs


def load_chorale_matrices():
    """Parse the full music21 chorale corpus into per-piece (pitch, duration) T x 4 matrices."""
    pieces = []
    for score in corpus.chorales.Iterator():
        if len(score.parts) != 4:
            continue
        names = tuple(p.partName for p in score.parts)
        if names != ("Soprano", "Alto", "Tenor", "Bass"):
            continue

        voice_events = [_voice_events(p) for p in score.parts]
        grid = sorted({round(e["offset"], 6) for events in voice_events for e in events})
        if len(grid) < config.WINDOW_SIZE + 1:
            continue

        pitch_cols, dur_cols = [], []
        for events in voice_events:
            p_seq, d_seq = _align_to_grid(events, grid)
            pitch_cols.append(p_seq)
            dur_cols.append(d_seq)

        pitch_matrix = np.array(pitch_cols, dtype=object).T  # T x 4
        dur_matrix = np.array(dur_cols, dtype=object).T
        pieces.append({
            "id": score.metadata.title or f"chorale_{len(pieces)}",
            "pitch": pitch_matrix,
            "duration": dur_matrix,
        })
    return pieces


class Vocab:
    def __init__(self, tokens):
        self.itos = [config.PAD_TOKEN] + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, token):
        return self.stoi[token]

    def decode(self, idx):
        return self.itos[idx]

    def __len__(self):
        return len(self.itos)


def build_vocabs(pieces):
    pitches = set()
    durations = set()
    for piece in pieces:
        for row in piece["pitch"]:
            for p in row:
                pitches.add(config.REST_TOKEN if p is None else p)
        for row in piece["duration"]:
            for d in row:
                durations.add(round(d, 6))

    pitch_tokens = [config.REST_TOKEN] + sorted(t for t in pitches if t != config.REST_TOKEN)
    duration_tokens = sorted(durations)
    return Vocab(pitch_tokens), Vocab(duration_tokens)


def encode_piece(piece, pitch_vocab, dur_vocab):
    T = piece["pitch"].shape[0]
    pitch_idx = np.zeros((T, config.NUM_VOICES), dtype=np.int64)
    dur_idx = np.zeros((T, config.NUM_VOICES), dtype=np.int64)
    for t in range(T):
        for v in range(config.NUM_VOICES):
            p = piece["pitch"][t, v]
            token = config.REST_TOKEN if p is None else p
            pitch_idx[t, v] = pitch_vocab.encode(token)
            dur_idx[t, v] = dur_vocab.encode(round(piece["duration"][t, v], 6))
    return pitch_idx, dur_idx


def make_windows(pitch_idx, dur_idx, window_size):
    X_pitch, X_dur, y_pitch, y_dur = [], [], [], []
    T = pitch_idx.shape[0]
    for i in range(T - window_size):
        X_pitch.append(pitch_idx[i:i + window_size])
        X_dur.append(dur_idx[i:i + window_size])
        y_pitch.append(pitch_idx[i + window_size])
        y_dur.append(dur_idx[i + window_size])
    return X_pitch, X_dur, y_pitch, y_dur


def split_pieces(pieces, seed=config.SEED):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pieces))
    n_train = int(len(pieces) * config.TRAIN_FRAC)
    n_val = int(len(pieces) * config.VAL_FRAC)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return (
        [pieces[i] for i in train_idx],
        [pieces[i] for i in val_idx],
        [pieces[i] for i in test_idx],
    )


class ChoraleWindowDataset(Dataset):
    def __init__(self, piece_subset, pitch_vocab, dur_vocab, window_size=config.WINDOW_SIZE):
        self.X_pitch, self.X_dur, self.y_pitch, self.y_dur = [], [], [], []
        for piece in piece_subset:
            pitch_idx, dur_idx = encode_piece(piece, pitch_vocab, dur_vocab)
            xp, xd, yp, yd = make_windows(pitch_idx, dur_idx, window_size)
            self.X_pitch.extend(xp)
            self.X_dur.extend(xd)
            self.y_pitch.extend(yp)
            self.y_dur.extend(yd)

    def __len__(self):
        return len(self.X_pitch)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_pitch[idx]),
            torch.from_numpy(self.X_dur[idx]),
            torch.from_numpy(self.y_pitch[idx]),
            torch.from_numpy(self.y_dur[idx]),
        )


def prepare_data(cache_path=os.path.join(config.ROOT_DIR, "outputs", "data_cache.pkl")):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    pieces = load_chorale_matrices()
    train_pieces, val_pieces, test_pieces = split_pieces(pieces)
    pitch_vocab, dur_vocab = build_vocabs(train_pieces)  # vocab built from training data only

    # fold in any pitch/duration values that only appear in val/test as unseen-token safety net
    for extra in (val_pieces, test_pieces):
        for piece in extra:
            for row in piece["pitch"]:
                for p in row:
                    token = config.REST_TOKEN if p is None else p
                    if token not in pitch_vocab.stoi:
                        pitch_vocab.stoi[token] = len(pitch_vocab.itos)
                        pitch_vocab.itos.append(token)
            for row in piece["duration"]:
                for d in row:
                    d = round(d, 6)
                    if d not in dur_vocab.stoi:
                        dur_vocab.stoi[d] = len(dur_vocab.itos)
                        dur_vocab.itos.append(d)

    bundle = {
        "pieces": pieces,
        "train_pieces": train_pieces,
        "val_pieces": val_pieces,
        "test_pieces": test_pieces,
        "pitch_vocab": pitch_vocab,
        "dur_vocab": dur_vocab,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


if __name__ == "__main__":
    bundle = prepare_data()
    print("total pieces:", len(bundle["pieces"]))
    print("train/val/test:", len(bundle["train_pieces"]), len(bundle["val_pieces"]), len(bundle["test_pieces"]))
    print("pitch vocab size:", len(bundle["pitch_vocab"]))
    print("duration vocab size:", len(bundle["dur_vocab"]))
