import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import music21
import config


class Vocabulary:
    """Mapping between tokens (pitches or durations) and integer IDs."""
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = [config.PAD_TOKEN, config.REST_TOKEN]
        self.token2id = {}
        self.id2token = {}
        for token in special_tokens:
            self.add_token(token)

    def add_token(self, token):
        str_token = str(token)
        if str_token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[str_token] = idx
            self.id2token[idx] = str_token
            return idx
        return self.token2id[str_token]

    def encode(self, token):
        str_token = str(token)
        return self.token2id.get(str_token, self.token2id[config.PAD_TOKEN])

    def decode(self, idx):
        return self.id2token.get(idx, config.PAD_TOKEN)

    def __len__(self):
        return len(self.token2id)


def extract_chorale_matrices(score):
    """
    Extracts synchronized SATB pitch (T x 4) and duration (T x 4) matrices from a music21 score.
    Preserves voice identity by sorting parts by average pitch (Soprano, Alto, Tenor, Bass).
    """
    parts = list(score.parts)
    if len(parts) < 4:
        return None, None

    # Sort parts by average pitch to reliably identify S, A, T, B
    part_info = []
    for p in parts:
        elements = p.recurse().getElementsByClass(['Note', 'Rest', 'Chord'])
        pitches = []
        for n in elements:
            if isinstance(n, music21.note.Note):
                pitches.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                pitches.append(max(pt.midi for pt in n.pitches))
        avg_pitch = sum(pitches) / len(pitches) if pitches else 0
        part_info.append((avg_pitch, p))

    part_info.sort(key=lambda x: x[0], reverse=True)
    satb_parts = [p[1] for p in part_info[:4]]

    part_events = []
    all_offsets = set()
    for p in satb_parts:
        events = {}
        elements = p.recurse().getElementsByClass(['Note', 'Rest', 'Chord'])
        for element in elements:
            off = float(element.getOffsetInHierarchy(p))
            dur = float(element.quarterLength)
            if isinstance(element, music21.note.Note):
                pitch = element.pitch.midi
            elif isinstance(element, music21.chord.Chord):
                pitch = max(pt.midi for pt in element.pitches)
            else:
                pitch = config.REST_TOKEN
            events[off] = (pitch, dur)
            all_offsets.add(off)
        part_events.append(events)

    sorted_offsets = sorted(list(all_offsets))
    if len(sorted_offsets) < config.WINDOW_LENGTH + 1:
        return None, None

    pitch_matrix = []
    dur_matrix = []
    last_pitch = [config.REST_TOKEN] * 4
    last_dur = [1.0] * 4

    for off in sorted_offsets:
        p_row = []
        d_row = []
        for v in range(4):
            if off in part_events[v]:
                p, d = part_events[v][off]
                last_pitch[v] = p
                last_dur[v] = d
            else:
                p = last_pitch[v]
                d = last_dur[v]
            p_row.append(p)
            d_row.append(d)
        pitch_matrix.append(p_row)
        dur_matrix.append(d_row)

    return pitch_matrix, dur_matrix


def load_and_preprocess_corpus(max_chorales=None):
    """Parses Bach chorales from music21 corpus into SATB matrices."""
    print("Loading Bach Chorale Corpus from music21...")
    chorale_files = [f for f in music21.corpus.getComposer('bach') if f.name.endswith('.xml') or f.name.endswith('.mxl') or f.name.endswith('.krn')]
    
    if max_chorales:
        chorale_files = chorale_files[:max_chorales]

    dataset = []
    for idx, f in enumerate(chorale_files):
        try:
            score = music21.corpus.parse(f)
            p_mat, d_mat = extract_chorale_matrices(score)
            if p_mat is not None and d_mat is not None:
                dataset.append({
                    "id": idx,
                    "filename": f.name,
                    "pitch_matrix": p_mat,
                    "dur_matrix": d_mat
                })
        except Exception as e:
            continue

    print(f"Successfully processed {len(dataset)} Bach chorales.")
    return dataset


def build_vocabularies(train_chorales):
    """Builds Pitch and Duration Vocabularies from training set."""
    pitch_vocab = Vocabulary(special_tokens=[config.PAD_TOKEN, config.REST_TOKEN])
    dur_vocab = Vocabulary(special_tokens=[config.PAD_TOKEN])

    for chorale in train_chorales:
        for p_row, d_row in zip(chorale["pitch_matrix"], chorale["dur_matrix"]):
            for p in p_row:
                pitch_vocab.add_token(p)
            for d in d_row:
                dur_vocab.add_token(d)

    print(f"Pitch Vocab Size: {len(pitch_vocab)} | Duration Vocab Size: {len(dur_vocab)}")
    return pitch_vocab, dur_vocab


class BachChoraleDataset(Dataset):
    """PyTorch Dataset generating sliding windows of SATB event matrices."""
    def __init__(self, chorales, pitch_vocab, dur_vocab, window_length=config.WINDOW_LENGTH):
        self.samples = []
        for chorale in chorales:
            p_mat = chorale["pitch_matrix"]
            d_mat = chorale["dur_matrix"]
            T = len(p_mat)
            if T <= window_length:
                continue

            # Encode tokens
            enc_p = [[pitch_vocab.encode(p) for p in row] for row in p_mat]
            enc_d = [[dur_vocab.encode(d) for d in row] for row in d_mat]

            for i in range(T - window_length):
                p_in = enc_p[i : i + window_length]
                d_in = enc_d[i : i + window_length]
                p_target = enc_p[i + window_length]
                d_target = enc_d[i + window_length]

                self.samples.append({
                    "pitch_in": torch.tensor(p_in, dtype=torch.long),
                    "dur_in": torch.tensor(d_in, dtype=torch.long),
                    "pitch_target": torch.tensor(p_target, dtype=torch.long),
                    "dur_target": torch.tensor(d_target, dtype=torch.long)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloaders(max_chorales=None, batch_size=config.BATCH_SIZE, window_length=config.WINDOW_LENGTH):
    """Parses corpus, splits by chorale ID, builds vocabs, and returns PyTorch DataLoaders."""
    chorales = load_and_preprocess_corpus(max_chorales=max_chorales)

    # Set seed for reproducible split
    random.seed(config.SEED)
    random.shuffle(chorales)

    num_chorales = len(chorales)
    n_train = int(num_chorales * config.TRAIN_RATIO)
    n_val = int(num_chorales * config.VAL_RATIO)

    train_chorales = chorales[:n_train]
    val_chorales = chorales[n_train : n_train + n_val]
    test_chorales = chorales[n_train + n_val:]

    print(f"Data Split by Chorale: Train={len(train_chorales)}, Val={len(val_chorales)}, Test={len(test_chorales)}")

    pitch_vocab, dur_vocab = build_vocabularies(train_chorales)

    train_ds = BachChoraleDataset(train_chorales, pitch_vocab, dur_vocab, window_length)
    val_ds = BachChoraleDataset(val_chorales, pitch_vocab, dur_vocab, window_length)
    test_ds = BachChoraleDataset(test_chorales, pitch_vocab, dur_vocab, window_length)

    print(f"Sliding Window Samples: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    raw_splits = {
        "train": train_chorales,
        "val": val_chorales,
        "test": test_chorales
    }

    return train_loader, val_loader, test_loader, pitch_vocab, dur_vocab, raw_splits


if __name__ == "__main__":
    tr, va, te, pv, dv, splits = get_dataloaders(max_chorales=10)
    sample = next(iter(tr))
    print("Sample pitch_in shape:", sample["pitch_in"].shape)
    print("Sample pitch_target shape:", sample["pitch_target"].shape)
