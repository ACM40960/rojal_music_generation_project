import torch
import torch.nn as nn
import config


def init_weights(m):
    """Xavier / Kaiming Initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BaseMusicModel(nn.Module):
    """Base class with dual embedding layers and ModuleList output heads for SATB voices."""
    def __init__(self, num_pitches, num_durations,
                 pitch_embed_dim=config.PITCH_EMBED_DIM,
                 dur_embed_dim=config.DURATION_EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.num_pitches = num_pitches
        self.num_durations = num_durations
        
        self.pitch_embed = nn.Embedding(num_pitches, pitch_embed_dim, padding_idx=0)
        self.dur_embed = nn.Embedding(num_durations, dur_embed_dim, padding_idx=0)
        
        self.frame_input_dim = 4 * (pitch_embed_dim + dur_embed_dim)
        
        # Clean ModuleList Head Organization as requested
        self.pitch_heads = nn.ModuleList([nn.Linear(hidden_dim, num_pitches) for _ in range(4)])
        self.duration_heads = nn.ModuleList([nn.Linear(hidden_dim, num_durations) for _ in range(4)])

    def embed_frames(self, pitch_in, dur_in):
        """
        pitch_in: (B, W, 4)
        dur_in: (B, W, 4)
        Returns: (B, W, 4 * (pitch_dim + dur_dim))
        """
        B, W, _ = pitch_in.shape
        p_emb = self.pitch_embed(pitch_in)  # (B, W, 4, pitch_dim)
        d_emb = self.dur_embed(dur_in)      # (B, W, 4, dur_dim)
        
        frame_emb = torch.cat([p_emb, d_emb], dim=-1)  # (B, W, 4, pitch_dim + dur_dim)
        frame_emb = frame_emb.view(B, W, -1)           # (B, W, 4 * (pitch_dim + dur_dim))
        return frame_emb

    def compute_heads(self, hidden_features):
        """
        hidden_features: (B, hidden_dim)
        Returns:
            pitch_logits: List of 4 tensors, each of shape (B, num_pitches)
            dur_logits: List of 4 tensors, each of shape (B, num_durations)
        """
        pitch_logits = [head(hidden_features) for head in self.pitch_heads]
        dur_logits = [head(hidden_features) for head in self.duration_heads]
        return pitch_logits, dur_logits


class MLPBaseline(BaseMusicModel):
    """Linear Neural Network Baseline."""
    def __init__(self, num_pitches, num_durations, window_length=config.WINDOW_LENGTH, **kwargs):
        super().__init__(num_pitches, num_durations, **kwargs)
        flattened_dim = window_length * self.frame_input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        self.apply(init_weights)

    def forward(self, pitch_in, dur_in):
        frame_emb = self.embed_frames(pitch_in, dur_in)  # (B, W, frame_input_dim)
        flattened = frame_emb.reshape(frame_emb.size(0), -1)  # (B, W * frame_input_dim)
        hidden = self.mlp(flattened)  # (B, hidden_dim)
        return self.compute_heads(hidden)


class VanillaRNN(BaseMusicModel):
    """2-Layer Vanilla RNN Model."""
    def __init__(self, num_pitches, num_durations, **kwargs):
        super().__init__(num_pitches, num_durations, **kwargs)
        self.rnn = nn.RNN(
            input_size=self.frame_input_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0
        )
        self.norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.apply(init_weights)

    def forward(self, pitch_in, dur_in):
        frame_emb = self.embed_frames(pitch_in, dur_in)  # (B, W, frame_input_dim)
        out, _ = self.rnn(frame_emb)  # (B, W, hidden_dim)
        last_out = self.dropout(self.norm(out[:, -1, :]))  # (B, hidden_dim)
        return self.compute_heads(last_out)


class LSTMMusic(BaseMusicModel):
    """2-Layer LSTM Model."""
    def __init__(self, num_pitches, num_durations, **kwargs):
        super().__init__(num_pitches, num_durations, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.frame_input_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0
        )
        self.norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.apply(init_weights)

    def forward(self, pitch_in, dur_in):
        frame_emb = self.embed_frames(pitch_in, dur_in)  # (B, W, frame_input_dim)
        out, _ = self.lstm(frame_emb)  # (B, W, hidden_dim)
        last_out = self.dropout(self.norm(out[:, -1, :]))  # (B, hidden_dim)
        return self.compute_heads(last_out)


class GRUMusic(BaseMusicModel):
    """2-Layer GRU Model."""
    def __init__(self, num_pitches, num_durations, **kwargs):
        super().__init__(num_pitches, num_durations, **kwargs)
        self.gru = nn.GRU(
            input_size=self.frame_input_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0
        )
        self.norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.apply(init_weights)

    def forward(self, pitch_in, dur_in):
        frame_emb = self.embed_frames(pitch_in, dur_in)  # (B, W, frame_input_dim)
        out, _ = self.gru(frame_emb)  # (B, W, hidden_dim)
        last_out = self.dropout(self.norm(out[:, -1, :]))  # (B, hidden_dim)
        return self.compute_heads(last_out)


class PitchOnlyBaseline(nn.Module):
    """Ablation Model: LSTM without Duration Embeddings."""
    def __init__(self, num_pitches, pitch_embed_dim=config.PITCH_EMBED_DIM, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.num_pitches = num_pitches
        self.pitch_embed = nn.Embedding(num_pitches, pitch_embed_dim, padding_idx=0)
        self.frame_input_dim = 4 * pitch_embed_dim
        
        self.lstm = nn.LSTM(
            input_size=self.frame_input_dim,
            hidden_size=hidden_dim,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.pitch_heads = nn.ModuleList([nn.Linear(hidden_dim, num_pitches) for _ in range(4)])
        self.apply(init_weights)

    def forward(self, pitch_in, dur_in=None):
        B, W, _ = pitch_in.shape
        p_emb = self.pitch_embed(pitch_in).view(B, W, -1)  # (B, W, 4 * pitch_embed_dim)
        out, _ = self.lstm(p_emb)
        last_out = self.dropout(self.norm(out[:, -1, :]))
        pitch_logits = [head(last_out) for head in self.pitch_heads]
        # Return dummy duration logits matching dummy duration targets
        return pitch_logits, None


if __name__ == "__main__":
    B, W = 16, 32
    p_in = torch.randint(0, 40, (B, W, 4))
    d_in = torch.randint(0, 8, (B, W, 4))

    for ModelClass in [MLPBaseline, VanillaRNN, LSTMMusic, GRUMusic]:
        model = ModelClass(num_pitches=42, num_durations=10)
        p_logits, d_logits = model(p_in, d_in)
        print(f"{ModelClass.__name__} -> Pitch Logits shape: 4 x {p_logits[0].shape}, Dur Logits shape: 4 x {d_logits[0].shape}")

    ablation_model = PitchOnlyBaseline(num_pitches=42)
    p_logits, _ = ablation_model(p_in)
    print(f"PitchOnlyBaseline -> Pitch Logits shape: 4 x {p_logits[0].shape}")
