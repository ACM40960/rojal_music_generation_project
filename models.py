import torch
import torch.nn as nn

import config


class FrameEmbedding(nn.Module):
    """Embeds pitch+duration per voice, concatenates across the 4 voices into one
    synchronized SATB frame vector per timestep."""

    def __init__(self, pitch_vocab_size, dur_vocab_size, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.pitch_embeds = nn.ModuleList(
            [nn.Embedding(pitch_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.dur_embeds = nn.ModuleList(
            [nn.Embedding(dur_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.frame_dim = config.NUM_VOICES * 2 * embed_dim

    def forward(self, pitch_idx, dur_idx):
        # pitch_idx, dur_idx: (batch, seq_len, 4)
        voice_frames = []
        for v in range(config.NUM_VOICES):
            p_emb = self.pitch_embeds[v](pitch_idx[..., v])
            d_emb = self.dur_embeds[v](dur_idx[..., v])
            voice_frames.append(torch.cat([p_emb, d_emb], dim=-1))
        return torch.cat(voice_frames, dim=-1)  # (batch, seq_len, frame_dim)


class OutputHeads(nn.Module):
    def __init__(self, in_dim, pitch_vocab_size, dur_vocab_size):
        super().__init__()
        self.pitch_heads = nn.ModuleList([nn.Linear(in_dim, pitch_vocab_size) for _ in range(config.NUM_VOICES)])
        self.duration_heads = nn.ModuleList([nn.Linear(in_dim, dur_vocab_size) for _ in range(config.NUM_VOICES)])

    def forward(self, x):
        pitch_logits = [head(x) for head in self.pitch_heads]
        dur_logits = [head(x) for head in self.duration_heads]
        return pitch_logits, dur_logits


class ChoraleMLP(nn.Module):
    def __init__(self, pitch_vocab_size, dur_vocab_size, window_size=config.WINDOW_SIZE,
                 embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.frame_embed = FrameEmbedding(pitch_vocab_size, dur_vocab_size, embed_dim)
        flat_dim = window_size * self.frame_embed.frame_dim
        self.fc = nn.Linear(flat_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.heads = OutputHeads(hidden_dim, pitch_vocab_size, dur_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pitch_idx, dur_idx):
        frames = self.frame_embed(pitch_idx, dur_idx)
        flat = frames.reshape(frames.size(0), -1)
        x = self.act(self.norm(self.fc(flat)))
        return self.heads(x)


class _RecurrentBase(nn.Module):
    rnn_cls = None

    def __init__(self, pitch_vocab_size, dur_vocab_size, embed_dim=config.EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.frame_embed = FrameEmbedding(pitch_vocab_size, dur_vocab_size, embed_dim)
        self.rnn = self.rnn_cls(
            input_size=self.frame_embed.frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = OutputHeads(hidden_dim, pitch_vocab_size, dur_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in (self.heads,):
            for lin in m.modules():
                if isinstance(lin, nn.Linear):
                    nn.init.xavier_uniform_(lin.weight)
                    nn.init.zeros_(lin.bias)

    def forward(self, pitch_idx, dur_idx):
        frames = self.frame_embed(pitch_idx, dur_idx)
        out, _ = self.rnn(frames)
        last = self.dropout(self.norm(out[:, -1, :]))
        return self.heads(last)


class ChoraleRNN(_RecurrentBase):
    rnn_cls = nn.RNN


class ChoraleLSTM(_RecurrentBase):
    rnn_cls = nn.LSTM


class ChoraleGRU(_RecurrentBase):
    rnn_cls = nn.GRU


MODEL_REGISTRY = {
    "mlp": ChoraleMLP,
    "rnn": ChoraleRNN,
    "lstm": ChoraleLSTM,
    "gru": ChoraleGRU,
}


def build_model(name, pitch_vocab_size, dur_vocab_size):
    return MODEL_REGISTRY[name](pitch_vocab_size, dur_vocab_size)
