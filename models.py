import torch
import torch.nn as nn

import config


class FrameEmbedding(nn.Module):
    """Embeds pitch+duration per voice, concatenates across the 4 voices into one
    synchronized SATB frame vector per timestep."""

    def __init__(self, pitch_vocab_size, dur_vocab_size, embed_dim=config.EMBED_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.pitch_embeds = nn.ModuleList(
            [nn.Embedding(pitch_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.dur_embeds = nn.ModuleList(
            [nn.Embedding(dur_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.frame_dim = config.NUM_VOICES * 2 * embed_dim
        self.norm = nn.LayerNorm(self.frame_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pitch_idx, dur_idx):
        # pitch_idx, dur_idx: (batch, seq_len, 4)
        voice_frames = []
        for v in range(config.NUM_VOICES):
            p_emb = self.pitch_embeds[v](pitch_idx[..., v])
            d_emb = self.dur_embeds[v](dur_idx[..., v])
            voice_frames.append(torch.cat([p_emb, d_emb], dim=-1))
        concat = torch.cat(voice_frames, dim=-1)  # (batch, seq_len, frame_dim)
        return self.dropout(self.norm(concat))


class ChainedOutputHeads(nn.Module):
    """Predicts S->A->T->B sequentially within one timestep: each voice's pitch head
    sees the hidden state + already-decided voices' pitch/duration embeddings, and
    each duration head additionally sees its own voice's just-decided pitch."""

    def __init__(self, in_dim, pitch_vocab_size, dur_vocab_size, embed_dim=config.EMBED_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.pitch_embeds = nn.ModuleList(
            [nn.Embedding(pitch_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.dur_embeds = nn.ModuleList(
            [nn.Embedding(dur_vocab_size, embed_dim, padding_idx=0) for _ in range(config.NUM_VOICES)]
        )
        self.pitch_heads = nn.ModuleList()
        self.dur_heads = nn.ModuleList()
        cond_dim = 0
        hidden_mid = in_dim // 2
        for _ in range(config.NUM_VOICES):
            self.pitch_heads.append(
                nn.Sequential(
                    nn.Linear(in_dim + cond_dim, hidden_mid),
                    nn.LayerNorm(hidden_mid),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_mid, pitch_vocab_size)
                )
            )
            cond_dim += embed_dim
            self.dur_heads.append(
                nn.Sequential(
                    nn.Linear(in_dim + cond_dim, hidden_mid),
                    nn.LayerNorm(hidden_mid),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_mid, dur_vocab_size)
                )
            )
            cond_dim += embed_dim

    def forward(self, x, y_pitch=None, y_dur=None, temperature=0.9, top_k=5):
        """x: (batch, in_dim). y_pitch/y_dur: (batch, 4) for teacher forcing (training).
        If None, samples autoregressively (generation)."""
        cond, pitch_logits, dur_logits, pitch_out, dur_out = [], [], [], [], []
        for v in range(config.NUM_VOICES):
            h = torch.cat([x] + cond, dim=-1) if cond else x
            p_logits = self.pitch_heads[v](h)
            pitch_logits.append(p_logits)
            p_idx = y_pitch[:, v] if y_pitch is not None else self._sample(p_logits, temperature, top_k)
            pitch_out.append(p_idx)
            cond.append(self.pitch_embeds[v](p_idx))

            h = torch.cat([x] + cond, dim=-1)
            d_logits = self.dur_heads[v](h)
            dur_logits.append(d_logits)
            d_idx = y_dur[:, v] if y_dur is not None else self._sample(d_logits, temperature, top_k)
            dur_out.append(d_idx)
            cond.append(self.dur_embeds[v](d_idx))

        return pitch_logits, dur_logits, torch.stack(pitch_out, -1), torch.stack(dur_out, -1)

    @staticmethod
    def _sample(logits, temperature=0.9, top_k=5):
        if temperature <= 0:
            return logits.argmax(-1)
        logits = logits / temperature
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            min_topk = v[:, -1:]
            logits = torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


class ChoraleMLP(nn.Module):
    def __init__(self, pitch_vocab_size, dur_vocab_size, window_size=config.WINDOW_SIZE,
                 embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.frame_embed = FrameEmbedding(pitch_vocab_size, dur_vocab_size, embed_dim, dropout)
        flat_dim = window_size * self.frame_embed.frame_dim
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.heads = ChainedOutputHeads(hidden_dim, pitch_vocab_size, dur_vocab_size, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pitch_idx, dur_idx, y_pitch=None, y_dur=None, temperature=0.9, top_k=5):
        frames = self.frame_embed(pitch_idx, dur_idx)
        flat = frames.reshape(frames.size(0), -1)
        h1 = self.drop1(self.act1(self.norm1(self.fc1(flat))))
        h2 = self.drop2(self.act2(self.norm2(self.fc2(h1)))) + h1  # Residual connection
        return self.heads(h2, y_pitch=y_pitch, y_dur=y_dur, temperature=temperature, top_k=top_k)


class _RecurrentBase(nn.Module):
    rnn_cls = None

    def __init__(self, pitch_vocab_size, dur_vocab_size, embed_dim=config.EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.frame_embed = FrameEmbedding(pitch_vocab_size, dur_vocab_size, embed_dim, dropout)
        self.rnn = self.rnn_cls(
            input_size=self.frame_embed.frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = ChainedOutputHeads(hidden_dim, pitch_vocab_size, dur_vocab_size, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pitch_idx, dur_idx, y_pitch=None, y_dur=None, temperature=0.9, top_k=5):
        frames = self.frame_embed(pitch_idx, dur_idx)
        out, _ = self.rnn(frames)
        last = self.dropout(self.norm(out[:, -1, :]))
        return self.heads(last, y_pitch=y_pitch, y_dur=y_dur, temperature=temperature, top_k=top_k)



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

