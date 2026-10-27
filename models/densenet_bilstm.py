"""
models/densenet_bilstm.py

PyTorch DenseNet1D + BiLSTM + Attention classifier — the architecture in
the Diagnosis box of the pipeline diagram. `agents/diagnosis` originally
imported this module but it was never committed; this is that module.

The production weights live in the Keras checkpoint `DenseNet1D_clean.keras`
(use `models.backend.KerasBackend`). This port is used when a PyTorch
checkpoint is available and by the test suite / synthetic demo.

Input: (N, 1, 360). Output: (N, num_classes) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.ecg_state import NUM_CLASSES


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, growth_rate, kernel_size=7, padding=3, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        layers = [_DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        super().__init__(*layers)
        self.out_channels = in_channels + num_layers * growth_rate


class _Transition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(2),
        )


class TemporalAttention(nn.Module):
    """Additive attention over BiLSTM timesteps. Returns (context, weights)
    so a forward hook can pick the weights up (see agents/explainability)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, h):  # h: (N, T, H)
        weights = torch.softmax(torch.tanh(self.score(h)), dim=1)  # (N, T, 1)
        context = (h * weights).sum(dim=1)                         # (N, H)
        return context, weights.squeeze(-1)


class DenseNet1DBiLSTMAttention(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, growth_rate: int = 12,
                 block_layers: tuple[int, ...] = (3, 3), lstm_hidden: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        channels = 2 * growth_rate
        stages: list[nn.Module] = [nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3, bias=False)]
        for i, n_layers in enumerate(block_layers):
            block = _DenseBlock(channels, n_layers, growth_rate)
            stages.append(block)
            channels = block.out_channels
            if i != len(block_layers) - 1:
                stages.append(_Transition(channels, channels // 2))
                channels //= 2
        stages += [nn.BatchNorm1d(channels), nn.ReLU()]
        self.features = nn.Sequential(*stages)
        self.bilstm = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(2 * lstm_hidden)
        self.embedding = nn.Sequential(nn.Linear(2 * lstm_hidden, embedding_dim), nn.ReLU())
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _encode(self, x):
        h, _ = self.bilstm(self.features(x).transpose(1, 2))  # (N, T, 2H)
        return self.attention(h)

    def embed(self, x):
        context, _ = self._encode(x)
        return self.embedding(context)

    def attention_weights(self, x):
        _, weights = self._encode(x)
        return weights

    def forward(self, x):
        return self.classifier(self.embed(x))


def train_classifier(model: DenseNet1DBiLSTMAttention, signals, labels, epochs: int = 10,
                     lr: float = 3e-3, batch_size: int = 64, seed: int = 0) -> DenseNet1DBiLSTMAttention:
    """Minimal training loop for synthetic demos and tests. Real training
    happens in the team's Colab notebook. Pass an already-constructed model;
    use `build_and_train_classifier` for a fully seeded run."""
    import numpy as np

    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(signals, dtype=np.float32).reshape(len(signals), 1, -1))
    y = torch.tensor(np.asarray(labels, dtype=np.int64))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            optimizer.step()
    return model.eval()


def build_and_train_classifier(signals, labels, epochs: int = 10, seed: int = 0, **model_kwargs) -> DenseNet1DBiLSTMAttention:
    """Seeds weight initialization as well as training, so repeated runs
    produce the same model."""
    torch.manual_seed(seed)
    return train_classifier(DenseNet1DBiLSTMAttention(**model_kwargs), signals, labels, epochs=epochs, seed=seed)
