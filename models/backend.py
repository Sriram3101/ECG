"""
models/backend.py

One interface over the trained ECG classifier, so every agent that needs
the model (label/signal poisoning detection, diagnosis, explainability)
talks to it the same way regardless of framework.

The team's trained checkpoint is the Keras `DenseNet1D_clean.keras`
(DenseNet1D + BiLSTM + AttentionLayer) used in the poison-detection
notebooks -> `KerasBackend`. The diagnosis/explainability agents were
originally written against a PyTorch port -> `TorchBackend`, which wraps
`models.densenet_bilstm.DenseNet1DBiLSTMAttention` (or any module exposing
the same `embed` / `attention_weights` methods).

All inputs are float arrays of shape (N, 360) or (N, 360, 1); all outputs
are numpy.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

SIGNAL_LENGTH = 360


def as_batch(x: np.ndarray) -> np.ndarray:
    """(360,), (360,1), (N,360) or (N,360,1) -> float32 (N, 360)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape == (SIGNAL_LENGTH, 1)):
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.shape[1] != SIGNAL_LENGTH:
        raise ValueError(f"expected beats of length {SIGNAL_LENGTH}, got shape {np.shape(x)}")
    return arr


@runtime_checkable
class ClassifierBackend(Protocol):
    num_classes: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """(N, 360) -> (N, num_classes) softmax probabilities."""

    def embed(self, x: np.ndarray) -> np.ndarray:
        """(N, 360) -> (N, D) penultimate-layer embedding."""

    def input_gradient(self, x: np.ndarray, class_index: np.ndarray) -> np.ndarray:
        """(N, 360) -> (N, 360) d p[class_index] / d x."""

    def attention(self, x: np.ndarray) -> Optional[np.ndarray]:
        """(N, 360) -> (N, 360) attention weights resampled to the input
        length, or None if the model has no attention layer."""


def _resample_to_signal_length(weights: np.ndarray) -> np.ndarray:
    """(N, T) -> (N, 360) by linear interpolation; renormalized to sum 1."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[1] != SIGNAL_LENGTH:
        src = np.linspace(0.0, 1.0, weights.shape[1])
        dst = np.linspace(0.0, 1.0, SIGNAL_LENGTH)
        weights = np.stack([np.interp(dst, src, w) for w in weights])
    return weights / (weights.sum(axis=1, keepdims=True) + 1e-12)


class TorchBackend:
    def __init__(self, module, device: str = "cpu"):
        import torch

        self._torch = torch
        self.module = module.to(device).eval()
        self.device = device
        self.num_classes = int(getattr(module, "num_classes"))

    def _tensor(self, x: np.ndarray):
        return self._torch.tensor(as_batch(x), device=self.device).unsqueeze(1)  # (N, 1, 360)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return self._torch.softmax(self.module(self._tensor(x)), dim=1).cpu().numpy()

    def embed(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return self.module.embed(self._tensor(x)).cpu().numpy()

    def input_gradient(self, x: np.ndarray, class_index: np.ndarray) -> np.ndarray:
        xt = self._tensor(x).requires_grad_(True)
        probs = self._torch.softmax(self.module(xt), dim=1)
        idx = self._torch.as_tensor(np.asarray(class_index, dtype=np.int64), device=self.device)
        probs.gather(1, idx.view(-1, 1)).sum().backward()
        return xt.grad[:, 0, :].cpu().numpy()

    def attention(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not hasattr(self.module, "attention_weights"):
            return None
        with self._torch.no_grad():
            weights = self.module.attention_weights(self._tensor(x)).cpu().numpy()
        return _resample_to_signal_length(weights)


class KerasBackend:
    """Wraps the trained `DenseNet1D_clean.keras` checkpoint.

    The embedding is the penultimate layer's output, the same choice the
    poison-detection notebook makes (`model.layers[-2]`); pass
    `embedding_layer` to pick a specific layer instead.
    """

    def __init__(self, model, embedding_layer: Optional[str] = None):
        import tensorflow as tf

        from models.keras_layers import AttentionLayer

        self._tf = tf
        self.model = model
        self.num_classes = int(model.output_shape[-1])
        if embedding_layer:
            feature_layer = model.get_layer(embedding_layer)
        else:
            # Penultimate layer, skipping Dropout (identity at inference) so
            # the embedding is the Dense(128) output after attention pooling.
            feature_layer = next(layer for layer in reversed(model.layers[:-1])
                                 if not isinstance(layer, tf.keras.layers.Dropout))
        self._embedding_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)

        attention_layers = [layer for layer in model.layers if isinstance(layer, AttentionLayer)]
        self._attention_layer = attention_layers[0] if attention_layers else None
        self._attention_input_model = (
            tf.keras.Model(inputs=model.inputs, outputs=self._attention_layer.input)
            if self._attention_layer is not None else None
        )
        self._compile()

    @classmethod
    def load(cls, checkpoint_path: str, embedding_layer: Optional[str] = None) -> "KerasBackend":
        import tensorflow as tf

        from models.keras_layers import AttentionLayer

        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"AttentionLayer": AttentionLayer})
        return cls(model, embedding_layer=embedding_layer)

    @staticmethod
    def _keras_input(x: np.ndarray) -> np.ndarray:
        return as_batch(x)[..., None]  # (N, 360, 1)

    def _compile(self) -> None:
        """Graph-compile every model call once. Eager calls of the BiLSTM cost
        ~400 ms each and model.predict() ~80 ms of overhead; the agents make
        many small calls per beat, so this is what keeps the pipeline usable."""
        tf = self._tf
        spec = [tf.TensorSpec([None, SIGNAL_LENGTH, 1], tf.float32)]
        self._predict_fn = tf.function(lambda x: self.model(x, training=False), input_signature=spec)
        self._embed_fn = tf.function(lambda x: self._embedding_model(x, training=False), input_signature=spec)
        if self._attention_input_model is not None:
            self._hidden_fn = tf.function(lambda x: self._attention_input_model(x, training=False),
                                          input_signature=spec)

        @tf.function(input_signature=spec + [tf.TensorSpec([None], tf.int32)])
        def gradient_fn(x, idx):
            with tf.GradientTape() as tape:
                tape.watch(x)
                selected = tf.gather(self.model(x, training=False), idx, axis=1, batch_dims=1)
            return tape.gradient(selected, x)

        self._gradient_fn = gradient_fn

    _BATCH = 1024

    def _run(self, fn, x: np.ndarray) -> np.ndarray:
        inputs = self._keras_input(x)
        return np.concatenate([fn(inputs[i:i + self._BATCH]).numpy()
                               for i in range(0, len(inputs), self._BATCH)])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._run(self._predict_fn, x)

    def embed(self, x: np.ndarray) -> np.ndarray:
        emb = self._run(self._embed_fn, x)
        return emb.reshape(len(emb), -1)

    def input_gradient(self, x: np.ndarray, class_index: np.ndarray) -> np.ndarray:
        grads = self._gradient_fn(self._keras_input(x), np.asarray(class_index, dtype=np.int32))
        return grads.numpy()[..., 0]

    def attention(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._attention_layer is None:
            return None
        hidden = self._run(self._hidden_fn, x)  # (N, T, F)
        weights = self._attention_layer.attention_weights(hidden).numpy()  # (N, T, 1)
        return _resample_to_signal_length(weights[..., 0])
