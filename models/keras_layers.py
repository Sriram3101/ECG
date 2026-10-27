"""
models/keras_layers.py

Custom Keras layers needed to load `DenseNet1D_clean.keras`. This is the
same AttentionLayer the poison-detection notebooks define inline; keep the
weight names ("att_weight", "att_bias") unchanged or saved checkpoints will
stop loading.
"""
import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def attention_weights(self, x):
        """Per-timestep softmax weights, shape (N, T, 1). Used by the
        Explainability agent for attention maps."""
        scores = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        return tf.nn.softmax(scores, axis=1)

    def call(self, x):
        return tf.reduce_sum(x * self.attention_weights(x), axis=1)

    def get_config(self):
        return super().get_config()
