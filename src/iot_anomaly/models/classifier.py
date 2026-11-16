"""
Classifier head: attention mechanism applied to the CAE *encoder* bottleneck.

Fixed bugs vs original:
  BUG-13  Attention is now applied to the encoder output (compressed embedding),
          NOT to the autoencoder's decoder/reconstruction output.
  BUG-18  Custom KerasClassifier wrapper renamed to avoid shadowing ART's class.

Architecture:
    Input
      └── encoder (frozen or fine-tuned)
            └── bottleneck  [batch, time/4, filters/2]
                  └── TemporalAttention
                        └── Flatten
                              └── Dense(ReLU) + Dropout
                                    └── Dense(sigmoid)  → binary label
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    Multiply,
    Permute,
)
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator


# ── Attention block ────────────────────────────────────────────────────────────

def temporal_attention(x: tf.Tensor, name_prefix: str = "") -> tf.Tensor:
    """
    Soft temporal self-attention.

    Computes attention weights over the time axis of a Conv1D feature map
    (shape [batch, time, channels]) and returns a weighted sum.

    Parameters
    ----------
    x : tensor of shape (batch, time, channels)
    name_prefix : str — used to disambiguate layer names in combined models.
    """
    time_steps = K.int_shape(x)[1]

    # Permute to (batch, channels, time) → learn weights over time
    a = Permute((2, 1), name=f"{name_prefix}att_perm1")(x)
    a = Dense(time_steps, activation="softmax",
              name=f"{name_prefix}att_dense")(a)
    # Permute back to (batch, time, channels)
    a = Permute((2, 1), name=f"{name_prefix}att_vec")(a)
    # Element-wise multiplication: apply weights
    return Multiply(name=f"{name_prefix}att_mul")([x, a])


# ── Full classifier ────────────────────────────────────────────────────────────

def build_classifier(
    encoder: Model,
    input_shape: tuple[int, int],
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    l1_reg: float = 1e-5,
    l2_reg: float = 1e-4,
    learning_rate: float = 3e-4,
    freeze_encoder: bool = False,
) -> Model:
    """
    Build the full anomaly classification model.

    FIX BUG-13: encoder output (bottleneck) feeds into attention,
                not the autoencoder reconstruction output.

    Parameters
    ----------
    encoder : keras.Model  — the *encoder half* from build_autoencoder().
    input_shape : (n_features, 1)
    dense_units : int — hidden units in classifier head.
    dropout_rate : float
    l1_reg, l2_reg : float — L1/L2 regularisation on the dense layer.
    learning_rate : float
    freeze_encoder : bool — if True, encoder weights are not updated during
                    classifier training (useful for fine-tuning step).
    """
    if freeze_encoder:
        encoder.trainable = False

    inputs = Input(shape=input_shape, name="clf_input")

    # FIX BUG-13: pass through the *encoder* (not the full autoencoder)
    # bottleneck shape: (batch, time/4, filters/2)
    bottleneck = encoder(inputs, training=not freeze_encoder)

    # Temporal attention over the compressed embedding
    attended = temporal_attention(bottleneck, name_prefix="clf_")

    flat = Flatten(name="clf_flatten")(attended)
    hidden = Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
        name="clf_dense",
    )(flat)
    hidden = Dropout(dropout_rate, name="clf_dropout")(hidden)
    output = Dense(1, activation="sigmoid", name="clf_output")(hidden)

    model = Model(inputs, output, name="anomaly_classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ── Sklearn-compatible wrapper ─────────────────────────────────────────────────

class KerasAnomalyClassifier(BaseEstimator):
    """
    FIX BUG-18: renamed from KerasClassifier to avoid shadowing
    art.estimators.classification.KerasClassifier.

    Thin sklearn-compatible wrapper around the Keras classifier, enabling
    use with sklearn's cross_val_score and GridSearchCV / BayesSearchCV.
    """

    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model_: Model | None = None

    def fit(self, X, y, **fit_params):
        self.model_ = self.build_fn(**self.kwargs)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X) -> "np.ndarray":
        proba = self.model_.predict(X).ravel()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X) -> "np.ndarray":
        return self.model_.predict(X)

    def score(self, X, y) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def set_params(self, **params):
        self.kwargs.update(params)
        return self
