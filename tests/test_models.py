"""Model architecture: shapes for every input length (BUG-24) and the encoder wiring (BUG-13)."""

import math

import numpy as np
import pytest

from iot_anomaly.models.autoencoder import build_autoencoder
from iot_anomaly.models.classifier import build_classifier

SMALL = dict(filters=8, kernel_size=3, dropout_rate=0.1)


@pytest.mark.parametrize("length", [4, 8, 11, 13, 15, 16, 23])
def test_autoencoder_reconstruction_matches_input_length(length):
    # BUG-24: lengths not divisible by 4 used to decode to a longer sequence.
    ae, encoder = build_autoencoder((length, 1), **SMALL)
    assert ae.output_shape == (None, length, 1)
    bottleneck_len = math.ceil(math.ceil(length / 2) / 2)
    assert encoder.output_shape == (None, bottleneck_len, SMALL["filters"] // 2)
    x = np.random.default_rng(0).normal(size=(4, length, 1)).astype("float32")
    assert ae.predict(x, verbose=0).shape == x.shape


def test_autoencoder_uses_mse_and_linear_output():
    # BUG-14: reconstruction of standardised features needs MSE + linear output.
    ae, _ = build_autoencoder((12, 1), **SMALL)
    assert ae.loss == "mse"
    assert ae.get_layer("reconstruction").activation.__name__ == "linear"


@pytest.mark.parametrize("length", [11, 16])
def test_classifier_outputs_probabilities(length):
    _, encoder = build_autoencoder((length, 1), **SMALL)
    clf = build_classifier(encoder, (length, 1), dense_units=4, dropout_rate=0.1)
    x = np.random.default_rng(0).normal(size=(5, length, 1)).astype("float32")
    p = clf.predict(x, verbose=0)
    assert p.shape == (5, 1)
    assert np.all((p >= 0) & (p <= 1))


def test_classifier_shares_encoder_weights():
    # BUG-13: the classifier must sit on the encoder bottleneck, so pretraining
    # the autoencoder changes the classifier's encoder weights too.
    ae, encoder = build_autoencoder((12, 1), **SMALL)
    clf = build_classifier(encoder, (12, 1), dense_units=4)
    assert clf.get_layer("encoder") is encoder
    conv = encoder.get_layer("enc_conv1")
    assert ae.get_layer("enc_conv1") is conv


@pytest.mark.parametrize("freeze", [False, True])
def test_classifier_inference_is_deterministic(freeze):
    # BUG-28: training=True was baked into the encoder call, so encoder dropout
    # stayed active at inference and predictions changed between calls.
    _, encoder = build_autoencoder((12, 1), filters=8, kernel_size=3, dropout_rate=0.5)
    clf = build_classifier(encoder, (12, 1), dense_units=4, dropout_rate=0.5,
                           freeze_encoder=freeze)
    x = np.random.default_rng(0).normal(size=(16, 12, 1)).astype("float32")
    first = clf(x, training=False).numpy()
    np.testing.assert_allclose(clf(x, training=False).numpy(), first, rtol=1e-6)
    np.testing.assert_allclose(clf.predict(x, verbose=0), first, rtol=1e-5, atol=1e-6)


def test_classifier_training_mode_still_applies_dropout():
    _, encoder = build_autoencoder((12, 1), filters=8, kernel_size=3, dropout_rate=0.5)
    clf = build_classifier(encoder, (12, 1), dense_units=4, dropout_rate=0.5)
    x = np.random.default_rng(0).normal(size=(64, 12, 1)).astype("float32")
    assert not np.allclose(clf(x, training=True).numpy(), clf(x, training=True).numpy())


def _real_size_models():
    _, encoder = build_autoencoder((11, 1), filters=112, kernel_size=6, dropout_rate=0.43)
    ae, _ = build_autoencoder((11, 1), filters=112, kernel_size=6, dropout_rate=0.43)
    clf = build_classifier(encoder, (11, 1), dense_units=43, dropout_rate=0.43,
                           l1_reg=2.9e-6, l2_reg=6.25e-4)
    return ae, clf


def test_classifier_head_applies_relu():
    # BUG-29: on tensorflow-metal the hidden ReLU was dropped in graph mode.
    # predict() runs in graph mode, so compare it with a NumPy forward pass.
    import tensorflow as tf

    _, clf = _real_size_models()
    x = np.random.default_rng(0).normal(size=(2000, 11, 1)).astype("float32")
    flat = tf.keras.Model(clf.input, clf.get_layer("clf_flatten").output)
    h = flat(x, training=False).numpy()
    w1, b1 = clf.get_layer("clf_dense").get_weights()
    w2, b2 = clf.get_layer("clf_output").get_weights()
    expected = 1 / (1 + np.exp(-(np.maximum(h @ w1 + b1, 0) @ w2 + b2)))
    np.testing.assert_allclose(clf.predict(x, verbose=0, batch_size=256), expected,
                               rtol=1e-4, atol=1e-5)


def test_gpu_graph_mode_matches_cpu():
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("no GPU: graph-mode kernels are the CPU ones")
    ae, clf = _real_size_models()
    x = np.random.default_rng(0).normal(size=(2000, 11, 1)).astype("float32")
    for model in (ae, clf):
        with tf.device("/CPU:0"):
            cpu = model(x, training=False).numpy()
        gpu = model.predict(x, verbose=0, batch_size=256)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-5, err_msg=model.name)


def test_freeze_encoder():
    _, encoder = build_autoencoder((12, 1), **SMALL)
    build_classifier(encoder, (12, 1), dense_units=4, freeze_encoder=True)
    assert encoder.trainable is False
