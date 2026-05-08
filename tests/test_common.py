import numpy as np
import tensorflow as tf
import pytest

from model.common import (
    normalize, denormalize,
    normalize_01, normalize_m11, denormalize_m11,
    psnr, ssim,
    resolve, resolve_single, resolve_tiled,
    evaluate, evaluate_metrics,
)


def _dummy_model(scale=2):
    """Nearest-neighbor upsampler as a trivial model for testing."""
    inp = tf.keras.Input(shape=(None, None, 3))
    out = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, [tf.shape(x)[1] * scale, tf.shape(x)[2] * scale], method='nearest')
    )(inp)
    return tf.keras.Model(inp, out)


def test_normalize_roundtrip():
    x = tf.constant([[[128., 128., 128.]]])
    assert np.allclose(denormalize(normalize(x)).numpy(), x.numpy(), atol=1.0)


def test_normalize_01():
    x = tf.constant([[[255., 0., 128.]]])
    out = normalize_01(x)
    assert out.numpy().max() <= 1.0 and out.numpy().min() >= 0.0


def test_normalize_m11_range():
    x = tf.constant([[[255., 0., 128.]]])
    out = normalize_m11(x)
    assert out.numpy().max() <= 1.0 and out.numpy().min() >= -1.0


def test_normalize_m11_roundtrip():
    x = tf.constant([[[200., 100., 50.]]])
    assert np.allclose(denormalize_m11(normalize_m11(x)).numpy(), x.numpy(), atol=1e-3)


def test_psnr_identical_images():
    img = tf.constant(np.ones((1, 16, 16, 3), dtype=np.uint8) * 128)
    val = psnr(img, img)
    assert tf.math.is_inf(val[0])


def test_psnr_different_images():
    img1 = tf.constant(np.ones((1, 16, 16, 3), dtype=np.uint8) * 100)
    img2 = tf.constant(np.ones((1, 16, 16, 3), dtype=np.uint8) * 150)
    val = psnr(img1, img2)
    assert val[0].numpy() > 0 and not tf.math.is_inf(val[0])


def test_ssim_identical_images():
    img = tf.constant(np.ones((1, 16, 16, 3), dtype=np.uint8) * 128)
    val = ssim(img, img)
    assert abs(val[0].numpy() - 1.0) < 1e-5


def test_ssim_range():
    img1 = tf.constant(np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8))
    img2 = tf.constant(np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8))
    val = ssim(img1, img2)
    assert 0.0 <= val[0].numpy() <= 1.0


def test_resolve_output_dtype():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (1, 8, 8, 3), dtype=np.uint8))
    sr = resolve(model, lr)
    assert sr.dtype == tf.uint8


def test_resolve_output_shape():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (1, 8, 8, 3), dtype=np.uint8))
    sr = resolve(model, lr)
    assert sr.shape == (1, 16, 16, 3)


def test_resolve_output_range():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (1, 8, 8, 3), dtype=np.uint8))
    sr = resolve(model, lr)
    assert sr.numpy().max() <= 255 and sr.numpy().min() >= 0


def test_resolve_single_shape():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
    sr = resolve_single(model, lr)
    assert sr.shape == (16, 16, 3)


def test_resolve_tiled_output_dtype():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (1, 32, 32, 3), dtype=np.uint8))
    sr = resolve_tiled(model, lr, tile_size=16, tile_overlap=2)
    assert sr.dtype == tf.uint8


def test_resolve_tiled_output_shape():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(0, 255, (1, 32, 32, 3), dtype=np.uint8))
    sr = resolve_tiled(model, lr, tile_size=16, tile_overlap=2)
    assert sr.shape == (1, 64, 64, 3)


def test_evaluate_returns_positive_psnr():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(100, 150, (1, 8, 8, 3), dtype=np.uint8))
    hr = tf.constant(np.random.randint(100, 150, (1, 16, 16, 3), dtype=np.uint8))
    ds = tf.data.Dataset.from_tensors((lr, hr))
    result = evaluate(model, ds)
    assert result.numpy() > 0


def test_evaluate_metrics_returns_both():
    model = _dummy_model(scale=2)
    lr = tf.constant(np.random.randint(100, 150, (1, 8, 8, 3), dtype=np.uint8))
    hr = tf.constant(np.random.randint(100, 150, (1, 16, 16, 3), dtype=np.uint8))
    ds = tf.data.Dataset.from_tensors((lr, hr))
    p, s = evaluate_metrics(model, ds)
    assert p.numpy() > 0
    assert 0.0 <= s.numpy() <= 1.0
