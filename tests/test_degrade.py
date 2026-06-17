import io
import numpy as np
import pytest
from PIL import Image

from degrade import (
    DEGRADATIONS,
    downscale,
    apply_jpeg,
    apply_gaussian_blur,
    apply_gaussian_noise,
    degrade,
)


def _solid_image(w=128, h=128, value=128):
    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8))


def test_downscale_size_bicubic():
    img = _solid_image(128, 128)
    out = downscale(img, scale=4, method='bicubic')
    assert out.size == (32, 32)


def test_downscale_size_lanczos():
    img = _solid_image(64, 64)
    out = downscale(img, scale=2, method='lanczos')
    assert out.size == (32, 32)


def test_downscale_size_bilinear():
    img = _solid_image(64, 128)
    out = downscale(img, scale=2, method='bilinear')
    assert out.size == (32, 64)


def test_apply_jpeg_returns_pil_image():
    img = _solid_image(64, 64)
    out = apply_jpeg(img, quality=10)
    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)


def test_apply_gaussian_blur_returns_pil_image():
    img = _solid_image(64, 64)
    out = apply_gaussian_blur(img, sigma=2.0)
    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)


def test_apply_gaussian_noise_range():
    img = _solid_image(64, 64, value=128)
    out = apply_gaussian_noise(img, std=5.0)
    arr = np.array(out)
    assert arr.min() >= 0 and arr.max() <= 255


def test_degrade_all_methods_correct_output_size():
    img = _solid_image(64, 64)
    for method in DEGRADATIONS:
        out = degrade(img, method=method, scale=2)
        assert out.size == (32, 32), f'Method {method!r} produced wrong size: {out.size}'


def test_degrade_invalid_method_raises():
    img = _solid_image(64, 64)
    with pytest.raises(ValueError, match='Unknown degradation'):
        degrade(img, method='invalid_method', scale=2)


def test_degrade_bicubic_pixel_values():
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    img = Image.fromarray(arr)
    out = degrade(img, method='bicubic', scale=2)
    out_arr = np.array(out)
    # Solid-color image should be preserved after bicubic downscale
    assert np.allclose(out_arr, 200, atol=1)


def test_degrade_jpeg_introduces_artifacts():
    # Non-uniform image: JPEG should change pixel values vs bicubic alone
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    lr_bicubic = np.array(degrade(img, 'bicubic', scale=2))
    lr_jpeg = np.array(degrade(img, 'jpeg', scale=2, jpeg_quality=5))
    assert not np.array_equal(lr_bicubic, lr_jpeg)


def test_degrade_noise_bicubic_differs_from_bicubic():
    rng = np.random.default_rng(0)
    arr = rng.integers(100, 200, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    lr_bicubic = np.array(degrade(img, 'bicubic', scale=2))
    lr_noise = np.array(degrade(img, 'noise_bicubic', scale=2, noise_std=20.0))
    assert not np.array_equal(lr_bicubic, lr_noise)
