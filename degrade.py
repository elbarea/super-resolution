"""
Image degradation utilities for generating LR/HR training pairs from custom images.

Supported degradation methods:
  bicubic       - bicubic downscale (default, fastest)
  bilinear      - bilinear downscale
  lanczos       - Lanczos downscale (highest quality, slowest)
  jpeg          - bicubic downscale + JPEG compression artifacts
  blur_bicubic  - Gaussian blur then bicubic downscale
  noise_bicubic - Gaussian noise then bicubic downscale
"""

import io

import numpy as np
from PIL import Image, ImageFilter

DEGRADATIONS = ['bicubic', 'bilinear', 'lanczos', 'jpeg', 'blur_bicubic', 'noise_bicubic']

_PIL_FILTERS = {
    'bicubic': Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'lanczos': Image.LANCZOS,
}


def downscale(img: Image.Image, scale: int, method: str = 'bicubic') -> Image.Image:
    """Downscale a PIL Image by an integer scale factor."""
    w, h = img.size
    return img.resize((w // scale, h // scale), _PIL_FILTERS[method])


def apply_jpeg(img: Image.Image, quality: int = 30) -> Image.Image:
    """Round-trip through JPEG at the given quality to introduce compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_gaussian_blur(img: Image.Image, sigma: float = 1.0) -> Image.Image:
    """Apply Gaussian blur with the given radius/sigma."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_gaussian_noise(img: Image.Image, std: float = 5.0) -> Image.Image:
    """Add zero-mean Gaussian noise with the given standard deviation."""
    arr = np.array(img, dtype=np.float32)
    noisy = arr + np.random.normal(0.0, std, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def degrade(img: Image.Image, method: str, scale: int,
            jpeg_quality: int = 30,
            blur_sigma: float = 1.0,
            noise_std: float = 5.0) -> Image.Image:
    """
    Apply a named degradation pipeline to produce a low-resolution image.

    Args:
        img:          Source PIL Image (HR).
        method:       One of DEGRADATIONS.
        scale:        Integer downscale factor.
        jpeg_quality: JPEG quality (1-95) for 'jpeg' method.
        blur_sigma:   Gaussian blur radius for 'blur_bicubic' method.
        noise_std:    Gaussian noise std-dev for 'noise_bicubic' method.

    Returns:
        Degraded PIL Image at 1/scale resolution.
    """
    if method == 'bicubic':
        return downscale(img, scale, 'bicubic')
    elif method == 'bilinear':
        return downscale(img, scale, 'bilinear')
    elif method == 'lanczos':
        return downscale(img, scale, 'lanczos')
    elif method == 'jpeg':
        return apply_jpeg(downscale(img, scale, 'bicubic'), quality=jpeg_quality)
    elif method == 'blur_bicubic':
        return downscale(apply_gaussian_blur(img, blur_sigma), scale, 'bicubic')
    elif method == 'noise_bicubic':
        return downscale(apply_gaussian_noise(img, noise_std), scale, 'bicubic')
    else:
        raise ValueError(f"Unknown degradation '{method}'. Choose from: {DEGRADATIONS}")
