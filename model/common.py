import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def resolve_tiled(model, lr_batch, tile_size=128, tile_overlap=8):
    """Resolves in overlapping tiles to avoid OOM on large images."""
    lr_batch = tf.cast(lr_batch, tf.float32)
    b = tf.shape(lr_batch)[0].numpy()
    h = tf.shape(lr_batch)[1].numpy()
    w = tf.shape(lr_batch)[2].numpy()

    # Infer upscaling factor from a small test tile
    th, tw = min(tile_size, h), min(tile_size, w)
    scale = tf.shape(model(lr_batch[:1, :th, :tw, :]))[1].numpy() // th

    stride = tile_size - tile_overlap
    canvas = np.zeros((b, h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((b, h * scale, w * scale, 1), dtype=np.float32)

    def _tile_starts(length):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size, stride))
        if not starts or starts[-1] + tile_size < length:
            starts.append(length - tile_size)
        return starts

    for y0 in _tile_starts(h):
        y1 = min(y0 + tile_size, h)
        for x0 in _tile_starts(w):
            x1 = min(x0 + tile_size, w)
            sr_tile = model(lr_batch[:, y0:y1, x0:x1, :]).numpy()
            canvas[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale, :] += sr_tile
            weight[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale, :] += 1.0

    sr = np.clip(np.round(canvas / weight), 0, 255).astype(np.uint8)
    return tf.constant(sr)


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def evaluate_metrics(model, dataset):
    """Returns (mean_psnr, mean_ssim) over dataset."""
    psnr_values = []
    ssim_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_values.append(psnr(hr, sr)[0])
        ssim_values.append(ssim(hr, sr)[0])
    return tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


def ssim(x1, x2):
    return tf.image.ssim(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
