import os
import numpy as np
import pytest
from PIL import Image

from prepare_dataset import prepare, _find_images, _make_divisible


def _write_images(directory, count=3, size=(128, 128)):
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(count):
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        path = os.path.join(directory, f'img_{i:03}.png')
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def test_prepare_creates_hr_lr_dirs(tmp_path):
    src = str(tmp_path / 'input')
    _write_images(src, count=3, size=(128, 128))
    out = str(tmp_path / 'output')
    n = prepare(src, out, scale=4, method='bicubic')
    assert n == 3
    assert os.path.isdir(os.path.join(out, 'hr'))
    assert os.path.isdir(os.path.join(out, 'lr'))


def test_prepare_output_filenames_match(tmp_path):
    src = str(tmp_path / 'input')
    _write_images(src, count=2, size=(128, 128))
    out = str(tmp_path / 'output')
    prepare(src, out, scale=4, method='bicubic')
    hr_files = sorted(os.listdir(os.path.join(out, 'hr')))
    lr_files = sorted(os.listdir(os.path.join(out, 'lr')))
    assert hr_files == lr_files


def test_prepare_lr_is_smaller_than_hr(tmp_path):
    src = str(tmp_path / 'input')
    _write_images(src, count=1, size=(128, 128))
    out = str(tmp_path / 'output')
    prepare(src, out, scale=4, method='bicubic')
    hr_img = Image.open(os.path.join(out, 'hr', 'img_000.png'))
    lr_img = Image.open(os.path.join(out, 'lr', 'img_000.png'))
    hw, hh = hr_img.size
    lw, lh = lr_img.size
    assert hw == lw * 4 and hh == lh * 4


def test_prepare_all_degradation_methods(tmp_path):
    from degrade import DEGRADATIONS
    src = str(tmp_path / 'input')
    _write_images(src, count=1, size=(64, 64))
    for method in DEGRADATIONS:
        out = str(tmp_path / f'out_{method}')
        prepare(src, out, scale=2, method=method)
        lr = Image.open(os.path.join(out, 'lr', 'img_000.png'))
        assert lr.size == (32, 32), f'Method {method!r} produced wrong LR size'


def test_prepare_hr_size_crops_image(tmp_path):
    src = str(tmp_path / 'input')
    _write_images(src, count=1, size=(256, 256))
    out = str(tmp_path / 'output')
    prepare(src, out, scale=4, method='bicubic', hr_size=128)
    hr = Image.open(os.path.join(out, 'hr', 'img_000.png'))
    assert hr.size == (128, 128)


def test_prepare_raises_if_no_images(tmp_path):
    src = str(tmp_path / 'empty')
    os.makedirs(src)
    out = str(tmp_path / 'output')
    with pytest.raises(FileNotFoundError):
        prepare(src, out, scale=4)


def test_find_images_non_recursive(tmp_path):
    _write_images(str(tmp_path), count=2, size=(64, 64))
    subdir = os.path.join(str(tmp_path), 'sub')
    _write_images(subdir, count=1, size=(64, 64))
    found = _find_images(str(tmp_path), recursive=False)
    assert len(found) == 2


def test_find_images_recursive(tmp_path):
    _write_images(str(tmp_path), count=2, size=(64, 64))
    subdir = os.path.join(str(tmp_path), 'sub')
    _write_images(subdir, count=1, size=(64, 64))
    found = _find_images(str(tmp_path), recursive=True)
    assert len(found) == 3


def test_make_divisible_no_change():
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    out = _make_divisible(img, scale=4)
    assert out.size == (64, 64)


def test_make_divisible_trims():
    img = Image.fromarray(np.zeros((65, 67, 3), dtype=np.uint8))
    out = _make_divisible(img, scale=4)
    assert out.size == (64, 64)
