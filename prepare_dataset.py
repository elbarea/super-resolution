"""
Bulk dataset preparation script.

Reads all images from an input directory, generates HR/LR pairs by applying
a chosen degradation, and writes them into <output>/hr/ and <output>/lr/.

Usage:
    python prepare_dataset.py --input /path/to/images --output /path/to/dataset --scale 4
    python prepare_dataset.py -i my_photos/ -o dataset/ -s 4 -d jpeg --jpeg-quality 20
    python prepare_dataset.py -i raw/ -o dataset/ -s 2 -d blur_bicubic --recursive
"""

import argparse
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from degrade import degrade, DEGRADATIONS

_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _find_images(directory: str, recursive: bool) -> list:
    files = []
    if recursive:
        for root, _, names in os.walk(directory):
            for name in names:
                if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                files.append(os.path.join(directory, name))
    return sorted(files)


def _make_divisible(img: Image.Image, scale: int) -> Image.Image:
    """Crop image so both dimensions are divisible by scale."""
    w, h = img.size
    new_w = (w // scale) * scale
    new_h = (h // scale) * scale
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return img


def _process_one(src_path: str, hr_dir: str, lr_dir: str, scale: int,
                 method: str, jpeg_quality: int, blur_sigma: float,
                 noise_std: float, hr_size: int) -> str:
    """Process a single image: save HR and LR. Returns filename on success."""
    img = Image.open(src_path).convert('RGB')

    if hr_size:
        w, h = img.size
        short = min(w, h)
        if short < hr_size:
            factor = hr_size / short
            img = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
        # Center crop to hr_size × hr_size
        w, h = img.size
        left = (w - hr_size) // 2
        top = (h - hr_size) // 2
        img = img.crop((left, top, left + hr_size, top + hr_size))

    img = _make_divisible(img, scale)

    stem = os.path.splitext(os.path.basename(src_path))[0]
    hr_path = os.path.join(hr_dir, f'{stem}.png')
    lr_path = os.path.join(lr_dir, f'{stem}.png')

    img.save(hr_path, format='PNG')
    lr = degrade(img, method=method, scale=scale,
                 jpeg_quality=jpeg_quality, blur_sigma=blur_sigma, noise_std=noise_std)
    lr.save(lr_path, format='PNG')
    return stem


def prepare(input_dir: str, output_dir: str, scale: int = 4,
            method: str = 'bicubic', jpeg_quality: int = 30,
            blur_sigma: float = 1.0, noise_std: float = 5.0,
            hr_size: int = 0, recursive: bool = False,
            workers: int = None) -> int:
    """
    Generate HR/LR pairs from all images in input_dir.

    Returns the number of images processed.
    """
    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir = os.path.join(output_dir, 'lr')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    images = _find_images(input_dir, recursive)
    if not images:
        raise FileNotFoundError(f'No images found in {input_dir!r}')

    if workers is None:
        workers = min(os.cpu_count() or 1, len(images))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, path, hr_dir, lr_dir, scale,
                        method, jpeg_quality, blur_sigma, noise_std, hr_size): path
            for path in images
        }
        with tqdm(total=len(futures), desc='Generating dataset', unit='img') as bar:
            for fut in as_completed(futures):
                fut.result()  # re-raise any exception from the worker
                bar.update(1)

    return len(images)


def main():
    parser = argparse.ArgumentParser(
        description='Bulk HR/LR dataset generation from custom images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Directory containing source HR images.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory. Creates hr/ and lr/ subdirs.')
    parser.add_argument('-s', '--scale', type=int, default=4,
                        help='Downscale factor for LR generation.')
    parser.add_argument('-d', '--degradation', default='bicubic',
                        choices=DEGRADATIONS,
                        help='Degradation method to apply.')
    parser.add_argument('--jpeg-quality', type=int, default=30, metavar='Q',
                        help='JPEG quality (1-95) for the "jpeg" degradation.')
    parser.add_argument('--blur-sigma', type=float, default=1.0, metavar='S',
                        help='Gaussian blur radius for "blur_bicubic" degradation.')
    parser.add_argument('--noise-std', type=float, default=5.0, metavar='N',
                        help='Gaussian noise std-dev for "noise_bicubic" degradation.')
    parser.add_argument('--hr-size', type=int, default=0, metavar='PX',
                        help='Center-crop HR images to this square size before scaling. '
                             '0 = keep original dimensions.')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recurse into subdirectories of --input.')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of parallel worker threads. Defaults to cpu_count.')
    args = parser.parse_args()

    n = prepare(
        input_dir=args.input,
        output_dir=args.output,
        scale=args.scale,
        method=args.degradation,
        jpeg_quality=args.jpeg_quality,
        blur_sigma=args.blur_sigma,
        noise_std=args.noise_std,
        hr_size=args.hr_size,
        recursive=args.recursive,
        workers=args.workers,
    )
    print(f'\nDone. {n} image(s) saved to {args.output!r}  (hr/ and lr/ subdirs).')


if __name__ == '__main__':
    main()
