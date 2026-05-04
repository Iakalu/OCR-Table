from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw


# STREAM JSONL
def stream_jsonl_url(url: str) -> Iterable[dict]:
    from urllib.request import urlopen

    with urlopen(url) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)


# IMAGE LOADING
def load_image_from_url(url: str) -> Image.Image | None:
    from urllib.request import urlopen

    try:
        with urlopen(url, timeout=10) as response:
            return Image.open(BytesIO(response.read())).convert("L")
    except Exception:
        return None


# MASK BUILDING
def normalize_line_position(value: float | int, length: int) -> int:
    if isinstance(value, float) and 0 <= value <= 1:
        return int(value * (length - 1))
    return int(value)


def masks_from_lines(
    vertical_lines,
    horizontal_lines,
    size: int,
    width: int = 3,
) -> np.ndarray:

    v_mask = Image.new("L", (size, size), 0)
    h_mask = Image.new("L", (size, size), 0)

    dv = ImageDraw.Draw(v_mask)
    dh = ImageDraw.Draw(h_mask)

    for x in vertical_lines:
        pos = normalize_line_position(x, size)
        dv.line((pos, 0, pos, size - 1), fill=255, width=width)

    for y in horizontal_lines:
        pos = normalize_line_position(y, size)
        dh.line((0, pos, size - 1, pos), fill=255, width=width)

    return np.stack(
        [
            np.asarray(h_mask, dtype=np.float32) / 255.0,
            np.asarray(v_mask, dtype=np.float32) / 255.0,
        ],
        axis=0,
    )


# SAMPLE BUILDERS
def sample_from_manifest_item(item: dict, image_size: int):
    image = load_image_from_url(item["image_url"])

    if image is None:
        return None

    image = image.resize((image_size, image_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0

    mask = masks_from_lines(
        item.get("vertical_lines", []),
        item.get("horizontal_lines", []),
        size=image_size,
        width=int(item.get("line_mask_width", 3)),
    )

    return arr[None, :, :], mask


# HUGGINGFACE STREAM
def stream_hf_dataset(dataset_name: str, split: str = "train", config_name: str | None = None):
    from datasets import load_dataset

    config_name = config_name or split

    try:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    except ValueError:
        ds_dict = load_dataset(dataset_name, config_name, streaming=True)
        ds = ds_dict.get(split) or next(iter(ds_dict.values()))

    yield from ds


def sample_from_hf_structure_item(item: dict, image_size: int):
    try:
        image = item["image"].convert("L")
    except Exception:
        return None

    w, h = image.size

    image = image.resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    v_lines = []
    h_lines = []

    boxes = item.get("boxes", [])
    labels = item.get("category_ids", [])

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(float, box)

        sx1 = int((x1 / max(1, w)) * image_size)
        sx2 = int((x2 / max(1, w)) * image_size)
        sy1 = int((y1 / max(1, h)) * image_size)
        sy2 = int((y2 / max(1, h)) * image_size)

        # 2 = column, 3 = row
        if int(label) == 2:
            v_lines.extend([sx1, sx2])
        elif int(label) == 3:
            h_lines.extend([sy1, sy2])

    mask = masks_from_lines(
        sorted(set(v_lines)),
        sorted(set(h_lines)),
        size=image_size,
        width=3,
    )

    return arr[None, :, :], mask


# PYTORCH DATASET
class FinTabNetDataset:

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_size: int = 256,
        max_samples: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples

        self.stream = stream_hf_dataset(dataset_name, split)

    def __iter__(self):
        count = 0

        for item in self.stream:
            sample = sample_from_hf_structure_item(item, self.image_size)

            if sample is None:
                continue

            image, mask = sample

            yield image.astype(np.float32), mask.astype(np.float32)

            count += 1

            if self.max_samples and count >= self.max_samples:
                break


# COLLATE
def collate_fn(batch):
    import torch

    images = [b[0] for b in batch]
    masks = [b[1] for b in batch]

    images = torch.tensor(images)
    masks = torch.tensor(masks)

    return images, masks