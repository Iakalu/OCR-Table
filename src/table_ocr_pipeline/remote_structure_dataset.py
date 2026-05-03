from __future__ import annotations

import json
from io import BytesIO
from typing import Iterable
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw


def stream_jsonl_url(url: str) -> Iterable[dict]:
    with urlopen(url) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def load_image_from_url(url: str) -> Image.Image:
    with urlopen(url) as response:
        return Image.open(BytesIO(response.read())).convert("L")


def normalize_line_position(value: float | int, length: int) -> int:
    if isinstance(value, float) and 0 <= value <= 1:
        return int(value * (length - 1))
    return int(value)


def masks_from_lines(
    vertical_lines: list[float | int],
    horizontal_lines: list[float | int],
    size: int,
    width: int = 3,
) -> np.ndarray:
    vertical_mask = Image.new("L", (size, size), 0)
    horizontal_mask = Image.new("L", (size, size), 0)
    draw_v = ImageDraw.Draw(vertical_mask)
    draw_h = ImageDraw.Draw(horizontal_mask)

    for x in vertical_lines:
        pos = normalize_line_position(x, size)
        draw_v.line((pos, 0, pos, size - 1), fill=255, width=width)

    for y in horizontal_lines:
        pos = normalize_line_position(y, size)
        draw_h.line((0, pos, size - 1, pos), fill=255, width=width)

    return np.stack(
        [
            np.asarray(vertical_mask).astype("float32") / 255.0,
            np.asarray(horizontal_mask).astype("float32") / 255.0,
        ],
        axis=0,
    )


def sample_from_manifest_item(item: dict, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = load_image_from_url(item["image_url"]).resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    mask = masks_from_lines(
        item.get("vertical_lines", []),
        item.get("horizontal_lines", []),
        size=image_size,
        width=int(item.get("line_mask_width", 3)),
    )
    return arr[None, :, :], mask


def stream_hf_dataset(dataset_name: str, split: str = "train", config_name: str | None = None) -> Iterable[dict]:
    from datasets import load_dataset

    config_name = config_name or split
    try:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    except ValueError:
        dataset_dict = load_dataset(dataset_name, config_name, streaming=True)
        if split in dataset_dict:
            dataset = dataset_dict[split]
        else:
            first_split = next(iter(dataset_dict.keys()))
            dataset = dataset_dict[first_split]
    yield from dataset


def sample_from_hf_structure_item(item: dict, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = item["image"].convert("L")
    width, height = image.size
    resized = image.resize((image_size, image_size))
    arr = np.asarray(resized).astype("float32") / 255.0

    vertical_positions: list[int] = []
    horizontal_positions: list[int] = []
    boxes = item.get("boxes", [])
    category_ids = item.get("category_ids", [])

    for box, category_id in zip(boxes, category_ids):
        x1, y1, x2, y2 = [float(value) for value in box]
        sx1 = int((x1 / max(1, width)) * image_size)
        sx2 = int((x2 / max(1, width)) * image_size)
        sy1 = int((y1 / max(1, height)) * image_size)
        sy2 = int((y2 / max(1, height)) * image_size)

        # katphlab/fintabnet-pubtables-full follows the TATR schema:
        # 2 = Column, 3 = Row. Boundaries are enough for this line-mask model.
        if int(category_id) == 2:
            vertical_positions.extend([sx1, sx2])
        elif int(category_id) == 3:
            horizontal_positions.extend([sy1, sy2])

    mask = masks_from_lines(
        sorted(set(vertical_positions)),
        sorted(set(horizontal_positions)),
        size=image_size,
        width=3,
    )
    return arr[None, :, :], mask
