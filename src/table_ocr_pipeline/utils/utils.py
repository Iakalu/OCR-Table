from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from ..utils.types import BBox 


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox

    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def crop(image: Image.Image, bbox: BBox, padding: int = 0) -> Image.Image:
    x1, y1, x2, y2 = bbox

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    bbox = clamp_bbox((x1, y1, x2, y2), image.width, image.height)

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return image.copy()

    return image.crop(bbox)


def draw_boxes(
    image: Image.Image,
    boxes: Iterable[BBox],
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for b in boxes:
        draw.rectangle(b, outline=color, width=width)

    return img


def create_synthetic_table(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        ["Product", "Q1", "Q2", "Growth"],
        ["A", "120", "160", "33%"],
        ["B", "90", "110", "22%"],
        ["C", "210", "205", "-2%"],
    ]

    cell_w, cell_h = 150, 56
    margin = 60

    width = margin * 2 + cell_w * len(rows[0])
    height = margin * 2 + cell_h * len(rows)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
        header_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        header_font = font

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            x1 = margin + c * cell_w
            y1 = margin + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            fill = "#eef4ff" if r == 0 else "white"

            draw.rectangle((x1, y1, x2, y2), fill=fill, outline="black", width=2)

            draw.text(
                (x1 + 14, y1 + 16),
                val,
                fill="black",
                font=header_font if r == 0 else font,
            )

    img.save(output_path)

    return output_path