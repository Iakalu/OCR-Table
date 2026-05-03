from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .types import BBox


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def crop(image: Image.Image, bbox: BBox, padding: int = 0) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop(clamp_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), image.width, image.height))


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
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        header_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        header_font = font

    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            x1 = margin + col_idx * cell_w
            y1 = margin + row_idx * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            fill = "#eef4ff" if row_idx == 0 else "white"
            draw.rectangle((x1, y1, x2, y2), fill=fill, outline="black", width=2)
            draw.text((x1 + 14, y1 + 16), value, fill="black", font=header_font if row_idx == 0 else font)

    image.save(output_path)
    return output_path

