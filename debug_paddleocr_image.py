from __future__ import annotations

from src.table_ocr_pipeline.utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def draw_ocr_boxes(image: Image.Image, result) -> Image.Image:
    draw = ImageDraw.Draw(image)

    for line in result[0] if result else []:
        if isinstance(line, dict):
            text = line.get("text") or line.get("rec_text") or ""
            score = float(line.get("score") or line.get("rec_score") or 0.0)
            box = line.get("box") or line.get("points")
        else:
            box = line[0]
            text = line[1][0]
            score = float(line[1][1])

        if not box:
            continue

        pts = np.asarray(box).reshape(-1, 2)

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1 - 10), f"{text[:20]} ({score:.2f})", fill="red")

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug PaddleOCR on one image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="debug_ocr.png")
    args = parser.parse_args()

    from paddleocr import PaddleOCR

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    print(f"\nimage: {image_path}")
    print(f"size: {image.size}\n")

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception:
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception:
            ocr = PaddleOCR(lang="en")

    result = ocr.ocr(np.array(image), cls=True)

    print("=== OCR TEXT RESULT ===\n")

    lines = result[0] if result else []

    for idx, line in enumerate(lines):
        if isinstance(line, dict):
            text = line.get("text") or line.get("rec_text") or ""
            score = float(line.get("score") or line.get("rec_score") or 0.0)
        else:
            text = line[1][0]
            score = float(line[1][1])

        print(f"{idx:03d} | {score:.2f} | {text}")

    print(f"\nTotal lines: {len(lines)}")

    debug_image = draw_ocr_boxes(image.copy(), result)

    output_path = Path(args.out)
    debug_image.save(output_path)

    print(f"\nDebug image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
