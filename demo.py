from __future__ import annotations

import argparse
from pathlib import Path

from src.table_ocr_pipeline.utils.config import load_config
from src.table_ocr_pipeline.pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.pipeline.reconstruct import write_csv, write_html, write_json
from src.table_ocr_pipeline.utils.types import Cell
from src.table_ocr_pipeline.utils.utils import create_synthetic_table


SYNTHETIC_TEXT = [
    ["Product", "Q1", "Q2", "Growth"],
    ["A", "120", "160", "33%"],
    ["B", "90", "110", "22%"],
    ["C", "210", "205", "-2%"],
]


def fill_synthetic_text(output_dir: Path) -> None:
    cells = []

    for r, row in enumerate(SYNTHETIC_TEXT):
        for c, text in enumerate(row):
            cells.append(
                Cell(
                    row=r,
                    col=c,
                    bbox=(0, 0, 0, 0),
                    text=text,
                    score=1.0,
                )
            )

    write_csv(SYNTHETIC_TEXT, output_dir / "result.csv")
    write_html(SYNTHETIC_TEXT, output_dir / "result.html")
    write_json(cells, output_dir / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Table OCR Pipeline Demo")

    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--output-dir", default="outputs/demo", help="Output directory")
    parser.add_argument("--no-fill-synthetic-text", action="store_true", help="Disable synthetic text filling")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    if args.image:
        image_path = Path(args.image)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        print("No image provided -> using synthetic table")
        image_path = create_synthetic_table(output_dir / "synthetic_table.png")

    print(f"\nInput image: {image_path}")
    print(f"Output dir : {output_dir}")


    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    print(f"Config     : {config_path}")

    print("\nRunning Table OCR pipeline...\n")

    pipeline = TableOCRPipeline(config)
    result = pipeline.run(image_path, output_dir)


    if args.image is None and not args.no_fill_synthetic_text:
        print("Filling synthetic text (demo mode)")
        fill_synthetic_text(output_dir)

    print("\nDONE\n")

    for key, value in result.items():
        print(f"{key:>8}: {value}")

    print("\nFiles generated:")
    print(f" - CSV : {result['csv']}")
    print(f" - HTML: {result['html']}")
    print(f" - JSON: {result['json']}")


if __name__ == "__main__":
    main()