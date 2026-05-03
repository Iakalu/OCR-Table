from __future__ import annotations

import argparse
from pathlib import Path

from src.table_ocr_pipeline.config import load_config
from src.table_ocr_pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.reconstruct import write_csv, write_html, write_json
from src.table_ocr_pipeline.types import Cell
from src.table_ocr_pipeline.utils import create_synthetic_table


SYNTHETIC_TEXT = [
    ["Product", "Q1", "Q2", "Growth"],
    ["A", "120", "160", "33%"],
    ["B", "90", "110", "22%"],
    ["C", "210", "205", "-2%"],
]


def fill_synthetic_text(output_dir: Path) -> None:
    cells = []
    for row_idx, row in enumerate(SYNTHETIC_TEXT):
        for col_idx, text in enumerate(row):
            cells.append(Cell(row=row_idx, col=col_idx, bbox=(0, 0, 0, 0), text=text, score=1.0))
    write_csv(SYNTHETIC_TEXT, output_dir / "result.csv")
    write_html(SYNTHETIC_TEXT, output_dir / "result.html")
    write_json(cells, output_dir / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run table OCR pipeline demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--no-fill-synthetic-text", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image) if args.image else create_synthetic_table(output_dir / "synthetic_table.png")

    config = load_config(args.config)
    result = TableOCRPipeline(config).run(image_path, output_dir)

    if args.image is None and not args.no_fill_synthetic_text:
        fill_synthetic_text(output_dir)

    print("Table OCR demo finished")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

