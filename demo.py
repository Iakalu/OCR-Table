from __future__ import annotations

from src.table_ocr_pipeline.utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import argparse
import sys
from pathlib import Path

from src.table_ocr_pipeline.pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.pipeline.reconstruct import write_csv, write_html, write_json
from src.table_ocr_pipeline.utils.config import load_config
from src.table_ocr_pipeline.utils.types import Cell
from src.table_ocr_pipeline.utils.utils import create_synthetic_table


SYNTHETIC_TEXT = [
    ["Product", "Q1", "Q2", "Growth"],
    ["A", "120", "160", "33%"],
    ["B", "90", "110", "22%"],
    ["C", "210", "205", "-2%"],
]


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


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

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        safe_print("No image provided -> using synthetic table")
        image_path = create_synthetic_table(output_dir / "synthetic_table.png")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    safe_print(f"\nInput image: {image_path}")
    safe_print(f"Output dir : {output_dir}")
    safe_print(f"Config     : {config_path}")
    safe_print("\nRunning Table OCR pipeline...\n")

    config = load_config(config_path)
    result = TableOCRPipeline(config).run(image_path, output_dir)

    if args.image is None and not args.no_fill_synthetic_text:
        safe_print("Filling synthetic text (demo mode)")
        fill_synthetic_text(output_dir)

    safe_print("\nDONE\n")
    for key, value in result.items():
        safe_print(f"{key:>8}: {value}")

    safe_print("\nFiles generated:")
    safe_print(f" - CSV : {result['csv']}")
    safe_print(f" - HTML: {result['html']}")
    safe_print(f" - JSON: {result['json']}")


if __name__ == "__main__":
    main()