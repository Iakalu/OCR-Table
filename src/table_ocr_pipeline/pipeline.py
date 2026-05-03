from __future__ import annotations

from pathlib import Path

from PIL import Image

from .detection import TableDetector
from .ocr import CellOCR
from .reconstruct import cells_to_matrix, write_csv, write_html, write_json
from .structure import StructureRecognizer
from .types import Cell
from .utils import crop


class TableOCRPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.detector = TableDetector(config.get("detection", {}))
        self.structure = StructureRecognizer(config.get("structure", {}))
        self.ocr = CellOCR(config.get("ocr", {}))

    def run(self, image_path: str | Path, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        table_boxes = self.detector.predict(image)
        all_cells: list[Cell] = []
        cell_padding = int(self.config.get("ocr", {}).get("cell_padding", 3))

        for table_idx, table_box in enumerate(table_boxes):
            table_image = crop(image, table_box.bbox)
            table_image.save(output_dir / f"table_{table_idx}.png")
            cells = self.structure.predict(table_image)
            for cell in cells:
                cell_image = crop(table_image, cell.bbox, padding=cell_padding)
                token = self.ocr.recognize_cell(cell_image)
                cell.text = token.text
                cell.score = token.score
            all_cells.extend(cells)

        matrix = cells_to_matrix(all_cells)
        write_csv(matrix, output_dir / "result.csv")
        write_html(matrix, output_dir / "result.html")
        write_json(all_cells, output_dir / "result.json")
        return {
            "tables": len(table_boxes),
            "cells": len(all_cells),
            "csv": str(output_dir / "result.csv"),
            "html": str(output_dir / "result.html"),
            "json": str(output_dir / "result.json"),
        }

