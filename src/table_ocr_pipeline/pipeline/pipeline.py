from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image

from .detection import TableDetector
from .reconstruct import cells_to_matrix, write_csv, write_html, write_json
from .structure import StructureRecognizer
from ..utils.types import Cell
from ..utils.utils import crop


class TableOCRPipeline:

    def _get_ocr():
        from .ocr import CellOCR
        return CellOCR

    def __init__(self, config: dict):
        self.config = config


        # COMPONENTS
        self.detector = TableDetector(config)
        self.structure = StructureRecognizer(config)
        self.ocr = OCRReader(config)


    # MAIN ENTRY
    def run(self, image_path: str | Path, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # debug OCR
        if hasattr(self.ocr, "debug_dir"):
            self.ocr.debug_dir = output_dir

        image = Image.open(image_path).convert("RGB")


        # DETECTION
        # detect() returns list[list[int]]
        table_boxes = self.detector.detect(self._pil_to_np(image))

        all_cells: List[Cell] = []

        cell_padding = int(self.config.get("ocr", {}).get("cell_padding", 3))


        # LOOP TABLES
        for table_idx, bbox in enumerate(table_boxes):
            table_image = crop(image, bbox)

            table_image.save(output_dir / f"table_{table_idx}.png")


            # STRUCTURE
            cells = self.structure.predict(table_image)


            # OCR TOKENS
            table_tokens = self.ocr.recognize_tokens(table_image)

            # DEBUG TOKENS
            if table_tokens:
                (output_dir / f"table_{table_idx}_ocr_tokens.json").write_text(
                    json.dumps(
                        [
                            {
                                "text": t.text,
                                "bbox": t.bbox,
                                "score": t.score
                            }
                            for t in table_tokens
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )


            # ASSIGN TEXT TO CELLS
            for cell in cells:
                assigned_tokens = self._tokens_inside_cell(table_tokens, cell.bbox)

                if assigned_tokens:
                    # token-based OCR
                    cell.text = " ".join(t.text for t in assigned_tokens).strip()
                    cell.score = sum(t.score for t in assigned_tokens) / len(assigned_tokens)
                else:
                    # fallback: OCR per cell
                    cell_img = crop(table_image, cell.bbox, padding=cell_padding)

                    text = self.ocr.read(self._pil_to_np(cell_img))

                    cell.text = text
                    cell.score = 0.0 if not text else 1.0

            all_cells.extend(cells)


        # RECONSTRUCT TABLE
        matrix = cells_to_matrix(all_cells)

        csv_path = output_dir / "result.csv"
        html_path = output_dir / "result.html"
        json_path = output_dir / "result.json"

        write_csv(matrix, csv_path)
        write_html(matrix, html_path)
        write_json(all_cells, json_path)

        return {
            "tables": len(table_boxes),
            "cells": len(all_cells),
            "csv": str(csv_path),
            "html": str(html_path),
            "json": str(json_path),
        }


    # TOKEN → CELL MATCHING
    def _tokens_inside_cell(self, tokens, cell_bbox):
        x1, y1, x2, y2 = cell_bbox

        assigned = []

        for token in tokens:
            tx1, ty1, tx2, ty2 = token.bbox

            # center point
            cx = (tx1 + tx2) / 2
            cy = (ty1 + ty2) / 2

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                assigned.append(token)

        # sort top-left → bottom-right
        return sorted(assigned, key=lambda t: (t.bbox[1], t.bbox[0]))


    # UTILS
    def _pil_to_np(self, image: Image.Image):
        import numpy as np
        return np.array(image)