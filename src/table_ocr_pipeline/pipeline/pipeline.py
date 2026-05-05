from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .detection import TableDetector
from .ocr import OCRReader
from .reconstruct import cells_to_matrix, write_csv, write_html, write_json
from .structure import StructureRecognizer
from ..utils.types import Cell
from ..utils.utils import crop


class TableOCRPipeline:

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
        self._clear_previous_debug_outputs(output_dir)

        # debug OCR
        if hasattr(self.ocr, "debug_dir"):
            self.ocr.debug_dir = output_dir

        image = Image.open(image_path).convert("RGB")
        image = self._preprocess(image)


        # DETECTION
        # detect() returns list[list[int]]
        table_boxes = self.detector.detect(self._pil_to_np(image))

        all_cells: List[Cell] = []

        cell_padding = int(self.config.get("ocr", {}).get("cell_padding", 3))
        per_cell_fallback = bool(self.config.get("ocr", {}).get("per_cell_fallback", True))


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
                elif per_cell_fallback:
                    # fallback: OCR per cell
                    cell_img = crop(table_image, cell.bbox, padding=cell_padding)

                    text = self.ocr.read(self._pil_to_np(cell_img))

                    cell.text = text
                    cell.score = 0.0 if not text else 1.0
                else:
                    cell.text = ""
                    cell.score = 0.0

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

    def _clear_previous_debug_outputs(self, output_dir: Path) -> None:
        for pattern in ("ocr_debug.log", "table_*_ocr_tokens.json"):
            for path in output_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass


    # TOKEN → CELL MATCHING
    def _tokens_inside_cell(self, tokens, cell_bbox):
        x1, y1, x2, y2 = cell_bbox

        def area(b):
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        def intersection(a, b):
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            if ix2 <= ix1 or iy2 <= iy1:
                return 0
            return (ix2 - ix1) * (iy2 - iy1)

        assigned = []
        cell_area = area((x1, y1, x2, y2)) or 1

        for token in tokens:
            tx1, ty1, tx2, ty2 = token.bbox
            inter = intersection((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
            if inter <= 0:
                continue

            # Keep token if it meaningfully overlaps the cell.
            # This is more robust than using the token center for small fonts / slight bbox shifts.
            overlap = inter / min(cell_area, area((tx1, ty1, tx2, ty2)) or 1)
            if overlap >= 0.2:
                assigned.append(token)

        return sorted(assigned, key=lambda t: (t.bbox[1], t.bbox[0]))


    # UTILS
    def _pil_to_np(self, image: Image.Image):
        import numpy as np
        return np.array(image)

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Lightweight PIL-only preprocessing to improve detection/structure/OCR on
        screenshot-like tables (colored cells, small text).
        Controlled by `preprocess` config section (already present in YAMLs).
        """
        cfg = self.config.get("preprocess", {}) or {}

        # resize down if needed (keeps aspect ratio)
        max_side = int(cfg.get("max_side", 0) or 0)
        if max_side > 0:
            w, h = image.size
            scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
            if scale > 1.0:
                image = image.resize((int(w / scale), int(h / scale)), resample=Image.Resampling.LANCZOS)

        # denoise (median helps on JPEG artifacts / screenshots)
        if bool(cfg.get("denoise", False)):
            image = image.filter(ImageFilter.MedianFilter(size=3))

        # boost contrast & sharpness only when explicitly requested by config
        if bool(cfg.get("autocontrast", True)):
            image = ImageOps.autocontrast(image, cutoff=int(cfg.get("autocontrast_cutoff", 1)))
        image = ImageEnhance.Contrast(image).enhance(float(cfg.get("contrast", 1.0)))
        image = ImageEnhance.Sharpness(image).enhance(float(cfg.get("sharpness", 1.0)))

        return image
