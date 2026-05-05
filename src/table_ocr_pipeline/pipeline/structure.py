# src/pipeline/structure.py

from __future__ import annotations

import numpy as np
from PIL import Image

from ..utils.types import Cell


class StructureRecognizer:

    def __init__(self, config: dict):
        cfg = config.get("structure", {})

        self.backend = cfg.get("backend", "auto")
        self.checkpoint_path = cfg.get("checkpoint_path")
        self.model_name = cfg.get(
            "model_name",
            "microsoft/table-transformer-structure-recognition"
        )

        self.line_threshold = float(cfg.get("line_threshold", 0.45))
        self.merge_tolerance = int(cfg.get("merge_tolerance", 8))

        self.min_cell_width = int(cfg.get("min_cell_width", 24))
        self.min_cell_height = int(cfg.get("min_cell_height", 16))

        self._line_cnn = None
        self._tt = None


    # PUBLIC
    def predict(self, image: Image.Image) -> list[Cell]:

        # 1. Table Transformer
        if self.backend in {"auto", "table_transformer"}:
            model = self._load_table_transformer()
            if model:
                cells = self._predict_tt(image, model)
                if cells:
                    return cells

        # 2. Line CNN
        if self.backend in {"auto", "line_cnn"}:
            model = self._load_line_cnn()
            if model:
                cells = self._predict_linecnn(image, model)
                if cells:
                    return cells

        # 3. fallback
        return self._predict_heuristic(image)


    # TABLE TRANSFORMER
    def _load_table_transformer(self):
        if self._tt is not None:
            return self._tt

        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = TableTransformerForObjectDetection.from_pretrained(self.model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            self._tt = (torch, processor, model, device)
        except Exception:
            self._tt = False

        return self._tt

    def _predict_tt(self, image: Image.Image, backend):
        torch, processor, model, device = backend

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)

        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]

        rows, cols = [], []

        for label, box in zip(results["labels"], results["boxes"]):
            name = model.config.id2label[int(label)].lower()
            box = tuple(int(v) for v in box.detach().cpu().tolist())

            if "row" in name:
                rows.append(box)
            elif "column" in name:
                cols.append(box)

        rows = self._merge_boxes(rows, axis="y")
        cols = self._merge_boxes(cols, axis="x")

        if len(rows) < 1 or len(cols) < 1:
            return []

        return self._build_cells(rows, cols)


    # LINE CNN
    def _load_line_cnn(self):
        if self._line_cnn is not None:
            return self._line_cnn

        try:
            import torch

            from ..model.structure_model import load_structure_model

            if not self.checkpoint_path:
                self._line_cnn = False
                return self._line_cnn

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = load_structure_model(str(self.checkpoint_path), device)

            self._line_cnn = (torch, model, device)

        except Exception:
            self._line_cnn = False

        return self._line_cnn

    def _predict_linecnn(self, image: Image.Image, backend):
        torch, model, device = backend

        size = 256
        gray = image.convert("L").resize((size, size))
        arr = np.array(gray).astype("float32") / 255.0

        x = torch.tensor(arr[None, None], device=device)

        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0].cpu().numpy()


        horiz = pred[0].mean(axis=1)  # rows
        vert = pred[1].mean(axis=0)   # columns

        rows = self._extract_positions(horiz, image.height)
        cols = self._extract_positions(vert, image.width)

        if len(rows) < 2 or len(cols) < 2:
            return []

        return self._build_cells_from_positions(rows, cols)


    # HEURISTIC
    def _predict_heuristic(self, image: Image.Image):
        gray = image.convert("L")

        rows = self._scan_lines(gray, axis="y")
        cols = self._scan_lines(gray, axis="x")

        if len(rows) < 2 or len(cols) < 2:
            return [Cell(row=0, col=0, bbox=(0, 0, image.width, image.height))]

        return self._build_cells_from_positions(rows, cols)


    # CORE BUILDING LOGIC
    def _build_cells(self, rows, cols):
        cells = []

        for r, row_box in enumerate(rows):
            for c, col_box in enumerate(cols):
                x1 = max(row_box[0], col_box[0])
                y1 = max(row_box[1], col_box[1])
                x2 = min(row_box[2], col_box[2])
                y2 = min(row_box[3], col_box[3])

                if x2 - x1 >= self.min_cell_width and y2 - y1 >= self.min_cell_height:
                    cells.append(Cell(row=r, col=c, bbox=(x1, y1, x2, y2)))

        return cells

    def _build_cells_from_positions(self, rows, cols):
        cells = []

        for r in range(len(rows) - 1):
            for c in range(len(cols) - 1):
                x1, x2 = cols[c], cols[c + 1]
                y1, y2 = rows[r], rows[r + 1]

                if x2 - x1 >= self.min_cell_width and y2 - y1 >= self.min_cell_height:
                    cells.append(Cell(row=r, col=c, bbox=(x1, y1, x2, y2)))

        return cells


    # UTILS
    def _extract_positions(self, scores, target_len):
        idx = np.where(scores > self.line_threshold)[0]
        idx = self._merge_positions(idx.tolist())

        scale = target_len / len(scores)

        return [int(i * scale) for i in idx]

    def _scan_lines(self, gray, axis):
        w, h = gray.size

        if axis == "x":
            scores = [
                sum(gray.getpixel((x, y)) < 80 for y in range(h)) / h
                for x in range(w)
            ]
        else:
            scores = [
                sum(gray.getpixel((x, y)) < 80 for x in range(w)) / w
                for y in range(h)
            ]

        idx = [i for i, s in enumerate(scores) if s > 0.08]

        return self._merge_positions(idx)

    def _merge_positions(self, positions):
        if not positions:
            return []

        groups = [[positions[0]]]

        for p in positions[1:]:
            if p - groups[-1][-1] <= self.merge_tolerance:
                groups[-1].append(p)
            else:
                groups.append([p])

        return [int(sum(g) / len(g)) for g in groups]

    def _merge_boxes(self, boxes, axis):
        if not boxes:
            return []

        idx = 1 if axis == "y" else 0
        boxes = sorted(boxes, key=lambda b: b[idx])

        merged = [boxes[0]]

        for box in boxes[1:]:
            if abs(box[idx] - merged[-1][idx]) <= self.merge_tolerance:
                prev = merged[-1]
                merged[-1] = (
                    min(prev[0], box[0]),
                    min(prev[1], box[1]),
                    max(prev[2], box[2]),
                    max(prev[3], box[3]),
                )
            else:
                merged.append(box)

        return merged