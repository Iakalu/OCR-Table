from __future__ import annotations

import numpy as np
from PIL import Image

from .types import Cell


class StructureRecognizer:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "heuristic")
        self.checkpoint_path = config.get("checkpoint_path", "checkpoints/structure_line_cnn.pt")
        self.model_name = config.get("model_name", "microsoft/table-transformer-structure-recognition")
        self.line_threshold = float(config.get("line_threshold", 0.45))
        self.structure_threshold = float(config.get("structure_threshold", 0.50))
        self.merge_tolerance = int(config.get("merge_tolerance", 8))
        self.min_cell_width = int(config.get("min_cell_width", 24))
        self.min_cell_height = int(config.get("min_cell_height", 16))
        self._line_cnn = None
        self._table_transformer = None

    def predict(self, table_image: Image.Image) -> list[Cell]:
        if self.backend == "table_transformer":
            model = self._load_table_transformer()
            if model:
                cells = self._predict_table_transformer(table_image, model)
                if cells:
                    return cells

        if self.backend in {"line_cnn", "auto"}:
            model = self._load_line_cnn()
            if model:
                return self._predict_line_cnn(table_image, model)
        return self._predict_ruled_grid(table_image)

    def _load_table_transformer(self):
        if self._table_transformer is not None:
            return self._table_transformer
        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            self._table_transformer = (torch, processor, model, device)
        except Exception:
            self._table_transformer = False
        return self._table_transformer

    def _predict_table_transformer(self, image: Image.Image, backend) -> list[Cell]:
        torch, processor, model, device = backend
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs,
            threshold=self.structure_threshold,
            target_sizes=target_sizes,
        )[0]

        rows: list[tuple[int, int, int, int]] = []
        columns: list[tuple[int, int, int, int]] = []
        spanning: list[tuple[int, int, int, int]] = []

        for label, box in zip(results["labels"], results["boxes"]):
            label_name = model.config.id2label[int(label)].lower()
            bbox = tuple(int(v) for v in box.detach().cpu().tolist())
            if "row" in label_name:
                rows.append(bbox)
            elif "column" in label_name and "header" not in label_name:
                columns.append(bbox)
            elif "spanning" in label_name:
                spanning.append(bbox)

        rows = self._sort_and_filter_boxes(rows, axis="y")
        columns = self._sort_and_filter_boxes(columns, axis="x")
        if len(rows) < 1 or len(columns) < 1:
            return []

        cells: list[Cell] = []
        for row_idx, row_box in enumerate(rows):
            for col_idx, col_box in enumerate(columns):
                x1 = max(row_box[0], col_box[0])
                y1 = max(row_box[1], col_box[1])
                x2 = min(row_box[2], col_box[2])
                y2 = min(row_box[3], col_box[3])
                if x2 - x1 >= self.min_cell_width and y2 - y1 >= self.min_cell_height:
                    cells.append(
                        Cell(
                            row=row_idx,
                            col=col_idx,
                            bbox=(x1, y1, x2, y2),
                            meta={"backend": "table_transformer", "spanning_candidates": len(spanning)},
                        )
                    )
        return cells

    def _sort_and_filter_boxes(self, boxes: list[tuple[int, int, int, int]], axis: str) -> list[tuple[int, int, int, int]]:
        index = 1 if axis == "y" else 0
        size_index_1 = 3 if axis == "y" else 2
        sorted_boxes = sorted(boxes, key=lambda box: box[index])
        filtered: list[tuple[int, int, int, int]] = []
        for box in sorted_boxes:
            length = box[size_index_1] - box[index]
            min_length = self.min_cell_height if axis == "y" else self.min_cell_width
            if length < min_length:
                continue
            if filtered and abs(box[index] - filtered[-1][index]) <= self.merge_tolerance:
                prev = filtered[-1]
                filtered[-1] = (
                    min(prev[0], box[0]),
                    min(prev[1], box[1]),
                    max(prev[2], box[2]),
                    max(prev[3], box[3]),
                )
            else:
                filtered.append(box)
        return filtered

    def _load_line_cnn(self):
        if self._line_cnn is not None:
            return self._line_cnn
        try:
            from pathlib import Path

            import torch

            from .structure_model import load_line_segmentation_model

            if not Path(self.checkpoint_path).exists():
                self._line_cnn = False
                return self._line_cnn
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_line_segmentation_model(self.checkpoint_path, device=device)
            self._line_cnn = (torch, model, device)
        except Exception:
            self._line_cnn = False
        return self._line_cnn

    def _predict_line_cnn(self, image: Image.Image, model_backend) -> list[Cell]:
        torch, model, device = model_backend
        original_size = image.size
        model_size = 256
        gray = image.convert("L").resize((model_size, model_size))
        arr = np.asarray(gray).astype("float32") / 255.0
        x = torch.tensor(arr[None, None, :, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        vertical_score = probs[0].mean(axis=0)
        horizontal_score = probs[1].mean(axis=1)
        vertical = self._positions_from_scores(vertical_score, original_size[0])
        horizontal = self._positions_from_scores(horizontal_score, original_size[1])

        if len(vertical) < 2 or len(horizontal) < 2:
            return self._predict_ruled_grid(image)

        cells: list[Cell] = []
        for row_idx, (y1, y2) in enumerate(zip(horizontal, horizontal[1:])):
            for col_idx, (x1, x2) in enumerate(zip(vertical, vertical[1:])):
                if x2 - x1 >= self.min_cell_width and y2 - y1 >= self.min_cell_height:
                    cells.append(Cell(row=row_idx, col=col_idx, bbox=(x1, y1, x2, y2), meta={"backend": "line_cnn"}))
        return cells or self._predict_ruled_grid(image)

    def _positions_from_scores(self, scores: np.ndarray, target_len: int) -> list[int]:
        raw = np.where(scores >= self.line_threshold)[0].tolist()
        merged = self._merge_close_positions(raw)
        scale = target_len / max(1, len(scores))
        return [int(pos * scale) for pos in merged]

    def _predict_ruled_grid(self, image: Image.Image) -> list[Cell]:
        gray = image.convert("L")
        vertical_lines = self._line_positions(gray, axis="x")
        horizontal_lines = self._line_positions(gray, axis="y")
        if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
            return [Cell(row=0, col=0, bbox=(0, 0, image.width, image.height))]

        cells: list[Cell] = []
        for row_idx, (y1, y2) in enumerate(zip(horizontal_lines, horizontal_lines[1:])):
            for col_idx, (x1, x2) in enumerate(zip(vertical_lines, vertical_lines[1:])):
                if x2 - x1 >= self.min_cell_width and y2 - y1 >= self.min_cell_height:
                    cells.append(Cell(row=row_idx, col=col_idx, bbox=(x1, y1, x2, y2)))
        return cells

    def _line_positions(self, gray: Image.Image, axis: str) -> list[int]:
        width, height = gray.size
        scan_len = width if axis == "x" else height
        line_len = height if axis == "x" else width
        raw_positions: list[int] = []

        for idx in range(scan_len):
            dark = 0
            for j in range(line_len):
                pixel = gray.getpixel((idx, j)) if axis == "x" else gray.getpixel((j, idx))
                if pixel < 80:
                    dark += 1
            if dark / max(1, line_len) > 0.45:
                raw_positions.append(idx)

        return self._merge_close_positions(raw_positions)

    def _merge_close_positions(self, positions: list[int]) -> list[int]:
        if not positions:
            return []
        groups: list[list[int]] = [[positions[0]]]
        for position in positions[1:]:
            if position - groups[-1][-1] <= self.merge_tolerance:
                groups[-1].append(position)
            else:
                groups.append([position])
        return [int(sum(group) / len(group)) for group in groups]
