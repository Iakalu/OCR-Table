# src/pipeline/detection.py

from __future__ import annotations

import numpy as np
from PIL import Image

from ..utils.types import TableBox


class TableDetector:

    def __init__(self, config: dict):
        self.config = config


        # Config
        self.backend = config.get("detection", {}).get("backend", "auto")
        self.threshold = float(config.get("detection", {}).get("confidence_threshold", 0.55))
        self.model_name = config.get(
            "detection", {}
        ).get("model_name", "microsoft/table-transformer-detection") 

        self._hf_backend = None


    # PUBLIC API 
    def detect(self, image_np) -> list[list[int]]:

        image = Image.fromarray(image_np)

        boxes = self._predict(image)

        return [list(b.bbox) for b in boxes]


    # INTERNAL
    def _predict(self, image: Image.Image) -> list[TableBox]:
        if self.backend in {"auto", "table_transformer"}:
            backend = self._load_table_transformer()
            if backend:
                boxes = self._predict_table_transformer(image, backend)
                if boxes:
                    return boxes
                if self.backend == "table_transformer":
                    return []

        return self._predict_heuristic(image)


    # TABLE TRANSFORMER
    def _load_table_transformer(self):
        if self._hf_backend is not None:
            return self._hf_backend

        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            model.eval()

            self._hf_backend = (torch, processor, model)

            print("[ok] Loaded Table Transformer")

        except Exception as e:
            print("[warn] Table Transformer not available:", e)
            self._hf_backend = False

        return self._hf_backend

    def _predict_table_transformer(self, image: Image.Image, backend) -> list[TableBox]:
        torch, processor, model = backend

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])

        results = processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )[0]

        boxes: list[TableBox] = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_name = model.config.id2label[int(label)].lower()

            if "table" in label_name:
                boxes.append(
                    TableBox(
                        tuple(int(v) for v in box.tolist()),
                        float(score)
                    )
                )

        return boxes


    # HEURISTIC FALLBACK
    def _predict_heuristic(self, image: Image.Image) -> list[TableBox]:
        # Step 1: try colored table
        colored = self._predict_colored_table_region(image)
        if colored:
            return colored

        # Step 2: fallback dark pixels
        gray = image.convert("L")
        pixels = gray.load()

        xs: list[int] = []
        ys: list[int] = []

        for y in range(gray.height):
            for x in range(gray.width):
                if pixels[x, y] < 80:
                    xs.append(x)
                    ys.append(y)

        if not xs:
            return [TableBox((0, 0, image.width, image.height), 0.1)]

        pad = 8

        return [
            TableBox(
                (
                    max(0, min(xs) - pad),
                    max(0, min(ys) - pad),
                    min(image.width, max(xs) + pad),
                    min(image.height, max(ys) + pad),
                ),
                0.5,
            )
        ]


    # COLOR DETECTION (EXCEL STYLE)
    def _predict_colored_table_region(self, image: Image.Image) -> list[TableBox]:
        arr = np.asarray(image.convert("RGB"))

        red = arr[:, :, 0].astype("int16")
        green = arr[:, :, 1].astype("int16")
        blue = arr[:, :, 2].astype("int16")

        mask = (blue > 120) & (green > 70) & (red < 150) & ((blue - red) > 35)

        components = self._connected_components(mask, min_pixels=80)

        if not components:
            return []

        candidates: list[tuple[int, int, int, int, int]] = []

        for pixels, x1, y1, x2, y2 in components:
            width = x2 - x1
            height = y2 - y1

            if width < 80 or height < 40:
                continue

            if width > image.width * 0.9 and height > image.height * 0.7:
                continue

            score = pixels + width * height // 100
            candidates.append((score, x1, y1, x2, y2))

        if not candidates:
            return []

        _, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])

        pad = 3

        return [
            TableBox(
                (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(image.width, x2 + pad),
                    min(image.height, y2 + pad),
                ),
                0.6,
            )
        ]


    # CONNECTED COMPONENTS
    def _connected_components(
        self,
        mask: np.ndarray,
        min_pixels: int
    ) -> list[tuple[int, int, int, int, int]]:

        height, width = mask.shape
        seen = np.zeros_like(mask, dtype=bool)

        components: list[tuple[int, int, int, int, int]] = []

        ys, xs = np.where(mask)

        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if seen[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            seen[start_y, start_x] = True

            count = 0
            min_x = max_x = start_x
            min_y = max_y = start_y

            while stack:
                y, x = stack.pop()
                count += 1

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if ny == y and nx == x:
                            continue

                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))

            if count >= min_pixels:
                components.append((count, min_x, min_y, max_x, max_y))

        return components