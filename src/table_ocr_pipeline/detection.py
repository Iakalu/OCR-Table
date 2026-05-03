from __future__ import annotations

from PIL import Image

from .types import TableBox


class TableDetector:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "auto")
        self.threshold = float(config.get("confidence_threshold", 0.55))
        self.model_name = config.get("model_name", "microsoft/table-transformer-detection")
        self._hf_backend = None

    def predict(self, image: Image.Image) -> list[TableBox]:
        if self.backend in {"auto", "table_transformer"}:
            backend = self._load_table_transformer()
            if backend:
                return self._predict_table_transformer(image, backend)
        return self._predict_heuristic(image)

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
        except Exception:
            self._hf_backend = False
        return self._hf_backend

    def _predict_table_transformer(self, image: Image.Image, backend) -> list[TableBox]:
        torch, processor, model = backend
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target_sizes)[0]

        boxes: list[TableBox] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[int(label)].lower()
            if "table" in label_name:
                boxes.append(TableBox(tuple(int(v) for v in box.tolist()), float(score)))
        return boxes

    def _predict_heuristic(self, image: Image.Image) -> list[TableBox]:
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

